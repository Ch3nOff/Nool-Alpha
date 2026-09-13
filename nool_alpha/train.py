import os
import time
import math
import json
import argparse
from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import NoolAlphaConfig
from .model import NoolAlphaForCausalLM
from .dataset import get_dataloaders


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.1):
    """Cosine learning rate decay with linear warmup."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * coeff

    return LambdaLR(optimizer, lr_lambda)


def train(
    config: Optional[NoolAlphaConfig] = None,
    dataset_name: str = "roneneldan/TinyStories",
    dataset_config: Optional[str] = None,
    tokenizer_name: str = "gpt2",
    output_dir: str = "./checkpoints",
    max_training_hours: float = 3.5,
    max_steps: int = 15000,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 200,
    eval_interval: int = 250,
    eval_steps: int = 20,
    save_interval: int = 500,
    log_interval: int = 20,
    device: Optional[str] = None,
):
    """
    Time-budgeted training loop designed for 3-5 hours runs on Kaggle.
    Monitors elapsed wall-clock time and guarantees safe checkpointing and generation probes.
    """
    start_time = time.time()
    max_training_seconds = max_training_hours * 3600.0
    os.makedirs(output_dir, exist_ok=True)

    # 1. Device and Precision Configuration
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    use_cuda = device.startswith("cuda")
    if use_cuda:
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using CUDA: {device_name}")
        # Check for bfloat16 support
        has_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if has_bf16 else torch.float16
        print(f"[Precision] Using mixed precision dtype: {dtype}")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print("[Device] Running on CPU")
        dtype = torch.float32

    # 2. Dataset & Tokenizer
    print(f"[Dataset] Loading streaming dataset '{dataset_name}' with tokenizer '{tokenizer_name}'...")
    train_loader, val_loader, tokenizer = get_dataloaders(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_name=tokenizer_name,
        seq_len=config.sliding_window if config and config.sliding_window else 512,
        batch_size=batch_size,
    )
    
    # Update vocab_size from tokenizer
    actual_vocab_size = len(tokenizer)
    if config is None:
        config = NoolAlphaConfig.kaggle_3_5h(vocab_size=actual_vocab_size)
    else:
        config.vocab_size = actual_vocab_size

    # 3. Instantiate Model
    print(f"[Model] Initializing Nool-Alpha (GSLA + HFK-MoE)...")
    model = NoolAlphaForCausalLM(config).to(device)
    total_params, active_params = model.get_num_params()
    print(f"[Model] Total Parameters:  {total_params / 1e6:.2f}M")
    print(f"[Model] Active Parameters: {active_params / 1e6:.2f}M ({active_params / total_params * 100:.1f}%)")
    print(f"[Model] Highway Alpha Init: {config.highway_alpha_init} (tanh bounds direct embedding path)")

    # 4. Optimizer & Scheduler
    # Separate decay / no-decay parameters
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and dtype == torch.float16))

    # 5. Training Loop State
    step = 0
    total_tokens_seen = 0
    running_loss = 0.0
    running_aux_loss = 0.0
    best_val_loss = float("inf")
    train_history = []
    
    train_iter = iter(train_loader)
    prompt_examples = [
        "Once upon a time, there was a little",
        "The girl found a shiny key in the",
        "Deep inside the magical forest, the",
    ]

    print(f"\n[Training] Starting run with time budget of {max_training_hours:.2f} hours ({max_training_seconds:.0f}s)...")
    model.train()
    optimizer.zero_grad()

    while step < max_steps:
        step_start_time = time.time()
        elapsed_time = step_start_time - start_time
        
        # Check time budget
        if elapsed_time >= max_training_seconds:
            print(f"\n[Time Budget] Reached allocated budget of {max_training_hours:.2f} hours. Wrapping up safely!")
            break

        # Accumulate gradients over grad_accum_steps
        accum_loss = 0.0
        accum_aux = 0.0

        for accum_idx in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except (StopIteration, Exception):
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", dtype=dtype, enabled=use_cuda):
                logits, loss, aux_loss, _ = model(input_ids=input_ids, labels=labels)
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()
            accum_loss += scaled_loss.item()
            accum_aux += (aux_loss.item() / grad_accum_steps)
            total_tokens_seen += input_ids.numel()

        # Gradient clipping and optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        running_loss += accum_loss
        running_aux_loss += accum_aux

        # Logging
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_aux = running_aux_loss / log_interval
            current_lr = scheduler.get_last_lr()[0]
            tokens_per_sec = total_tokens_seen / (time.time() - start_time)
            remaining_hrs = max(0.0, (max_training_seconds - (time.time() - start_time)) / 3600.0)

            print(
                f"Step {step:5d}/{max_steps} | "
                f"Loss: {avg_loss:.4f} (Aux: {avg_aux:.4f}) | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {tokens_per_sec:.0f} tok/s | "
                f"Tokens: {total_tokens_seen / 1e6:.2f}M | "
                f"Time Left: {remaining_hrs:.2f}h"
            )
            train_history.append({
                "step": step,
                "loss": avg_loss,
                "aux_loss": avg_aux,
                "lr": current_lr,
                "tokens_seen": total_tokens_seen,
                "elapsed_seconds": time.time() - start_time,
            })
            running_loss = 0.0
            running_aux_loss = 0.0

        # Periodic Validation & Generation Probe
        if step % eval_interval == 0:
            print(f"\n--- [Validation @ Step {step}] ---")
            model.eval()
            val_loss = 0.0
            val_iter = iter(val_loader)
            with torch.no_grad():
                for _ in range(eval_steps):
                    try:
                        vbatch = next(val_iter)
                    except (StopIteration, Exception):
                        val_iter = iter(val_loader)
                        vbatch = next(val_iter)
                    v_ids = vbatch["input_ids"].to(device)
                    v_labels = vbatch["labels"].to(device)
                    with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", dtype=dtype, enabled=use_cuda):
                        _, v_loss, _, _ = model(v_ids, labels=v_labels)
                    val_loss += v_loss.item()
            val_loss /= eval_steps
            val_ppl = math.exp(min(val_loss, 20.0))
            print(f"Validation Loss: {val_loss:.4f} | Validation Perplexity: {val_ppl:.2f}")

            # Generation Probe
            print("\n[Generation Probes]")
            for prompt in prompt_examples:
                enc = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                gen = model.generate(enc, max_new_tokens=30, temperature=0.8, top_p=0.9)
                text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
                print(f"  > \"{text.strip()}\"")
            print("---------------------------------\n")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(output_dir, "best_checkpoint.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                }, best_ckpt_path)
                print(f"Saved new best model checkpoint to {best_ckpt_path} (val_loss: {val_loss:.4f})")

            model.train()

        # Regular Checkpoint Save
        if step % save_interval == 0:
            ckpt_path = os.path.join(output_dir, "latest_checkpoint.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)

    # Final Save and Metrics Export
    final_ckpt_path = os.path.join(output_dir, "nool_alpha_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "total_tokens_seen": total_tokens_seen,
    }, final_ckpt_path)
    print(f"\n[Finished] Final checkpoint saved to {final_ckpt_path}")

    # Write training metrics to JSON
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_history, f, indent=2)
    print(f"[Finished] Training metrics saved to {metrics_path}")

    return model, train_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nool-Alpha on Hugging Face dataset")
    parser.add_argument("--hours", type=float, default=3.5, help="Max training time in hours")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories", help="Hugging Face dataset name")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Hugging Face tokenizer name")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    args = parser.parse_args()

    cfg = NoolAlphaConfig.kaggle_3_5h()
    train(
        config=cfg,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        max_training_hours=args.hours,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
