"""
Nool-Alpha-100M: Resume Pre-Training Script
Resumes training seamlessly from an existing checkpoint on Kaggle or local GPU.
Supports multi-domain streaming: English, Indonesian, and Python Code.
"""

import os
import glob
import time
import math
import json
import argparse
from typing import Optional, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM
from nool_alpha.dataset import MultilingualCodeStreamingDataset


def find_checkpoint(user_path: Optional[str] = None) -> Optional[str]:
    """Auto-detects checkpoint from common Kaggle input paths or local paths."""
    candidate_paths = [
        user_path,
        "/kaggle/input/datasets/chenstillstude/nool-cp/best_checkpoint.pt",
        "/kaggle/input/datasets/chenstillstude/nool-cp/nool_alpha_100m_final.pt",
        "/kaggle/input/nool-cp/best_checkpoint.pt",
        "/kaggle/input/nool-cp/nool_alpha_100m_final.pt",
        "/kaggle/working/checkpoints/best_checkpoint.pt",
        "/kaggle/working/checkpoints/nool_alpha_100m_final.pt",
        "./checkpoints/best_checkpoint.pt",
        "./checkpoints/nool_alpha_100m_final.pt",
    ]

    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.isfile(p):
            return p

    # Fuzzy search inside /kaggle/input/
    kaggle_matches = glob.glob("/kaggle/input/**/best_checkpoint.pt", recursive=True) + \
                     glob.glob("/kaggle/input/**/nool_alpha*final*.pt", recursive=True)
    if kaggle_matches:
        return kaggle_matches[0]

    return None


def generate_with_penalty(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: str,
    max_tokens: int = 50,
    temperature: float = 0.5,
    top_p: float = 0.85,
    repetition_penalty: float = 1.2,
) -> str:
    """Enhanced generation with repetition penalty and nucleus filtering."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            idx = input_ids[:, -model.config.max_position_embeddings:] if input_ids.shape[1] > model.config.max_position_embeddings else input_ids
            logits, _, _, _ = model(idx)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Apply repetition penalty
            for token_id in set(input_ids[0].tolist()):
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

            # Top-P Nucleus Filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def resume_train(
    checkpoint_path: Optional[str] = None,
    output_dir: str = "/kaggle/working/checkpoints",
    max_training_hours: float = 3.5,
    target_max_steps: int = 6000,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    eval_interval: int = 250,
    log_interval: int = 25,
    device: Optional[str] = None,
):
    start_time = time.time()
    max_seconds = max_training_hours * 3600.0
    os.makedirs(output_dir, exist_ok=True)

    # 1. Device and Precision
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device.startswith("cuda")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using CUDA GPU: {gpu_name}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[Precision] Mixed precision dtype: {dtype}")
    else:
        print("[Device] Running on CPU")
        dtype = torch.float32

    # 2. Locate Checkpoint
    ckpt_file = find_checkpoint(checkpoint_path)
    if not ckpt_file:
        raise FileNotFoundError(
            "Could not find checkpoint! Please ensure your Kaggle dataset is attached, "
            "e.g. at '/kaggle/input/datasets/chenstillstude/nool-cp/best_checkpoint.pt'"
        )

    print(f"\n========================================================")
    print(f" [Resume] Found Checkpoint: {ckpt_file}")
    print(f"========================================================")
    try:
        checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_file, map_location=device)
    saved_step = checkpoint.get("step", 1750)
    saved_loss = checkpoint.get("val_loss", checkpoint.get("loss", "N/A"))
    print(f" [Resume] Checkpoint step: {saved_step} | Previous loss: {saved_loss}")

    # 3. Model & Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load configuration
    config = checkpoint.get("config", NoolAlphaConfig.nool_100m(vocab_size=len(tokenizer)))
    config.vocab_size = len(tokenizer)

    model = NoolAlphaForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    total_p, active_p = model.get_num_params()
    print(f"[Model] Nool-Alpha-100M loaded successfully!")
    print(f"[Model] Total params: {total_p / 1e6:.2f}M | Active params: {active_p / 1e6:.2f}M")

    # 4. Immediate Voice Test (Sanity Check before training)
    print("\n--- [Sanity Check: Current Model Voice Before Phase 2] ---")
    test_p = "Ibu kota Nusantara (IKN) merupakan pusat pemerintahan baru"
    initial_sample = generate_with_penalty(model, tokenizer, test_p, device, max_tokens=35, temperature=0.5)
    print(f"Prompt: {test_p}")
    print(f"Output: {initial_sample}")
    print("----------------------------------------------------------\n")

    # 5. Multi-Domain Streaming Dataset (40% EN, 40% ID, 20% Code)
    print("[Dataset] Initializing streaming multi-domain dataloader (EN 40% + ID 40% + Code 20%)...")
    dataset = MultilingualCodeStreamingDataset(
        en_dataset="HuggingFaceFW/fineweb-edu",
        en_config="sample-10BT",
        id_dataset="wikimedia/wikipedia",
        id_config="20231101.id",
        code_dataset="iamtarun/python_code_instructions_18k_alpaca",
        domain_weights={"en": 0.40, "id": 0.40, "code": 0.20},
        tokenizer_name="gpt2",
        seq_len=512,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size)

    # 6. Optimizer & Continuation LR Scheduler
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    # Cosine scheduler for Phase 2 (from saved_step to target_max_steps)
    phase2_steps = max(100, target_max_steps - saved_step)
    phase2_warmup = 100

    def lr_schedule_phase2(step_offset: int):
        # step_offset starts at 0 for this session
        if step_offset < phase2_warmup:
            return float(step_offset) / float(max(1, phase2_warmup))
        prog = float(step_offset - phase2_warmup) / float(max(1, phase2_steps - phase2_warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = LambdaLR(optimizer, lr_schedule_phase2)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and dtype == torch.float16))

    # Multi-domain generation probes
    probe_prompts = [
        ("🇬🇧 English", "Artificial intelligence will transform the future of"),
        ("🇮🇩 Indonesian", "Ibu kota Nusantara (IKN) merupakan pusat pemerintahan baru"),
        ("💻 Python Code", "def quick_sort(arr):\n    # Implement quicksort in python\n"),
    ]

    # 7. Training Continuation Loop
    current_step = saved_step
    session_step = 0
    total_tokens_seen = 0
    running_loss = 0.0
    running_aux = 0.0
    best_phase2_loss = float("inf")
    history = []

    train_iter = iter(train_loader)
    print(f"🚀 Resuming training from step {current_step} -> Target: {target_max_steps} steps")
    print(f"⏱️ Time budget: {max_training_hours:.1f} hours ({max_seconds:.0f}s)...")
    model.train()
    optimizer.zero_grad()

    while current_step < target_max_steps:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\n⏱️ Time budget reached ({max_training_hours:.2f}h). Saving progress safely!")
            break

        accum_loss = 0.0
        accum_aux = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except Exception:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", dtype=dtype, enabled=use_cuda):
                _, loss, aux_loss, _ = model(input_ids, labels=labels)
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()
            accum_loss += scaled_loss.item()
            accum_aux += (aux_loss.item() / grad_accum_steps)
            total_tokens_seen += input_ids.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        current_step += 1
        session_step += 1
        running_loss += accum_loss
        running_aux += accum_aux

        # Logging
        if session_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_aux = running_aux / log_interval
            curr_lr = scheduler.get_last_lr()[0]
            tok_sec = total_tokens_seen / (time.time() - start_time)
            hrs_left = max(0.0, (max_seconds - (time.time() - start_time)) / 3600.0)

            print(
                f"Step {current_step:5d}/{target_max_steps} (+{session_step}) | "
                f"Loss: {avg_loss:.4f} (Aux: {avg_aux:.4f}) | "
                f"LR: {curr_lr:.2e} | "
                f"Speed: {tok_sec:.0f} tok/s | "
                f"Phase 2 Tokens: {total_tokens_seen/1e6:.2f}M | "
                f"Left: {hrs_left:.2f}h"
            )
            history.append({
                "step": current_step,
                "session_step": session_step,
                "loss": avg_loss,
                "aux_loss": avg_aux,
                "lr": curr_lr,
                "tokens": total_tokens_seen,
            })
            running_loss = 0.0
            running_aux = 0.0

        # Periodic Generation Probes & Best Checkpoint
        if session_step % eval_interval == 0:
            print(f"\n🔍 --- [Phase 2 Generation Probes @ Step {current_step}] ---")
            model.eval()
            for domain_label, prompt in probe_prompts:
                gen_text = generate_with_penalty(
                    model, tokenizer, prompt, device,
                    max_tokens=35, temperature=0.5, top_p=0.85, repetition_penalty=1.2
                )
                print(f"  [{domain_label}] {gen_text.strip()}")
            print("----------------------------------------------------\n")

            # Save best checkpoint of Phase 2
            if avg_loss < best_phase2_loss:
                best_phase2_loss = avg_loss
                best_path = os.path.join(output_dir, "best_checkpoint.pt")
                torch.save({
                    "step": current_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                }, best_path)
                print(f"💾 Saved new best checkpoint to {best_path} (Loss: {avg_loss:.4f})")

            model.train()

    # Final Save of Phase 2
    final_path = os.path.join(output_dir, "nool_alpha_100m_final.pt")
    torch.save({
        "step": current_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "loss": avg_loss if 'avg_loss' in locals() else saved_loss,
    }, final_path)
    print(f"\n🏁 Phase 2 training complete! Final checkpoint saved to: {final_path}")

    # Save metrics
    metrics_file = os.path.join(output_dir, "phase2_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📊 Training metrics saved to {metrics_file}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Nool-Alpha-100M pre-training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--hours", type=float, default=3.5, help="Max training time in hours")
    parser.add_argument("--target_steps", type=int, default=6000, help="Target total steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Phase 2 learning rate")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/checkpoints", help="Output directory")
    args = parser.parse_args()

    resume_train(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_training_hours=args.hours,
        target_max_steps=args.target_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
    )
