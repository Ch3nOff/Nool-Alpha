"""
4.5-Hour Deep Reasoning & Thought-Chain SFT Engine for Nool-Alpha-100M.

Features:
  - 4.5-hour strict training budget (16,200 seconds) with graceful final checkpointing.
  - Multi-stream randomized reasoning dataset with anti-memorization shuffle buffers.
  - Loss masking on instruction prompts, backpropagation active on <think> & answers.
  - Periodic reasoning probes (Math, Code, DeepSeek-style logic).
  - AdamW with warmup and cosine decay.
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM
from nool_alpha.reasoning_dataset import RandomizedReasoningSFTDataset

# Safe unpickling compatibility
if hasattr(sys.modules["__main__"], "NoolAlphaConfig") is False:
    setattr(sys.modules["__main__"], "NoolAlphaConfig", NoolAlphaConfig)

try:
    torch.serialization.add_safe_globals([NoolAlphaConfig])
except Exception:
    pass

REASONING_PROBES = [
    (
        "Math",
        "A store offers a 20% discount on a $150 jacket. If the sales tax is 8%, what is the final price?",
    ),
    (
        "Code",
        "Write a Python function `find_first_duplicate(lst)` that returns the first element that appears twice.",
    ),
    (
        "Logic",
        "Explain step-by-step why the sum of two odd integers is always an even integer.",
    ),
]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_reasoning_probe(model, tokenizer, device, prompt: str, max_tokens: int = 60) -> str:
    model.eval()
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    enc = tokenizer.encode(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        out = enc
        for _ in range(max_tokens):
            logits, _, _ = model(out)
            next_logit = logits[:, -1, :] / 0.6
            # Repetition penalty
            for prev in set(out[0].tolist()):
                if next_logit[0, prev] > 0:
                    next_logit[0, prev] /= 1.15
                else:
                    next_logit[0, prev] *= 1.15

            probs = torch.softmax(next_logit, dim=-1)
            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.85
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok], dim=-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break

    completion = tokenizer.decode(out[0][enc.shape[1] :], skip_special_tokens=True)
    return completion.strip()


def train_reasoning_sft(
    checkpoint_path: Optional[str] = None,
    output_dir: str = "checkpoints/reasoning_sft",
    total_hours: float = 4.5,
    max_steps: int = 5000,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 1.2e-4,
    min_lr: float = 1.0e-5,
    warmup_steps: int = 100,
    max_seq_len: int = 512,
    device_name: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"[+] Initializing Deep Reasoning SFT on: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model configuration
    config = NoolAlphaConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=10,
        num_heads=12,
        head_dim=64,
        d_c=192,
        d_pe=32,
        num_experts=8,
        top_k_experts=2,
        expert_rank=96,
        highway_alpha_init=0.05,
        logit_soft_cap=30.0,
    )
    model = NoolAlphaForCausalLM(config).to(device)

    # Load initial checkpoint if provided
    base_step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[+] Loading weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Filter matching keys
        model_dict = model.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        base_step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
        print(f"[OK] Loaded {len(matched)} matching tensors. Base step: {base_step}")

    # Dataset with 3-layer anti-memorization randomization
    print("[+] Building 7-Dataset Streaming Pipeline with Anti-Memorization Shuffle Buffers...")
    dataset = RandomizedReasoningSFTDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        buffer_size=10000,
        enable_random_skip=True,
        max_skip_offset=3000,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    # Optimizer & Scheduler
    no_decay = ["bias", "norm.weight", "scale"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    min_ratio = min_lr / learning_rate
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio=min_ratio)

    max_seconds = int(total_hours * 3600)
    start_time = time.time()
    best_loss = float("inf")
    accum_loss = 0.0
    accum_aux = 0.0
    tokens_trained = 0
    step = 0

    print("=" * 70)
    print(f"[*] Deep Reasoning SFT Active! Budget: {total_hours:.1f} Hours ({max_seconds}s)")
    print(f"[*] Target Steps: {max_steps} | Effective Batch Size: {batch_size * grad_accum_steps}")
    print("=" * 70)

    model.train()
    optimizer.zero_grad()
    data_iter = iter(loader)

    while step < max_steps:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\n[!] Reached time budget limit of {total_hours}h ({elapsed:.1f}s). Finalizing run gracefully.")
            break

        # Accumulate gradients
        batch_loss = 0.0
        batch_aux = 0.0
        accum_toks = 0

        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _, aux_loss = model(input_ids, attention_mask=attention_mask)
            # Standard causal LM cross-entropy with ignore_index=-100
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss = ce_loss + (aux_loss if aux_loss is not None else 0.0)

            scaled_loss = total_loss / grad_accum_steps
            scaled_loss.backward()

            batch_loss += ce_loss.item() / grad_accum_steps
            if aux_loss is not None:
                batch_aux += aux_loss.item() / grad_accum_steps

            # Count active tokens trained
            active_t = (shift_labels != -100).sum().item()
            accum_toks += active_t

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        tokens_trained += accum_toks
        accum_loss += batch_loss
        accum_aux += batch_aux

        # Logging every 25 steps
        if step % 25 == 0:
            avg_loss = accum_loss / 25
            avg_aux = accum_aux / 25
            cur_lr = scheduler.get_last_lr()[0]
            elapsed_sec = time.time() - start_time
            tok_speed = tokens_trained / max(elapsed_sec, 1e-4)
            rem_sec = max(0, max_seconds - elapsed_sec)

            print(
                f"Step {step:5d}/{max_steps} | Loss: {avg_loss:.4f} (Aux: {avg_aux:.4f}) | "
                f"LR: {cur_lr:.2e} | Speed: {tok_speed:.0f} tok/s | "
                f"Tokens: {tokens_trained/1e6:.2f}M | Remaining: {rem_sec/3600:.2f}h"
            )

            # Checkpoint best
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(output_dir, "best_reasoning_checkpoint.pt")
                torch.save(
                    {
                        "step": step,
                        "loss": best_loss,
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                    },
                    best_path,
                )

            accum_loss = 0.0
            accum_aux = 0.0

        # Periodic reasoning probes every 150 steps
        if step % 150 == 0:
            print("\n" + "=" * 50)
            print(f"🎯 [Reasoning Probes @ Step {step}]")
            for domain, probe_prompt in REASONING_PROBES:
                resp = run_reasoning_probe(model, tokenizer, device, probe_prompt, max_tokens=50)
                print(f"[{domain}] {resp[:120]}...")
            print("=" * 50 + "\n")
            model.train()

        # Regular checkpoint every 300 steps
        if step % 300 == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "loss": batch_loss,
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                },
                ckpt_path,
            )

    # Save final model
    final_path = os.path.join(output_dir, "nool_alpha_100m_reasoning_final.pt")
    torch.save(
        {
            "step": step,
            "loss": best_loss,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
        },
        final_path,
    )
    print(f"\n[OK] Deep Reasoning SFT Complete! Final Checkpoint: {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nool-Alpha-100M Deep Reasoning SFT.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Initial checkpoint path to continue from")
    parser.add_argument("--output_dir", type=str, default="checkpoints/reasoning_sft", help="Save directory")
    parser.add_argument("--hours", type=float, default=4.5, help="Training time budget in hours (default: 4.5)")
    parser.add_argument("--steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Peak learning rate")
    args = parser.parse_args()

    train_reasoning_sft(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        total_hours=args.hours,
        max_steps=args.steps,
        learning_rate=args.lr,
    )
