"""
Training Engine for Nool-Alpha-1.5B Knowledge & Reasoning Distillation.
Features:
  - Supports 8-bit AdamW (bitsandbytes) for ultra-low VRAM memory footprint.
  - Gradient checkpointing across all 24 Nool-Alpha layers.
  - Multi-stream teacher distillation dataset.
  - Periodic reasoning probes across Math, Code, Logic, and Bilingual Chat.
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure repo root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM
from nool_alpha.distill_dataset import MemorySafeDistillDataset

# Safe unpickling compatibility
if hasattr(sys.modules["__main__"], "NoolAlphaConfig") is False:
    setattr(sys.modules["__main__"], "NoolAlphaConfig", NoolAlphaConfig)

try:
    torch.serialization.add_safe_globals([NoolAlphaConfig])
except Exception:
    pass

PROBES_1_5B = [
    (
        "Math Reasoning",
        "A store offers a 20% discount on a $150 jacket. If sales tax is 8%, what is the final price? Calculate step by step.",
    ),
    (
        "Python Code",
        "Write a Python function `find_first_duplicate(lst)` that returns the first element that appears twice in a list.",
    ),
    (
        "Scientific Logic",
        "Explain step-by-step why the sum of two odd integers is always an even integer.",
    ),
    (
        "Bilingual Indonesian",
        "Selamat siang! Bagaimana cara kerja panel surya dalam mengubah energi matahari menjadi listrik?",
    ),
]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_probe(model, tokenizer, device, prompt: str, max_tokens: int = 80) -> str:
    model.eval()
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    enc = tokenizer.encode(formatted, return_tensors="pt").to(device)
    prompt_len = enc.shape[1]

    with torch.no_grad():
        out = enc
        for _ in range(max_tokens):
            idx = out[:, -1024:] if out.shape[1] > 1024 else out
            logits, _, _, _ = model(idx)
            next_logit = logits[:, -1, :] / 0.65

            for prev in set(out[0].tolist()):
                if next_logit[0, prev] > 0:
                    next_logit[0, prev] /= 1.15
                else:
                    next_logit[0, prev] *= 1.15

            probs = torch.softmax(next_logit, dim=-1)
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

    completion = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    model.train()
    return completion.strip()


def train_1_5b(
    checkpoint_path: Optional[str] = None,
    output_dir: str = "checkpoints/nool_alpha_1_5b",
    total_hours: float = 4.0,
    max_steps: int = 5000,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    learning_rate: float = 1.5e-4,
    min_lr: float = 1.0e-5,
    warmup_steps: int = 150,
    max_seq_len: int = 512,
    device_name: str = "cuda",
    use_8bit_adam: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"[+] Initializing Nool-Alpha-1.5B Distillation on: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1.5B Architecture Configuration
    config = NoolAlphaConfig.full_1_5b(vocab_size=50257, gradient_checkpointing=True)
    model = NoolAlphaForCausalLM(config).to(device)

    total_p, active_p = model.get_num_params()
    print(f"[+] Model Specs: {total_p/1e6:.1f}M Total Params | {active_p/1e6:.1f}M Active Params per token ({active_p/total_p*100:.1f}% compute)")

    # Resume checkpoint if specified
    base_step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[+] Loading base model weights from: {checkpoint_path}")
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            base_step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0

        model_dict = model.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"[OK] Successfully loaded {len(matched)} matching tensors. Base step: {base_step}")

    dataset = MemorySafeDistillDataset(tokenizer=tokenizer, max_seq_len=max_seq_len, reservoir_size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    data_iter = iter(dataloader)

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

    opt_cls = None
    if use_8bit_adam and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            opt_cls = bnb.optim.AdamW8bit
            print("[+] Using bitsandbytes 8-bit AdamW optimizer (Ultra-low VRAM)!")
        except Exception as e:
            print(f"[-] bitsandbytes not available ({e}), falling back to standard torch.optim.AdamW")
            from torch.optim import AdamW
            opt_cls = AdamW
    else:
        from torch.optim import AdamW
        opt_cls = AdamW

    optimizer = opt_cls([
        {"params": decay_params, "weight_decay": 0.05},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        min_lr_ratio=min_lr / learning_rate,
    )

    max_seconds = total_hours * 3600.0
    start_time = time.time()
    step = 0
    running_loss = 0.0
    running_aux = 0.0
    best_loss = float("inf")
    session_tokens = 0
    t_prev = time.time()

    print("=" * 70)
    print(f"🚀 Starting Nool-Alpha-1.5B Training ({total_hours}h Budget = {int(max_seconds)}s)")
    print(f"Effective Batch: {batch_size * grad_accum_steps} seqs ({batch_size * grad_accum_steps * max_seq_len} tok/step)")
    print("=" * 70)

    model.train()
    while step < max_steps:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\n⏱️ Time budget reached ({total_hours}h). Saving final checkpoint...")
            break

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_aux = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss, aux_loss, _ = model(input_ids, labels)
            else:
                logits, loss, aux_loss, _ = model(input_ids, labels)

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            step_loss += loss.item() / grad_accum_steps
            step_aux += aux_loss.item() / grad_accum_steps
            session_tokens += input_ids.numel()

            del logits, loss, aux_loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        running_loss += step_loss
        running_aux += step_aux

        if step % 25 == 0:
            avg_l = running_loss / 25
            avg_a = running_aux / 25
            running_loss = 0.0
            running_aux = 0.0

            dt = time.time() - t_prev
            speed = (25 * grad_accum_steps * batch_size * max_seq_len) / max(dt, 1e-5)
            t_prev = time.time()

            cur_lr = scheduler.get_last_lr()[0]
            rem_h = (max_seconds - (time.time() - start_time)) / 3600.0

            print(
                f"1.5B Step {step:5d}/{max_steps} | Loss: {avg_l:.4f} (Aux: {avg_a:.4f}) | "
                f"LR: {cur_lr:.2e} | Speed: {speed:.0f} tok/s | Tokens: {session_tokens/1e6:.2f}M | Left: {rem_h:.2f}h"
            )

            if avg_l < best_loss:
                best_loss = avg_l
                best_path = os.path.join(output_dir, "best_1_5b_checkpoint.pt")
                torch.save({
                    "step": step,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                }, best_path)
                torch.save({
                    "step": step,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                }, "best_1_5b_checkpoint.pt")
                print(f"  ⭐ Best Checkpoint Saved! (Loss: {best_loss:.4f})")

        if step % 150 == 0:
            print("\n" + "=" * 55)
            print(f"🎯 [Nool-Alpha-1.5B Probes @ Step {step}]")
            for tag, pr in PROBES_1_5B:
                ans = run_probe(model, tokenizer, device, pr, max_tokens=80)
                print(f"[{tag}]\n{ans[:160]}...\n")
            print("=" * 55 + "\n")

    # Final checkpoint
    final_path = os.path.join(output_dir, "nool_alpha_1_5b_final.pt")
    torch.save({
        "step": step,
        "loss": best_loss,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_path)
    torch.save({
        "step": step,
        "loss": best_loss,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, "nool_alpha_1_5b_final.pt")
    print(f"\n🎉 Nool-Alpha-1.5B Training Complete! Checkpoint saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nool-Alpha-1.5B Distillation Training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--hours", type=float, default=4.0, help="Training time budget in hours")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    args = parser.parse_args()

    train_1_5b(
        checkpoint_path=args.checkpoint,
        total_hours=args.hours,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
    )
