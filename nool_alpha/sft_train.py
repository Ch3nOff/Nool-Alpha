"""
Supervised Fine-Tuning (SFT) Training Engine for Nool-Alpha-100M.

Features:
  - Loss Masking: Gradients backpropagate only on response tokens.
  - SFT-calibrated learning rate (1e-4 -> 1e-5) to prevent catastrophic forgetting.
  - Multi-domain instruction probes (EN, ID, Python).
  - Time-budgeted loop with automatic checkpointing.
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM
from nool_alpha.sft_dataset import SFTStreamingDataset

# Register safe globals for PyTorch 2.6+ unpickling
if hasattr(sys.modules["__main__"], "NoolAlphaConfig") is False:
    setattr(sys.modules["__main__"], "NoolAlphaConfig", NoolAlphaConfig)

try:
    torch.serialization.add_safe_globals([NoolAlphaConfig])
except Exception:
    pass

INSTRUCTION_PROBES = [
    (
        "English Explanation",
        "### Instruction:\nExplain why the sky appears blue to human eyes.\n\n### Response:\n",
    ),
    (
        "Indonesian Tips",
        "### Instruction:\nSebutkan 3 tips hidup sehat sehari-hari secara ringkas.\n\n### Response:\n",
    ),
    (
        "Python Function",
        "### Instruction:\nWrite a Python function `is_prime(n)` to check if a number is prime.\n\n### Response:\n",
    ),
]


def generate_sft_response(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 80,
    temperature: float = 0.5,
    top_p: float = 0.85,
    repetition_penalty: float = 1.15,
) -> str:
    """Generates response for instruction prompt and stops at <|endoftext|>."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    eos_id = tokenizer.eos_token_id or 50256

    with torch.no_grad():
        for _ in range(max_tokens):
            idx = input_ids[:, -model.config.max_position_embeddings:] if input_ids.shape[1] > model.config.max_position_embeddings else input_ids
            logits, _, _, _ = model(idx)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Repetition penalty
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

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if token_id == eos_id:
                break

    full_output = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    # Extract only response part
    if "### Response:\n" in full_output:
        return full_output.split("### Response:\n", 1)[1].replace("<|endoftext|>", "").strip()
    return full_output[prompt_len:].replace("<|endoftext|>", "").strip()


def run_sft_train(
    checkpoint_path: str,
    output_dir: str = "/kaggle/working/sft_checkpoints",
    max_training_hours: float = 2.0,
    target_max_steps: int = 1500,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 50,
    weight_decay: float = 0.05,
    eval_interval: int = 150,
    log_interval: int = 25,
    device: Optional[str] = None,
):
    """Executes SFT on Nool-Alpha-100M base model."""
    start_time = time.time()
    max_seconds = max_training_hours * 3600.0
    os.makedirs(output_dir, exist_ok=True)

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 65)
    print("[+] Starting Nool-Alpha-100M SFT (Instruction Tuning)")
    print(f"[*] Source Checkpoint: {checkpoint_path}")
    print(f"[*] Device:            {dev}")
    print(f"[*] Time Budget:       {max_training_hours:.2f} hours")
    print(f"[*] Target Steps:      {target_max_steps}")
    print(f"[*] SFT Learning Rate: {learning_rate} (cosine decay to {min_lr})")
    print("=" * 65)

    # 1. Load Pre-trained Base Model Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    raw_cfg = ckpt.get("config")
    if isinstance(raw_cfg, dict):
        config = NoolAlphaConfig(**raw_cfg)
    elif isinstance(raw_cfg, NoolAlphaConfig):
        config = raw_cfg
    else:
        config = NoolAlphaConfig.nool_100m()

    model = NoolAlphaForCausalLM(config).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    base_step = ckpt.get("step", "unknown")
    print(f"[+] Loaded base model weights from Step: {base_step}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Setup SFT Streaming Dataset
    dataset = SFTStreamingDataset(tokenizer=tokenizer, max_seq_len=512)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    data_iter = iter(dataloader)

    # 3. Setup Optimizer & Cosine Scheduler
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    def sft_lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, target_max_steps - warmup_steps))
        return min_lr / learning_rate + 0.5 * (1.0 - min_lr / learning_rate) * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=sft_lr_lambda)

    # 4. Training Loop
    model.train()
    step = 0
    running_loss = 0.0
    running_aux_loss = 0.0
    best_loss = float("inf")
    session_tokens = 0
    t_prev = time.time()

    print("\n[+] SFT Training Loop started...")
    while step < target_max_steps:
        # Check time budget
        elapsed_sec = time.time() - start_time
        if elapsed_sec >= max_seconds:
            print(f"\n[*] Time budget reached ({max_training_hours}h). Gracefully saving final SFT model...")
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

            input_ids = batch["input_ids"].to(dev)
            labels = batch["labels"].to(dev)

            # Mixed precision if CUDA
            if dev.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss, aux_loss, _ = model(input_ids=input_ids, labels=labels)
            else:
                logits, loss, aux_loss, _ = model(input_ids=input_ids, labels=labels)

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            step_loss += loss.item() / grad_accum_steps
            step_aux += aux_loss.item() / grad_accum_steps
            session_tokens += input_ids.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        running_loss += step_loss
        running_aux_loss += step_aux

        # Logging
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_aux = running_aux_loss / log_interval
            running_loss = 0.0
            running_aux_loss = 0.0

            dt = time.time() - t_prev
            tok_per_sec = (log_interval * grad_accum_steps * batch_size * 512) / max(dt, 1e-5)
            t_prev = time.time()

            cur_lr = scheduler.get_last_lr()[0]
            remaining_h = (max_seconds - (time.time() - start_time)) / 3600.0

            print(
                f"SFT Step {step:5d}/{target_max_steps} | "
                f"Loss: {avg_loss:.4f} (Aux: {avg_aux:.4f}) | "
                f"LR: {cur_lr:.2e} | "
                f"Speed: {int(tok_per_sec)} tok/s | "
                f"Tokens: {session_tokens / 1e6:.2f}M | "
                f"Time Left: {max(0.0, remaining_h):.2f}h"
            )

        # Periodic Evaluation Probes & Checkpointing
        if step % eval_interval == 0 or step == target_max_steps:
            print("\n" + "=" * 50)
            print(f"[+] [SFT Instruction Probes @ Step {step}]")
            for name, p in INSTRUCTION_PROBES:
                resp = generate_sft_response(model, tokenizer, p, dev, max_tokens=65)
                print(f"  [{name}]\n  Prompt: {p.strip()}\n  Answer: {resp}\n")
            print("=" * 50 + "\n")
            model.train()

            # Save best checkpoint
            if step_loss < best_loss:
                best_loss = step_loss
                best_path = os.path.join(output_dir, "best_sft_checkpoint.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "loss": best_loss,
                    "type": "sft",
                }, best_path)
                print(f"[OK] Saved new best SFT checkpoint to {best_path} (Loss: {best_loss:.4f})")

    # Final save
    final_path = os.path.join(output_dir, "nool_alpha_100m_sft_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "loss": step_loss,
        "type": "sft",
    }, final_path)
    print(f"\n[OK] SFT session completed! Final model saved to {final_path}")
    return final_path
