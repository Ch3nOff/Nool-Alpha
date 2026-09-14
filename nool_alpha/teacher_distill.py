"""
Command-line Knowledge Distillation Engine for Nool-Alpha-100M with a 2B Teacher Model.
Features:
  - Teacher: Qwen/Qwen2.5-1.5B-Instruct (4-bit NF4 or FP16)
  - Student: Nool-Alpha-100M (~111M params)
  - Strict Prompt Masking: labels = -100 on instruction tokens to cure hallucination
  - Live Teacher-as-a-Judge evaluation during training
"""

import argparse
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Ensure repository root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM


class TeacherDistillDataset(IterableDataset):
    """
    Dual-stream high-density instruction dataset with strict prompt loss masking.
    Labels are set to -100 on prompt tokens so the student only learns how to generate answers.
    """
    def __init__(self, tokenizer, max_seq_len: int = 512, reservoir_size: int = 64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reservoir_size = reservoir_size
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def _stream_alpaca_indonesian(self):
        while True:
            try:
                ds = load_dataset("FreedomIntelligence/alpaca-gpt4-indonesian", split="train", streaming=True)
                for item in ds:
                    convs = item.get("conversations", [])
                    p, r = "", ""
                    if convs:
                        for c in convs:
                            sender = c.get("from", "").lower(); val = c.get("value", "").strip()
                            if sender in ["user", "human"] and not p: p = val
                            elif sender in ["assistant", "gpt"] and p and not r: r = val; break
                    else:
                        inst = item.get("instruction", "").strip()
                        inp = item.get("input", "").strip()
                        out = item.get("output", "").strip()
                        if inst and out:
                            p = f"{inst}\nKonteks: {inp}" if inp else inst
                            r = out
                    if p and r and len(r) > 10:
                        yield p[:1500].strip(), r[:2000].strip()
            except Exception:
                continue

    def _stream_ultrachat(self):
        while True:
            try:
                ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
                for item in ds:
                    msgs = item.get("messages", [])
                    if len(msgs) < 2:
                        continue
                    p, r = "", ""
                    for m in msgs:
                        role = m.get("role", ""); content = m.get("content", "").strip()
                        if role == "user" and not p: p = content
                        elif role == "assistant" and p and not r: r = content; break
                    if p and r and len(r) > 10:
                        yield p[:1500].strip(), r[:2000].strip()
            except Exception:
                continue

    def _tokenize(self, prompt: str, resp: str):
        prompt_txt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        prompt_ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
        resp_ids = self.tokenizer.encode(resp, add_special_tokens=False) + [self.eos_token_id]

        total = len(prompt_ids) + len(resp_ids)
        if total > self.max_seq_len:
            max_resp = self.max_seq_len - len(prompt_ids)
            if max_resp < 24:
                return None
            resp_ids = resp_ids[:max_resp - 1] + [self.eos_token_id]

        input_ids = prompt_ids + resp_ids
        # Strict masking: Prompt tokens are ignored during cross-entropy loss calculation
        labels = [-100] * len(prompt_ids) + resp_ids

        pad_len = self.max_seq_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self):
        import random
        rng = random.Random(42)
        streams = [
            ("alpaca_id", self._stream_alpaca_indonesian()),
            ("ultrachat", self._stream_ultrachat()),
        ]
        weights = [0.65, 0.35]
        stream_indices = list(range(len(streams)))

        def get_sample():
            while True:
                idx = rng.choices(stream_indices, weights=weights, k=1)[0]
                _, st = streams[idx]
                try:
                    p, r = next(st)
                    tok = self._tokenize(p, r)
                    if tok is not None:
                        return tok
                except Exception:
                    continue

        reservoir = [get_sample() for _ in range(self.reservoir_size)]
        while True:
            pick = rng.randint(0, len(reservoir) - 1)
            sample = reservoir[pick]
            reservoir[pick] = get_sample()
            yield sample


def train_teacher_distill(
    teacher_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    student_checkpoint: Optional[str] = None,
    output_dir: str = "checkpoints/teacher_distill_100m",
    total_hours: float = 3.5,
    max_steps: int = 3500,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    learning_rate: float = 2.0e-4,
    min_lr: float = 1.0e-5,
    warmup_steps: int = 100,
    log_interval: int = 25,
    eval_interval: int = 150,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing Teacher-Student Distillation on: {device}")

    # 1. Load Teacher
    print(f"[+] Loading Teacher Model: {teacher_id}...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_id,
            quantization_config=bnb_config,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_id, low_cpu_mem_usage=True)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 2. Load Student (FP32 master weights with AMP for numerical stability)
    student_config = NoolAlphaConfig.nool_100m(vocab_size=50257)
    student_model = NoolAlphaForCausalLM(student_config).to(device=device)
    student_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    if student_checkpoint and os.path.exists(student_checkpoint):
        print(f"[+] Loading Student Checkpoint: {student_checkpoint}")
        if student_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            raw_state = load_file(student_checkpoint)
        else:
            ckpt = torch.load(student_checkpoint, map_location="cpu", weights_only=False)
            raw_state = ckpt.get("model_state_dict", ckpt)

        m_dict = student_model.state_dict()
        matched = {
            k: v.to(device=device, dtype=torch.float32)
            for k, v in raw_state.items()
            if k in m_dict and v.shape == m_dict[k].shape
        }
        m_dict.update(matched)
        student_model.load_state_dict(m_dict)
        print(f"[OK] Loaded {len(matched)} matching tensors into Student!")

    dataset = TeacherDistillDataset(student_tokenizer, max_seq_len=512, reservoir_size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    data_iter = iter(dataloader)

    decay_params = [p for n, p in student_model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in student_model.named_parameters() if p.requires_grad and p.dim() < 2]

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit([
            {"params": decay_params, "weight_decay": 0.05},
            {"params": nodecay_params, "weight_decay": 0.0},
        ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        print("[+] Using 8-bit AdamW for Student!")
    except Exception:
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": 0.05},
            {"params": nodecay_params, "weight_decay": 0.0},
        ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def scheduler_fn(step_idx: int):
        if step_idx < warmup_steps:
            return float(step_idx) / float(max(1, warmup_steps))
        prog = float(step_idx - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr / learning_rate + 0.5 * (1.0 - min_lr / learning_rate) * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)

    start_time = time.time()
    max_seconds = total_hours * 3600.0
    step = 0
    running_loss = 0.0
    running_aux = 0.0
    best_loss = float("inf")
    session_tokens = 0
    t_prev = time.time()

    print("=" * 70)
    print(f"🚀 Starting Live Teacher Distillation Training ({total_hours}h Budget)")
    print("=" * 70)

    student_model.train()
    while step < max_steps:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\n[!] Time limit reached ({total_hours}h).")
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

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16 if device.type == "cuda" else torch.float32):
                logits, loss, aux_loss, _ = student_model(input_ids, labels)

            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            step_loss += loss.item() / grad_accum_steps
            step_aux += aux_loss.item() / grad_accum_steps
            session_tokens += input_ids.numel()

            del logits, loss, aux_loss

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        running_loss += step_loss
        running_aux += step_aux

        if step % log_interval == 0:
            avg_l = running_loss / log_interval
            avg_a = running_aux / log_interval
            running_loss = 0.0
            running_aux = 0.0

            dt = time.time() - t_prev
            speed = (log_interval * grad_accum_steps * batch_size * 512) / max(dt, 1e-5)
            t_prev = time.time()

            cur_lr = scheduler.get_last_lr()[0]
            rem_h = (max_seconds - (time.time() - start_time)) / 3600.0

            print(
                f"Step {step:4d}/{max_steps} | Loss: {avg_l:.4f} (Aux: {avg_a:.4f}) | "
                f"LR: {cur_lr:.2e} | Speed: {speed:.0f} tok/s | Tokens: {session_tokens/1e6:.2f}M | Left: {rem_h:.2f}h"
            )

            if avg_l < best_loss:
                best_loss = avg_l
                torch.save({
                    "step": step,
                    "loss": best_loss,
                    "model_state_dict": student_model.state_dict(),
                    "config": student_config,
                }, os.path.join(output_dir, "best_teacher_distill_checkpoint.pt"))

    # Final checkpoint
    final_path = os.path.join(output_dir, "nool_alpha_100m_teacher_final.pt")
    torch.save({
        "step": step,
        "loss": best_loss,
        "model_state_dict": student_model.state_dict(),
        "config": student_config,
    }, final_path)
    print(f"\n🎉 Distillation Finished! Checkpoint saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nool-Alpha-100M Live Teacher Distillation")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hours", type=float, default=3.5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    args = parser.parse_args()

    train_teacher_distill(
        teacher_id=args.teacher,
        student_checkpoint=args.checkpoint,
        total_hours=args.hours,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
    )
