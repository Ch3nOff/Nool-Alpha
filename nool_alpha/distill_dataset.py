"""
Memory-Safe Multi-Stream Distillation Dataset for Nool-Alpha-1.5B.
Features:
  1. Integrates frontier teacher reasoning & knowledge streams:
     - 40% Teacher Reasoning: ServiceNow-AI/R1-Distill-SFT, open-thoughts/OpenThoughts-114k, nvidia/OpenMathInstruct-1.
     - 25% Code & Algorithms: m-a-p/Code-Feedback.
     - 20% Foundational Knowledge: HuggingFaceFW/fineweb-edu (sample-10BT).
     - 15% Bilingual & Conversation: FreedomIntelligence/alpaca-gpt4-indonesian & HuggingFaceH4/ultrachat_200k.
  2. Zero-OOM Anti-Memorization Architecture:
     - Rolling reservoir buffer of 128 items (< 5 MB RAM).
     - Dynamic entropy seed per run.
     - Raw string pre-capping (prompt <= 2000 chars, resp <= 4000 chars).
  3. Prompt Loss Masking:
     - Labels for instructions assigned -100; backprop active only on target responses.
"""

import os
import random
import time
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class MemorySafeDistillDataset(IterableDataset):
    """
    Zero-OOM multi-stream distillation dataset with rolling reservoir anti-memorization sampling.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
        reservoir_size: int = 128,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reservoir_size = reservoir_size
        self.eos_token_id = tokenizer.eos_token_id or 50256
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_token_id

        if seed is None:
            self.seed = int(time.time_ns() % 1_000_000_007) ^ (os.getpid() << 16)
        else:
            self.seed = seed

        self.rng = random.Random(self.seed)

    def _stream_r1_distill(self):
        while True:
            try:
                ds = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1", split="train", streaming=True)
                for item in ds:
                    r = item.get("reannotated_assistant_content", "")
                    msgs = item.get("messages", []) or item.get("reannotated_messages", [])
                    p = ""
                    for m in msgs:
                        if m.get("role") == "user":
                            p = m.get("content", "")
                            break
                    if not r:
                        for m in msgs:
                            if m.get("role") == "assistant":
                                r = m.get("content", "")
                                break
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_openthoughts(self):
        while True:
            try:
                ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
                for item in ds:
                    convs = item.get("conversations", [])
                    p, r = "", ""
                    for c in convs:
                        sender = c.get("from", "").lower()
                        val = c.get("value", "")
                        if sender in ["user", "human"] and not p:
                            p = val
                        elif sender in ["assistant", "gpt"] and p and not r:
                            r = val
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_openmath(self):
        while True:
            try:
                ds = load_dataset("nvidia/OpenMathInstruct-1", split="train", streaming=True)
                for item in ds:
                    if "is_correct" in item and not item["is_correct"]:
                        continue
                    p = item.get("question", "")
                    r = item.get("generated_solution", "")
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_code_feedback(self):
        while True:
            try:
                ds = load_dataset("m-a-p/Code-Feedback", split="train", streaming=True)
                for item in ds:
                    msgs = item.get("messages", [])
                    p, r = "", ""
                    for m in msgs:
                        role = m.get("role")
                        content = m.get("content", "")
                        if role == "user" and not p:
                            p = content
                        elif role == "assistant" and p and not r:
                            r = content
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_fineweb_edu(self):
        while True:
            try:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
                for item in ds:
                    text = item.get("text", "").strip()
                    if len(text) > 80:
                        # Format educational raw text into knowledge completion prompt
                        if "\n\n" in text:
                            parts = text.split("\n\n", 1)
                            prompt = f"Explain the following concept in detail:\n{parts[0][:1500]}"
                            resp = parts[1][:4000]
                        else:
                            prompt = "Explain the fundamental principles of the following topic:"
                            resp = text[:4000]
                        yield prompt, resp
            except Exception:
                continue

    def _stream_alpaca_indonesian(self):
        while True:
            try:
                ds = load_dataset("FreedomIntelligence/alpaca-gpt4-indonesian", split="train", streaming=True)
                for item in ds:
                    convs = item.get("conversations", [])
                    p, r = "", ""
                    if convs:
                        for c in convs:
                            sender = c.get("from", "").lower()
                            val = c.get("value", "").strip()
                            if sender in ["user", "human"] and not p:
                                p = val
                            elif sender in ["assistant", "gpt"] and p and not r:
                                r = val
                                break
                    else:
                        inst = item.get("instruction", "").strip()
                        inp = item.get("input", "").strip()
                        out = item.get("output", "").strip()
                        if inst and out:
                            p = f"{inst}\n\nKonteks:\n{inp}" if inp else inst
                            r = out

                    if p and r and len(r) > 5:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_ultrachat(self):
        while True:
            try:
                ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
                for item in ds:
                    messages = item.get("messages", [])
                    if len(messages) < 2:
                        continue
                    p, r = "", ""
                    for m in messages:
                        role = m.get("role", "")
                        content = m.get("content", "").strip()
                        if role == "user" and not p:
                            p = content
                        elif role == "assistant" and p and not r:
                            r = content
                            break
                    if p and r and len(r) > 5:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _tokenize(self, prompt: str, resp: str) -> Optional[Dict[str, torch.Tensor]]:
        prompt_txt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        prompt_ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
        resp_ids = self.tokenizer.encode(resp, add_special_tokens=False) + [self.eos_token_id]

        total = len(prompt_ids) + len(resp_ids)
        if total > self.max_seq_len:
            max_resp = self.max_seq_len - len(prompt_ids)
            if max_resp < 16:
                return None
            resp_ids = resp_ids[: max_resp - 1] + [self.eos_token_id]

        input_ids = prompt_ids + resp_ids
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

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        streams = [
            ("r1_distill", self._stream_r1_distill()),
            ("openthoughts", self._stream_openthoughts()),
            ("openmath", self._stream_openmath()),
            ("code_feedback", self._stream_code_feedback()),
            ("fineweb_edu", self._stream_fineweb_edu()),
            ("alpaca_indonesian", self._stream_alpaca_indonesian()),
            ("ultrachat", self._stream_ultrachat()),
        ]
        # Weights:
        # R1 (15%), OpenThoughts (15%), OpenMath (10%) -> 40% Reasoning
        # Code Feedback (25%) -> 25% Code & Logic
        # FineWeb-Edu (20%) -> 20% Foundational Knowledge
        # Indonesian Alpaca (8%), UltraChat (7%) -> 15% Bilingual & Chat
        weights = [0.15, 0.15, 0.10, 0.25, 0.20, 0.08, 0.07]
        stream_indices = list(range(len(streams)))

        def get_next_sample():
            while True:
                chosen_idx = self.rng.choices(stream_indices, weights=weights, k=1)[0]
                _, stream = streams[chosen_idx]
                try:
                    p, r = next(stream)
                    tok = self._tokenize(p, r)
                    if tok is not None:
                        return tok
                except Exception:
                    continue

        reservoir = []
        for _ in range(self.reservoir_size):
            reservoir.append(get_next_sample())

        while True:
            pick_idx = self.rng.randint(0, len(reservoir) - 1)
            sample = reservoir[pick_idx]
            reservoir[pick_idx] = get_next_sample()
            yield sample
