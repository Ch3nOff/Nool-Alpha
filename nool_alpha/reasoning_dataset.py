"""
Memory-Safe Multi-Source Randomized Streaming Reasoning Dataset for Nool-Alpha-100M.

Features:
  1. Integrates 7 premier reasoning, math, code, and synthetic instruction datasets:
     - HuggingFaceTB/cosmopedia-100k
     - m-a-p/Code-Feedback
     - nvidia/OpenMathInstruct-1
     - ServiceNow-AI/R1-Distill-SFT (v1)
     - open-thoughts/OpenThoughts-114k
     - open-r1/Mixture-of-Thoughts (all)
     - IFM/Math-Reasoning (math-thinking-qwen)
  2. Zero-OOM Anti-Memorization Architecture:
     - Rolling reservoir buffer of 128 items (< 5 MB Host RAM) instead of massive multi-thousand buffers.
     - Raw string pre-capping (prompt <= 2000 chars, resp <= 4000 chars) to prevent 20k-token reasoning traces from blowing Host RAM.
     - Dynamic entropy seed (time_ns ^ pid) ensuring unique permutations per run.
     - Weighted multinomial sampling across the 7 streams.
  3. Prompt Loss Masking:
     - Labels for '### Instruction:\n{prompt}\n\n### Response:\n' are set to -100.
     - Active backpropagation loss exclusively on reasoning traces and final response tokens.
"""

import os
import random
import time
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class MemorySafeReasoningDataset(IterableDataset):
    """
    Zero-OOM streaming dataset with rolling reservoir anti-memorization sampling.
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

    def _stream_cosmopedia(self):
        while True:
            try:
                ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train", streaming=True)
                for item in ds:
                    p = item.get("prompt", "")
                    r = item.get("text", "")
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

    def _stream_mot(self):
        while True:
            try:
                ds = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="train", streaming=True)
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

    def _stream_math_reasoning(self):
        while True:
            try:
                ds = load_dataset("IFM/Math-Reasoning", "math-thinking-qwen", split="train", streaming=True)
                for item in ds:
                    t = item.get("text", "")
                    if "\n\n" in t:
                        parts = t.split("\n\n", 1)
                        yield parts[0][:2000].strip(), parts[1][:4000].strip()
                    elif t:
                        yield "Solve the following mathematical reasoning problem step by step:", t[:4000].strip()
            except Exception:
                continue

    def _tokenize(self, prompt: str, resp: str):
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

    def __iter__(self):
        streams = [
            ("r1", self._stream_r1_distill()),
            ("openthoughts", self._stream_openthoughts()),
            ("mot", self._stream_mot()),
            ("openmath", self._stream_openmath()),
            ("code", self._stream_code_feedback()),
            ("cosmopedia", self._stream_cosmopedia()),
            ("math_qwen", self._stream_math_reasoning()),
        ]
        weights = [0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10]
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
