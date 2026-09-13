"""
Multi-Source Randomized Streaming Reasoning Dataset for Nool-Alpha-100M.

Features:
  1. Integrates 7 premier reasoning, math, code, and synthetic instruction datasets:
     - HuggingFaceTB/cosmopedia-100k
     - m-a-p/Code-Feedback
     - nvidia/OpenMathInstruct-1
     - ServiceNow-AI/R1-Distill-SFT (v1)
     - open-thoughts/OpenThoughts-114k
     - open-r1/Mixture-of-Thoughts (all)
     - IFM/Math-Reasoning (math-thinking-qwen)
  2. Anti-Memorization 3-Layer Randomization:
     - Dynamic entropy seed (time_ns ^ pid) ensuring unique permutations per run.
     - 10,000-sample streaming shuffle buffer per dataset stream.
     - Random shard skip offsets (fast-forwarding 0-3000 items) to prevent reading from index 0.
     - Multinomial weighted sampling across the 7 streams.
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


REASONING_DATASET_CONFIGS = [
    {
        "tag": "r1_distill",
        "repo": "ServiceNow-AI/R1-Distill-SFT",
        "config": "v1",
        "weight": 0.20,
    },
    {
        "tag": "openthoughts",
        "repo": "open-thoughts/OpenThoughts-114k",
        "config": None,
        "weight": 0.15,
    },
    {
        "tag": "mot",
        "repo": "open-r1/Mixture-of-Thoughts",
        "config": "all",
        "weight": 0.15,
    },
    {
        "tag": "openmath",
        "repo": "nvidia/OpenMathInstruct-1",
        "config": None,
        "weight": 0.15,
    },
    {
        "tag": "code_feedback",
        "repo": "m-a-p/Code-Feedback",
        "config": None,
        "weight": 0.15,
    },
    {
        "tag": "cosmopedia",
        "repo": "HuggingFaceTB/cosmopedia-100k",
        "config": None,
        "weight": 0.10,
    },
    {
        "tag": "math_reasoning",
        "repo": "IFM/Math-Reasoning",
        "config": "math-thinking-qwen",
        "weight": 0.10,
    },
]


def extract_prompt_response(item: Dict, tag: str) -> Optional[Tuple[str, str]]:
    """Normalizes heterogenous dataset schemas into (prompt, response)."""
    try:
        if tag == "cosmopedia":
            prompt = item.get("prompt", "").strip()
            response = item.get("text", "").strip()
            if prompt and response:
                return prompt, response

        elif tag == "code_feedback":
            msgs = item.get("messages", [])
            prompt, response = "", ""
            for msg in msgs:
                role = msg.get("role")
                content = msg.get("content", "").strip()
                if role == "user" and not prompt:
                    prompt = content
                elif role == "assistant" and prompt and not response:
                    response = content
            if prompt and response:
                return prompt, response

        elif tag == "openmath":
            # Prefer correct solutions
            if "is_correct" in item and not item["is_correct"]:
                return None
            prompt = item.get("question", "").strip()
            response = item.get("generated_solution", "").strip()
            if prompt and response:
                return prompt, response

        elif tag == "r1_distill":
            # Check reannotated assistant content with <think> reasoning
            response = item.get("reannotated_assistant_content", "").strip()
            msgs = item.get("messages", []) or item.get("reannotated_messages", [])
            prompt = ""
            for msg in msgs:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "").strip()
                    break
            if not response:
                for msg in msgs:
                    if msg.get("role") == "assistant":
                        response = msg.get("content", "").strip()
                        break
            if prompt and response:
                return prompt, response

        elif tag == "openthoughts":
            convs = item.get("conversations", [])
            prompt, response = "", ""
            for turn in convs:
                sender = turn.get("from", "").lower()
                val = turn.get("value", "").strip()
                if sender in ["user", "human"] and not prompt:
                    prompt = val
                elif sender in ["assistant", "gpt"] and prompt and not response:
                    response = val
            if prompt and response:
                return prompt, response

        elif tag == "mot":
            msgs = item.get("messages", [])
            prompt, response = "", ""
            for msg in msgs:
                role = msg.get("role")
                content = msg.get("content", "").strip()
                if role == "user" and not prompt:
                    prompt = content
                elif role == "assistant" and prompt and not response:
                    response = content
            if prompt and response:
                return prompt, response

        elif tag == "math_reasoning":
            raw_text = item.get("text", "").strip()
            if "\n\n" in raw_text:
                parts = raw_text.split("\n\n", 1)
                prompt = parts[0].strip()
                response = parts[1].strip()
                if prompt and response:
                    return prompt, response
            elif "?" in raw_text:
                parts = raw_text.split("?", 1)
                prompt = parts[0].strip() + "?"
                response = parts[1].strip()
                if prompt and response:
                    return prompt, response
            if raw_text:
                return "Solve the following mathematical problem step by step:", raw_text

    except Exception:
        return None

    return None


class RandomizedReasoningSFTDataset(IterableDataset):
    """
    True Randomized Multi-Stream Reasoning Dataset.

    Prevents memorization and sequential bias across restarts by:
      - Instantiating non-deterministic seeds.
      - Maintaining a 10,000-sample shuffle reservoir on each stream.
      - Fast-forwarding streams by a random initial skip offset.
      - Dynamically sampling streams via weighted multinomial distribution.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
        buffer_size: int = 10000,
        enable_random_skip: bool = True,
        max_skip_offset: int = 3000,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.enable_random_skip = enable_random_skip
        self.max_skip_offset = max_skip_offset

        # Entropy-based seed if none provided
        if seed is None:
            self.seed = int(time.time_ns() % 1_000_000_007) ^ (os.getpid() << 16)
        else:
            self.seed = seed

        self.rng = random.Random(self.seed)
        self.weights = [cfg["weight"] for cfg in REASONING_DATASET_CONFIGS]
        self.dataset_configs = REASONING_DATASET_CONFIGS

    def _create_stream(self, cfg: Dict, stream_seed: int) -> Iterator:
        kwargs = {"split": "train", "streaming": True}
        if cfg["config"]:
            kwargs["name"] = cfg["config"]

        ds = load_dataset(cfg["repo"], **kwargs)
        # Apply streaming shuffle buffer
        ds = ds.shuffle(buffer_size=self.buffer_size, seed=stream_seed)

        # Apply random shard skip offset to avoid restarting at index 0
        if self.enable_random_skip and self.max_skip_offset > 0:
            skip_count = self.rng.randint(0, self.max_skip_offset)
            try:
                ds = ds.skip(skip_count)
            except Exception:
                pass

        return iter(ds)

    def _format_and_tokenize(self, prompt: str, response: str) -> Optional[Dict[str, torch.Tensor]]:
        prompt_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
        full_text = f"{prompt_text}{response}<|endoftext|>"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(prompt_ids) >= self.max_seq_len - 10:
            return None

        # Truncate to max_seq_len
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]

        input_ids = full_ids.copy()
        # Loss Masking: Set all prompt tokens to -100
        labels = full_ids.copy()
        mask_len = min(len(prompt_ids), len(labels))
        for i in range(mask_len):
            labels[i] = -100

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = [1] * len(full_ids) + [0] * pad_len
        else:
            attention_mask = [1] * len(input_ids)

        # Verify active response tokens remain
        active_tokens = sum(1 for l in labels if l != -100)
        if active_tokens < 4:
            return None

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Initialize each stream with an independent randomized seed
        streams = []
        active_indices = []

        for idx, cfg in enumerate(self.dataset_configs):
            stream_seed = (self.seed + idx * 7919) % (2**31 - 1)
            try:
                s = self._create_stream(cfg, stream_seed)
                streams.append(s)
                active_indices.append(idx)
            except Exception as e:
                print(f"[!] Warning: Failed to initialize stream '{cfg['tag']}': {e}")
                streams.append(None)

        if not active_indices:
            raise RuntimeError("No reasoning dataset streams could be initialized.")

        while active_indices:
            # Weighted random selection of stream
            current_weights = [self.weights[i] for i in active_indices]
            total_w = sum(current_weights)
            norm_weights = [w / total_w for w in current_weights]
            chosen_idx = self.rng.choices(active_indices, weights=norm_weights, k=1)[0]

            stream = streams[chosen_idx]
            tag = self.dataset_configs[chosen_idx]["tag"]

            try:
                item = next(stream)
            except StopIteration:
                # Stream exhausted, re-seed and restart stream
                new_seed = self.rng.randint(0, 2**31 - 1)
                try:
                    streams[chosen_idx] = self._create_stream(self.dataset_configs[chosen_idx], new_seed)
                    item = next(streams[chosen_idx])
                except Exception:
                    active_indices.remove(chosen_idx)
                    continue
            except Exception:
                continue

            extracted = extract_prompt_response(item, tag)
            if extracted is None:
                continue

            prompt, response = extracted
            tokenized = self._format_and_tokenize(prompt, response)
            if tokenized is not None:
                yield tokenized
