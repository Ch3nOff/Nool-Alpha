"""
Streaming multi-domain instruction dataset with Loss Masking for Nool-Alpha SFT.

Applies standard Alpaca formatting and masks out prompt tokens with -100
so that CrossEntropyLoss is ONLY computed over the assistant's response.
"""

import itertools
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


def format_alpaca_prompt(instruction: str, input_text: Optional[str] = None) -> str:
    """Formats an instruction and optional input into standard Alpaca prompt prefix."""
    instruction = instruction.strip()
    if input_text and input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text.strip()}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


class SFTStreamingDataset(IterableDataset):
    """
    Streaming instruction dataset interleaving:
      - 40% Indonesian: FreedomIntelligence/alpaca-gpt4-indonesian
      - 35% English: yahma/alpaca-cleaned
      - 25% Python Code: iamtarun/python_code_instructions_18k_alpaca
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.eos_token_id = tokenizer.eos_token_id or 50256
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_token_id

    def _stream_indonesian(self) -> Iterator[Tuple[str, str]]:
        """Yields (prompt, response) pairs from FreedomIntelligence/alpaca-gpt4-indonesian."""
        while True:
            try:
                ds = load_dataset(
                    "FreedomIntelligence/alpaca-gpt4-indonesian",
                    split="train",
                    streaming=True,
                )
                for item in ds:
                    convs = item.get("conversations", [])
                    if len(convs) >= 2:
                        human_msg = ""
                        gpt_msg = ""
                        for msg in convs:
                            if msg.get("from") == "human" and not human_msg:
                                human_msg = msg.get("value", "")
                            elif msg.get("from") == "gpt" and not gpt_msg:
                                gpt_msg = msg.get("value", "")
                        if human_msg and gpt_msg:
                            prompt = format_alpaca_prompt(human_msg)
                            yield prompt, gpt_msg.strip()
            except Exception as e:
                print(f"[Warning] Indonesian stream error: {e}. Reconnecting...")

    def _stream_english(self) -> Iterator[Tuple[str, str]]:
        """Yields (prompt, response) pairs from yahma/alpaca-cleaned."""
        while True:
            try:
                ds = load_dataset(
                    "yahma/alpaca-cleaned",
                    split="train",
                    streaming=True,
                )
                for item in ds:
                    inst = item.get("instruction", "")
                    inp = item.get("input", "")
                    out = item.get("output", "")
                    if inst and out:
                        prompt = format_alpaca_prompt(inst, inp)
                        yield prompt, out.strip()
            except Exception as e:
                print(f"[Warning] English stream error: {e}. Reconnecting...")

    def _stream_code(self) -> Iterator[Tuple[str, str]]:
        """Yields (prompt, response) pairs from iamtarun/python_code_instructions_18k_alpaca."""
        while True:
            try:
                ds = load_dataset(
                    "iamtarun/python_code_instructions_18k_alpaca",
                    split="train",
                    streaming=True,
                )
                for item in ds:
                    inst = item.get("instruction", "")
                    inp = item.get("input", "")
                    out = item.get("output", "")
                    if inst and out:
                        prompt = format_alpaca_prompt(inst, inp)
                        yield prompt, out.strip()
            except Exception as e:
                print(f"[Warning] Code stream error: {e}. Reconnecting...")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        id_iter = self._stream_indonesian()
        en_iter = self._stream_english()
        code_iter = self._stream_code()

        # Probabilities: 40% ID, 35% EN, 25% Code
        domain_choices = ["id", "en", "code"]
        domain_weights = [0.40, 0.35, 0.25]

        while True:
            chosen = rng.choices(domain_choices, weights=domain_weights, k=1)[0]
            try:
                if chosen == "id":
                    prompt, response = next(id_iter)
                elif chosen == "en":
                    prompt, response = next(en_iter)
                else:
                    prompt, response = next(code_iter)
            except StopIteration:
                continue

            # Tokenize prompt and response separately for loss masking
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response, add_special_tokens=False) + [self.eos_token_id]

            total_len = len(prompt_ids) + len(response_ids)
            if total_len > self.max_seq_len:
                # If total exceeds max_seq_len, truncate response first
                max_resp = self.max_seq_len - len(prompt_ids)
                if max_resp < 16:
                    # If prompt alone takes almost the whole window, skip
                    continue
                response_ids = response_ids[:max_resp - 1] + [self.eos_token_id]

            input_ids = prompt_ids + response_ids
            # Loss masking: prompt tokens receive -100, response tokens receive their token ids
            labels = [-100] * len(prompt_ids) + response_ids

            # Pad to max_seq_len
            pad_len = self.max_seq_len - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
