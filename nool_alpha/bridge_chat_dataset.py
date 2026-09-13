"""
Memory-Safe Multi-Source Streaming Dataset for Bilingual Bridge & Everyday Natural Conversation.
Designed for Nool-Alpha-100M SFT Stage 3.

Features:
  1. Integrates 3 premier multilingual & conversational datasets:
     - OPUS Translation (kaitchup/opus-Indonesian-to-English & hfxunlp/opus-100):
       Bidirectional English <-> Indonesian semantic alignment.
     - HuggingFaceH4/ultrachat_200k (train_sft):
       Everyday natural multi-turn chat & daily interactions.
     - FreedomIntelligence/alpaca-gpt4-indonesian:
       Fluent, polite everyday Indonesian conversational dialogue.
  2. Zero-OOM Anti-Memorization Architecture:
     - Rolling reservoir buffer of 128 items (< 5 MB Host RAM).
     - Raw string pre-capping (prompt <= 2000 chars, resp <= 4000 chars).
     - Dynamic entropy seed (time_ns ^ pid) ensuring unique permutations per run.
     - Weighted multinomial sampling: 35% OPUS, 35% UltraChat, 30% Indonesian Alpaca.
  3. Prompt Loss Masking:
     - Labels for '### Instruction:\n{prompt}\n\n### Response:\n' are set to -100.
     - Active backpropagation loss exclusively on assistant response tokens.
"""

import os
import random
import time
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class MemorySafeBridgeChatDataset(IterableDataset):
    """
    Zero-OOM streaming dataset with rolling reservoir anti-memorization sampling
    specialized for bilingual bridge (En <-> Id) and natural daily conversation.
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

    def _stream_opus_translation(self):
        """
        Stream translation pairs between Indonesian and English.
        Uses kaitchup/opus-Indonesian-to-English with fallback to hfxunlp/opus-100.
        """
        en_to_id_templates = [
            "Terjemahkan ke dalam bahasa Indonesia:\n{text}",
            "Terjemahkan kalimat berikut ke dalam bahasa Indonesia:\n{text}",
            "Translate the following sentence into Indonesian:\n{text}",
            "Bagaimana terjemahan bahasa Indonesia dari teks ini?\n{text}",
            "Translate to Indonesian:\n{text}",
        ]
        id_to_en_templates = [
            "Terjemahkan ke dalam bahasa Inggris:\n{text}",
            "Terjemahkan kalimat berikut ke dalam bahasa Inggris:\n{text}",
            "Translate the following Indonesian sentence into English:\n{text}",
            "Bagaimana terjemahan bahasa Inggris dari teks ini?\n{text}",
            "Translate to English:\n{text}",
        ]

        while True:
            # 1. Try kaitchup/opus-Indonesian-to-English
            try:
                ds = load_dataset("kaitchup/opus-Indonesian-to-English", split="train", streaming=True)
                for item in ds:
                    raw_text = item.get("text", "")
                    if "###>" in raw_text:
                        parts = raw_text.split("###>", 1)
                        id_text = parts[0].strip()
                        en_text = parts[1].strip()
                        if len(id_text) < 4 or len(en_text) < 4:
                            continue

                        # 50% chance En -> Id, 50% chance Id -> En
                        if self.rng.random() < 0.5:
                            tmpl = self.rng.choice(en_to_id_templates)
                            yield tmpl.format(text=en_text[:2000]), id_text[:4000]
                        else:
                            tmpl = self.rng.choice(id_to_en_templates)
                            yield tmpl.format(text=id_text[:2000]), en_text[:4000]
            except Exception:
                pass

            # 2. Fallback: hfxunlp/opus-100 (if accessible)
            try:
                ds = load_dataset("hfxunlp/opus-100", "en-id", split="train", streaming=True)
                for item in ds:
                    trans = item.get("translation", {})
                    en_text = trans.get("en", "").strip()
                    id_text = trans.get("id", "").strip()
                    if len(id_text) < 4 or len(en_text) < 4:
                        continue

                    if self.rng.random() < 0.5:
                        tmpl = self.rng.choice(en_to_id_templates)
                        yield tmpl.format(text=en_text[:2000]), id_text[:4000]
                    else:
                        tmpl = self.rng.choice(id_to_en_templates)
                        yield tmpl.format(text=id_text[:2000]), en_text[:4000]
            except Exception:
                pass

    def _stream_ultrachat(self):
        """
        Stream everyday conversational multi-turn dialogues from HuggingFaceH4/ultrachat_200k.
        """
        while True:
            try:
                ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
                for item in ds:
                    messages = item.get("messages", [])
                    if len(messages) < 2:
                        continue

                    prompt = ""
                    response = ""
                    for m in messages:
                        role = m.get("role", "")
                        content = m.get("content", "").strip()
                        if role == "user" and not prompt:
                            prompt = content
                        elif role == "assistant" and prompt and not response:
                            response = content
                            break

                    if prompt and response and len(response) > 5:
                        yield prompt[:2000].strip(), response[:4000].strip()
            except Exception:
                continue

    def _stream_indonesian_dialogue(self):
        """
        Stream everyday polite Indonesian conversational dialogue from FreedomIntelligence/alpaca-gpt4-indonesian.
        """
        while True:
            try:
                ds = load_dataset("FreedomIntelligence/alpaca-gpt4-indonesian", split="train", streaming=True)
                for item in ds:
                    convs = item.get("conversations", [])
                    prompt, response = "", ""
                    if convs:
                        for c in convs:
                            sender = c.get("from", "").lower()
                            val = c.get("value", "").strip()
                            if sender in ["user", "human"] and not prompt:
                                prompt = val
                            elif sender in ["assistant", "gpt"] and prompt and not response:
                                response = val
                                break
                    else:
                        inst = item.get("instruction", "").strip()
                        inp = item.get("input", "").strip()
                        out = item.get("output", "").strip()
                        if inst and out:
                            prompt = f"{inst}\n\nKonteks:\n{inp}" if inp else inst
                            response = out

                    if prompt and response and len(response) > 5:
                        yield prompt[:2000].strip(), response[:4000].strip()
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
            ("opus_translation", self._stream_opus_translation()),
            ("ultrachat_everyday", self._stream_ultrachat()),
            ("indonesian_dialogue", self._stream_indonesian_dialogue()),
        ]
        # 35% Bilingual Translation Bridge, 35% Everyday UltraChat, 30% Indonesian Dialogue
        weights = [0.35, 0.35, 0.30]
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

        # 1. Fill rolling reservoir buffer with initial samples
        reservoir = []
        for _ in range(self.reservoir_size):
            reservoir.append(get_next_sample())

        # 2. Continuous random replacement sampling (anti-memorization)
        while True:
            pick_idx = self.rng.randint(0, len(reservoir) - 1)
            sample = reservoir[pick_idx]
            # Replace picked item with fresh sample from stream
            reservoir[pick_idx] = get_next_sample()
            yield sample
