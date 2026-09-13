import random
from typing import Iterator, Optional, Dict, Any, List

import torch
from torch.utils.data import IterableDataset, DataLoader


class MultilingualCodeStreamingDataset(IterableDataset):
    """
    Multi-domain streaming dataset combining:
      1. English (e.g. HuggingFaceFW/fineweb-edu or roneneldan/TinyStories)
      2. Indonesian (e.g. wikimedia/wikipedia '20231101.id')
      3. Coding (e.g. iamtarun/python_code_instructions_18k_alpaca or m-a-p/CodeFeedback-Filtered-Instruction)
    
    Streams and packs tokens continuously with zero disk storage footprint,
    interleaving according to configured domain weights (default 40% EN, 40% ID, 20% Code).
    """
    def __init__(
        self,
        en_dataset: str = "HuggingFaceFW/fineweb-edu",
        en_config: Optional[str] = "sample-10BT",
        id_dataset: str = "wikimedia/wikipedia",
        id_config: Optional[str] = "20231101.id",
        code_dataset: str = "iamtarun/python_code_instructions_18k_alpaca",
        code_config: Optional[str] = None,
        domain_weights: Optional[Dict[str, float]] = None,
        tokenizer_name: str = "gpt2",
        seq_len: int = 512,
        buffer_size: int = 1000,
        split: str = "train",
    ):
        super().__init__()
        self.en_dataset = en_dataset
        self.en_config = en_config
        self.id_dataset = id_dataset
        self.id_config = id_config
        self.code_dataset = code_dataset
        self.code_config = code_config
        self.domain_weights = domain_weights or {"en": 0.40, "id": 0.40, "code": 0.20}
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.split = split
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _extract_text(self, item: Dict[str, Any], domain: str) -> str:
        """Extracts and formats text depending on the dataset structure."""
        if not isinstance(item, dict):
            return ""

        # Coding instruction dataset format
        if "instruction" in item and "output" in item:
            instr = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if inp:
                return f"# Task:\n{instr}\n# Input:\n{inp}\n# Code:\n{out}\n"
            return f"# Task:\n{instr}\n# Code:\n{out}\n"

        if "query" in item and "answer" in item:
            return f"# Query:\n{item['query']}\n# Solution:\n{item['answer']}\n"

        # Content or text column
        if "content" in item and isinstance(item["content"], str):
            return item["content"]
        if "text" in item and isinstance(item["text"], str):
            return item["text"]

        return ""

    def _load_stream(self, dataset_name: str, config: Optional[str]):
        from datasets import load_dataset
        try:
            if config:
                ds = load_dataset(dataset_name, config, split=self.split, streaming=True)
            else:
                ds = load_dataset(dataset_name, split=self.split, streaming=True)
            return ds.shuffle(buffer_size=self.buffer_size, seed=42)
        except Exception as e:
            print(f"[Dataset Warning] Failed to stream {dataset_name} ({config}): {e}. Trying fallback...")
            # Fallback to TinyStories if English, or continue
            if "fineweb" in dataset_name:
                return load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            raise e

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        tokenizer = self.tokenizer

        # Initialize iterators for each domain
        streams = {}
        try:
            streams["en"] = iter(self._load_stream(self.en_dataset, self.en_config))
        except Exception as e:
            print(f"[Dataset] EN stream init error: {e}")
            streams["en"] = None

        try:
            streams["id"] = iter(self._load_stream(self.id_dataset, self.id_config))
        except Exception as e:
            print(f"[Dataset] ID stream init error: {e}")
            streams["id"] = None

        try:
            streams["code"] = iter(self._load_stream(self.code_dataset, self.code_config))
        except Exception as e:
            print(f"[Dataset] Code stream init error: {e}")
            streams["code"] = None

        # Filter available domains
        active_domains = [d for d, s in streams.items() if s is not None]
        if not active_domains:
            raise RuntimeError("No active dataset streams available!")

        weights = [self.domain_weights.get(d, 0.33) for d in active_domains]
        token_buffer: List[int] = []

        while True:
            # Sample domain according to weights
            domain = random.choices(active_domains, weights=weights, k=1)[0]
            stream_iter = streams[domain]

            try:
                item = next(stream_iter)
            except (StopIteration, Exception):
                # Reset stream on exhaustion or error
                try:
                    if domain == "en":
                        streams["en"] = iter(self._load_stream(self.en_dataset, self.en_config))
                    elif domain == "id":
                        streams["id"] = iter(self._load_stream(self.id_dataset, self.id_config))
                    elif domain == "code":
                        streams["code"] = iter(self._load_stream(self.code_dataset, self.code_config))
                    stream_iter = streams[domain]
                    item = next(stream_iter)
                except Exception:
                    continue

            text = self._extract_text(item, domain)
            if not text:
                continue

            # Tokenize and append EOS
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)

            token_buffer.extend(tokens)

            # Yield chunks of exact seq_len
            while len(token_buffer) >= self.seq_len:
                chunk = token_buffer[: self.seq_len]
                token_buffer = token_buffer[self.seq_len :]
                input_tensor = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": input_tensor,
                    "labels": input_tensor.clone(),
                }


def get_multilingual_code_dataloaders(
    en_dataset: str = "HuggingFaceFW/fineweb-edu",
    en_config: Optional[str] = "sample-10BT",
    id_dataset: str = "wikimedia/wikipedia",
    id_config: Optional[str] = "20231101.id",
    code_dataset: str = "iamtarun/python_code_instructions_18k_alpaca",
    code_config: Optional[str] = None,
    domain_weights: Optional[Dict[str, float]] = None,
    tokenizer_name: str = "gpt2",
    seq_len: int = 512,
    batch_size: int = 8,
):
    """
    Factory returning (train_loader, tokenizer) configured for
    English + Indonesian + Coding streaming.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = MultilingualCodeStreamingDataset(
        en_dataset=en_dataset,
        en_config=en_config,
        id_dataset=id_dataset,
        id_config=id_config,
        code_dataset=code_dataset,
        code_config=code_config,
        domain_weights=domain_weights or {"en": 0.40, "id": 0.40, "code": 0.20},
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    return train_loader, tokenizer
