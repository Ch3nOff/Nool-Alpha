"""
Inference engine for Nool-Alpha models with streaming token generation,
checkpoint switching, and sampling parameter controls.
"""

import sys
import os
import time
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM

# Register safe globals and alias in __main__ for PyTorch checkpoint loading
if hasattr(sys.modules["__main__"], "NoolAlphaConfig") is False:
    setattr(sys.modules["__main__"], "NoolAlphaConfig", NoolAlphaConfig)

try:
    torch.serialization.add_safe_globals([NoolAlphaConfig])
except Exception:
    pass

DEFAULT_MODEL_DIR = r"C:\Users\Matthew Chen\Downloads\Nool_alpha model"


class NoolAlphaInference:
    """High-level inference engine for Nool-Alpha causal language models."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
        tokenizer_name: str = "gpt2",
    ):
        self.model_dir = model_dir
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model: Optional[NoolAlphaForCausalLM] = None
        self.config: Optional[NoolAlphaConfig] = None
        self.current_checkpoint_name: Optional[str] = None
        self.checkpoint_metadata: Dict = {}

        # Resolve initial checkpoint
        target_ckpt = checkpoint_path or self._find_default_checkpoint()
        if target_ckpt and os.path.exists(target_ckpt):
            self.load_checkpoint(target_ckpt)

    def _find_default_checkpoint(self) -> Optional[str]:
        """Finds latest SFT or best pre-trained checkpoint in model_dir."""
        if not os.path.isdir(self.model_dir):
            return None
        priority_candidates = [
            os.path.join(self.model_dir, "sft", "best_sft_checkpoint.pt"),
            os.path.join(self.model_dir, "sft", "nool_alpha_100m_sft_final.pt"),
            os.path.join(self.model_dir, "best_sft_checkpoint.pt"),
            os.path.join(self.model_dir, "nool_alpha_100m_sft_final.pt"),
            os.path.join(self.model_dir, "nool_alpha_100m_final.pt"),
            os.path.join(self.model_dir, "best_checkpoint.pt"),
        ]
        for candidate in priority_candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def get_available_checkpoints(self) -> List[Dict[str, str]]:
        """Lists all .pt checkpoints found in model_dir and subfolders."""
        if not os.path.isdir(self.model_dir):
            return []
        items = []
        for root, _, files in os.walk(self.model_dir):
            for fname in sorted(files):
                if fname.endswith(".pt"):
                    full_path = os.path.join(root, fname)
                    rel_name = os.path.relpath(full_path, self.model_dir).replace("\\", "/")
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    category = "SFT Chat" if "sft" in rel_name.lower() else "Base Pretrain"
                    items.append({
                        "filename": rel_name,
                        "display_name": f"{rel_name} [{category}]",
                        "category": category,
                        "path": full_path,
                        "size_mb": round(size_mb, 1),
                    })
        return items

    def load_checkpoint(self, checkpoint_path_or_filename: str) -> None:
        """Loads or switches checkpoint state dict."""
        if os.path.isabs(checkpoint_path_or_filename):
            full_path = checkpoint_path_or_filename
        else:
            direct = os.path.join(self.model_dir, checkpoint_path_or_filename)
            if os.path.exists(direct):
                full_path = direct
            else:
                found = None
                for root, _, files in os.walk(self.model_dir):
                    base_target = os.path.basename(checkpoint_path_or_filename)
                    if base_target in files:
                        found = os.path.join(root, base_target)
                        break
                full_path = found if found else direct

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Checkpoint not found at: {full_path}")

        print(f"Loading checkpoint from: {full_path} on {self.device}...")
        t0 = time.time()
        ckpt = torch.load(full_path, map_location=self.device, weights_only=False)

        raw_config = ckpt.get("config")
        if isinstance(raw_config, dict):
            self.config = NoolAlphaConfig(**raw_config)
        elif isinstance(raw_config, NoolAlphaConfig):
            self.config = raw_config
        else:
            from nool_alpha.config import nool_100m
            self.config = nool_100m()

        # Re-instantiate model if architecture config changed or not instantiated yet
        if self.model is None:
            self.model = NoolAlphaForCausalLM(self.config).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.current_checkpoint_name = os.path.basename(full_path)
        self.checkpoint_metadata = {
            "step": ckpt.get("step", "unknown"),
            "loss": ckpt.get("loss", None),
            "filename": self.current_checkpoint_name,
            "path": full_path,
            "load_time_sec": round(time.time() - t0, 2),
        }
        total_p, active_p = self.model.get_num_params()
        self.checkpoint_metadata["total_params_m"] = round(total_p / 1e6, 2)
        self.checkpoint_metadata["active_params_m"] = round(active_p / 1e6, 2)
        print(f"Loaded successfully in {self.checkpoint_metadata['load_time_sec']}s. Step: {self.checkpoint_metadata['step']}")

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.6,
        top_p: float = 0.85,
        repetition_penalty: float = 1.2,
    ) -> Generator[Dict[str, Union[str, float, int]], None, None]:
        """
        Generates tokens autoregressively and yields each new token in real-time.
        Yields dict with:
          token: decoded string for the new token
          accumulated_text: complete generated text so far
          token_idx: current step index (0-based)
          tok_per_sec: current generation speed
        """
        if self.model is None:
            raise RuntimeError("No model checkpoint is loaded.")

        self.model.eval()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        start_time = time.time()
        accumulated_tokens = input_ids[0].tolist()

        for step in range(max_new_tokens):
            # Truncate context if exceeding max_position_embeddings
            if input_ids.shape[1] > self.config.max_position_embeddings:
                idx = input_ids[:, -self.config.max_position_embeddings:]
            else:
                idx = input_ids

            logits, _, _, _ = self.model(idx)
            next_logits = logits[:, -1, :].clone() / max(temperature, 1e-5)

            # Apply repetition penalty across past tokens
            if repetition_penalty != 1.0:
                seen_tokens = set(accumulated_tokens)
                for token_id in seen_tokens:
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            # Top-P (Nucleus) Filtering
            if top_p < 1.0:
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
            accumulated_tokens.append(token_id)

            new_token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) / max(elapsed, 1e-5)

            full_text = self.tokenizer.decode(accumulated_tokens[prompt_len:], skip_special_tokens=True)

            yield {
                "token": new_token_text,
                "accumulated_text": full_text,
                "token_idx": step,
                "tok_per_sec": round(tok_per_sec, 1),
                "elapsed_sec": round(elapsed, 2),
                "is_finished": (step == max_new_tokens - 1) or (token_id == self.tokenizer.eos_token_id),
            }

            if token_id == self.tokenizer.eos_token_id:
                break

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.6,
        top_p: float = 0.85,
        repetition_penalty: float = 1.2,
    ) -> Dict[str, Union[str, float, int]]:
        """Synchronous generation returning complete generated output and metrics."""
        result = None
        for chunk in self.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            result = chunk

        if result is None:
            return {
                "prompt": prompt,
                "completion": "",
                "full_text": prompt,
                "tokens_generated": 0,
                "elapsed_sec": 0.0,
                "tok_per_sec": 0.0,
            }

        return {
            "prompt": prompt,
            "completion": result["accumulated_text"],
            "full_text": prompt + result["accumulated_text"],
            "tokens_generated": result["token_idx"] + 1,
            "elapsed_sec": result["elapsed_sec"],
            "tok_per_sec": result["tok_per_sec"],
        }
