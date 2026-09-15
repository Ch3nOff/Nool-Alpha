"""
Inference engine for Nool-Alpha models with streaming token generation,
checkpoint switching, and sampling parameter controls.
Supports both .pt checkpoints and Hugging Face / safetensors model directories.
"""

import json
import os
import sys
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


def remap_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remaps weight keys between export formats (Safetensors / HF)
    and the native NoolAlphaForCausalLM module architecture.
    """
    remapped = {}
    for k, v in sd.items():
        new_k = k
        if ".attn.w_uk.weight" in k:
            new_k = k.replace(".attn.w_uk.weight", ".attn.w_uk")
            if v.ndim == 2:
                # Shape (768, 192) -> (12, 64, 192)
                v = v.view(12, 64, v.shape[1])
        elif ".attn.w_uv.weight" in k:
            new_k = k.replace(".attn.w_uv.weight", ".attn.w_uv")
            if v.ndim == 2:
                # Shape (768, 192) -> (12, 64, 192)
                v = v.view(12, 64, v.shape[1])
        elif ".moe.shared_gate.weight" in k:
            new_k = k.replace(".moe.shared_gate.weight", ".moe.shared_w_gate.weight")
        elif ".moe.shared_up.weight" in k:
            new_k = k.replace(".moe.shared_up.weight", ".moe.shared_w_up.weight")
        elif ".moe.shared_down.weight" in k:
            new_k = k.replace(".moe.shared_down.weight", ".moe.shared_w_down.weight")
        remapped[new_k] = v
    return remapped


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
            if torch.cuda.is_available():
                try:
                    # Test if installed PyTorch binary has compiled kernels for this GPU architecture (e.g. sm_120)
                    _test = torch.zeros(1, device="cuda") + 1
                    self.device = torch.device("cuda")
                except Exception as e:
                    print(f"[!] Catatan Hardware: {e.args[0] if e.args else e}")
                    print(f"[*] GPU {torch.cuda.get_device_name(0)} (Blackwell sm_120) memerlukan PyTorch cu128+. Beralih ke CPU lokal berkecepatan tinggi (~25 tok/s).", flush=True)
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")


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
        """Finds latest Stage 2 enx model or best SFT/pre-trained checkpoint in model_dir."""
        if not os.path.isdir(self.model_dir):
            return None
        priority_candidates = [
            os.path.join(self.model_dir, "sft", "enx model"),
            os.path.join(self.model_dir, "sft", "Model"),
            os.path.join(self.model_dir, "model"),
            os.path.join(self.model_dir, "sft", "best_bridge_checkpoint.pt"),
            os.path.join(self.model_dir, "sft", "best_sft_checkpoint.pt"),
            os.path.join(self.model_dir, "sft", "nool_alpha_100m_bridge_final.pt"),
            os.path.join(self.model_dir, "sft", "nool_alpha_100m_sft_final.pt"),
            os.path.join(self.model_dir, "best_checkpoint.pt"),
            os.path.join(self.model_dir, "nool_alpha_100m_final.pt"),
        ]
        for candidate in priority_candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def get_available_checkpoints(self) -> List[Dict[str, str]]:
        """Lists all safetensors model directories and .pt checkpoints found in model_dir."""
        if not os.path.isdir(self.model_dir):
            return []
        items = []

        # 1. Look for safetensors model directories
        for root, dirs, files in os.walk(self.model_dir):
            if "model.safetensors" in files:
                rel_name = os.path.relpath(root, self.model_dir).replace("\\", "/")
                safetensors_path = os.path.join(root, "model.safetensors")
                size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
                if "enx" in rel_name.lower():
                    category = "Stage 2 Distill (Factual Grounding)"
                elif "sft" in rel_name.lower():
                    category = "SFT Chat"
                else:
                    category = "HuggingFace Format"
                items.append({
                    "filename": rel_name,
                    "display_name": f"{rel_name} [{category}]",
                    "category": category,
                    "path": root,
                    "size_mb": round(size_mb, 1),
                    "is_dir": True,
                })

        # 2. Look for .pt checkpoint files
        for root, _, files in os.walk(self.model_dir):
            for fname in sorted(files):
                if fname.endswith(".pt"):
                    full_path = os.path.join(root, fname)
                    rel_name = os.path.relpath(full_path, self.model_dir).replace("\\", "/")
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    if "bridge" in rel_name.lower():
                        category = "Bridge Chat"
                    elif "sft" in rel_name.lower():
                        category = "SFT Model"
                    else:
                        category = "Base Pretrain"
                    items.append({
                        "filename": rel_name,
                        "display_name": f"{rel_name} [{category}]",
                        "category": category,
                        "path": full_path,
                        "size_mb": round(size_mb, 1),
                        "is_dir": False,
                    })
        return items

    def load_checkpoint(self, checkpoint_path_or_filename: str) -> None:
        """Loads or switches checkpoint state dict from either a directory or .pt file."""
        if os.path.isabs(checkpoint_path_or_filename):
            full_path = checkpoint_path_or_filename
        else:
            direct = os.path.join(self.model_dir, checkpoint_path_or_filename)
            if os.path.exists(direct):
                full_path = direct
            else:
                found = None
                for root, dirs, files in os.walk(self.model_dir):
                    base_target = os.path.basename(checkpoint_path_or_filename)
                    if base_target in dirs or base_target in files:
                        found = os.path.join(root, base_target)
                        break
                full_path = found if found else direct

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model path not found at: {full_path}")

        print(f"Loading model from: {full_path} on {self.device}...")
        t0 = time.time()

        # Handle Directory containing model.safetensors
        if os.path.isdir(full_path):
            import safetensors.torch
            safetensors_file = os.path.join(full_path, "model.safetensors")
            if not os.path.exists(safetensors_file):
                raise FileNotFoundError(f"model.safetensors not found in directory: {full_path}")

            config_file = os.path.join(full_path, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    c_dict = json.load(f)
                step = c_dict.get("total_steps", "Stage 2 Final")
                loss = c_dict.get("best_loss", None)
                for k in ["architectures", "model_type", "stage", "teacher_model", "best_loss", "total_steps"]:
                    c_dict.pop(k, None)
                self.config = NoolAlphaConfig(**c_dict)
            else:
                from nool_alpha.config import nool_100m
                self.config = nool_100m()
                step = "unknown"
                loss = None

            raw_sd = safetensors.torch.load_file(safetensors_file)
            clean_sd = remap_state_dict(raw_sd)

            # Load custom tokenizer if present in directory
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(full_path)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            except Exception:
                pass

        else:
            # Handle standard .pt file
            ckpt = torch.load(full_path, map_location=self.device, weights_only=False)
            raw_config = ckpt.get("config")
            if isinstance(raw_config, dict):
                self.config = NoolAlphaConfig(**raw_config)
            elif isinstance(raw_config, NoolAlphaConfig):
                self.config = raw_config
            else:
                from nool_alpha.config import nool_100m
                self.config = nool_100m()

            raw_sd = ckpt.get("model_state_dict", ckpt)
            clean_sd = remap_state_dict(raw_sd)
            step = ckpt.get("step", "unknown")
            loss = ckpt.get("loss", None)

        # Instantiate or recreate model on device
        self.model = NoolAlphaForCausalLM(self.config).to(self.device)
        missing, unexpected = self.model.load_state_dict(clean_sd, strict=False)
        self.model.eval()

        self.current_checkpoint_name = os.path.basename(full_path.rstrip("/\\"))
        self.checkpoint_metadata = {
            "step": step,
            "loss": round(loss, 4) if isinstance(loss, (int, float)) else loss,
            "filename": self.current_checkpoint_name,
            "path": full_path,
            "load_time_sec": round(time.time() - t0, 2),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }
        total_p, active_p = self.model.get_num_params()
        self.checkpoint_metadata["total_params_m"] = round(total_p / 1e6, 2)
        self.checkpoint_metadata["active_params_m"] = round(active_p / 1e6, 2)
        print(f"Loaded successfully in {self.checkpoint_metadata['load_time_sec']}s. Step/Status: {self.checkpoint_metadata['step']}")

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.35,
        top_p: float = 0.85,
        repetition_penalty: float = 1.25,
    ) -> Generator[Dict[str, Union[str, float, int]], None, None]:
        """
        Generates tokens autoregressively and yields each new token in real-time.
        Applies Alpaca instruction format: ### Instruction:\n{prompt}\n\n### Response:\n
        """
        if self.model is None:
            raise RuntimeError("No model checkpoint is loaded.")

        self.model.eval()

        # Wrap in Alpaca instruction template if not already wrapped
        if "### Instruction:" not in prompt and "### Response:" not in prompt:
            formatted_prompt = f"### Instruction:\n{prompt.strip()}\n\n### Response:\n"
        else:
            formatted_prompt = prompt

        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        start_time = time.time()
        accumulated_tokens = input_ids[0].tolist()
        generated_tokens = []

        for step in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_position_embeddings:
                idx = input_ids[:, -self.config.max_position_embeddings:]
            else:
                idx = input_ids

            logits, _, _, _ = self.model(idx)
            next_logits = logits[:, -1, :].clone() / max(temperature, 1e-5)

            # Apply repetition penalty
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

            if temperature <= 0.05:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            accumulated_tokens.append(token_id)
            generated_tokens.append(token_id)

            new_token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) / max(elapsed, 1e-5)

            # Extract clean response
            full_decoded = self.tokenizer.decode(accumulated_tokens[prompt_len:], skip_special_tokens=False)
            clean_text = full_decoded.replace("<|endoftext|>", "").strip()

            # Check stop conditions
            is_eos = (token_id == self.tokenizer.eos_token_id) or ("<|endoftext|>" in new_token_text) or ("\n###" in clean_text)
            if "\n###" in clean_text:
                clean_text = clean_text.split("\n###")[0].strip()

            yield {
                "token": new_token_text.replace("<|endoftext|>", ""),
                "accumulated_text": clean_text,
                "token_idx": step,
                "tok_per_sec": round(tok_per_sec, 1),
                "elapsed_sec": round(elapsed, 2),
                "is_finished": (step == max_new_tokens - 1) or is_eos,
            }

            if is_eos:
                break

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.35,
        top_p: float = 0.85,
        repetition_penalty: float = 1.25,
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
                "tokens_generated": 0,
                "elapsed_sec": 0.0,
                "tok_per_sec": 0.0,
            }

        return {
            "prompt": prompt,
            "completion": result["accumulated_text"],
            "tokens_generated": result["token_idx"] + 1,
            "elapsed_sec": result["elapsed_sec"],
            "tok_per_sec": result["tok_per_sec"],
        }
