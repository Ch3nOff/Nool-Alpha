"""
Automated Upload Script for Nool-Alpha-100M to Hugging Face Hub.

Uploads:
  - model.safetensors
  - config.json
  - tokenizer files (tokenizer.json, tokenizer_config.json, vocab.json, etc.)
  - README.md (Model Card)
"""

import argparse
import os
import sys
from huggingface_hub import HfApi, login

def upload_nool_alpha(
    repo_id: str,
    model_dir: str = "exported_models/nool_alpha_100m_sft_best",
    token: str = None,
    private: bool = False,
):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    # Determine token
    hf_token = token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info.get("name")
    print(f"[+] Authenticated as Hugging Face user: {username}")

    # Resolve full repo_id if only model name is provided
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    print(f"[+] Target Hugging Face Repository: https://huggingface.co/{repo_id}")
    print(f"[+] Creating / Verifying repository...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    print(f"[+] Uploading files from '{model_dir}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload Nool-Alpha-100M weights and artifacts from {os.path.basename(model_dir)}",
    )
    print(f"\n[OK] Successfully uploaded Nool-Alpha to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Nool-Alpha to Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Nool-Alpha-100M-Chat",
        help="Repository ID on Hugging Face (e.g., 'username/Nool-Alpha-100M-Chat' or just 'Nool-Alpha-100M-Chat')",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="exported_models/nool_alpha_100m_sft_best",
        help="Path to exported model folder containing model.safetensors and config.json",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face Write Access Token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    args = parser.parse_args()

    upload_nool_alpha(
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        token=args.token,
        private=args.private,
    )
