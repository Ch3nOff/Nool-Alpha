"""
Comprehensive Intelligence & Reasoning Benchmark:
Nool-Alpha-100M-Reasoning vs. 100M-Class Peer Models:
  1. Nool-Alpha-100M (4.5h Deep Reasoning SFT, Step 3900)
  2. GPT-2 Small (124.4M Dense Baseline)
  3. SmolLM-135M-Instruct (134.5M Dense SOTA Instruct Baseline)

Evaluates:
  - Mathematical Reasoning (multi-step arithmetic & algebra)
  - Logical Deduction & Chain-of-Thought (<think> trace induction)
  - Python Code Generation & Syntax Validity (ast.parse)
  - Instruction Following & Structured Itemization
"""

import argparse
import ast
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM


INTELLIGENCE_TEST_SUITE = [
    # 1. Math Reasoning
    {
        "category": "Math",
        "id": "discount_tax",
        "prompt": "A store offers a 20% discount on a $150 jacket. If sales tax is 8%, what is the final price? Calculate step by step.",
    },
    {
        "category": "Math",
        "id": "algebra_solve",
        "prompt": "Solve for x: 3x + 7 = 22. Show your step-by-step reasoning.",
    },
    {
        "category": "Math",
        "id": "speed_distance",
        "prompt": "A car travels at a constant speed of 60 km/h for 2.5 hours. How far does the car travel in total?",
    },

    # 2. Logical Deduction & Step-by-Step Thinking
    {
        "category": "Logic",
        "id": "odd_numbers",
        "prompt": "Explain step by step why the sum of two odd integers is always an even integer.",
    },
    {
        "category": "Logic",
        "id": "apple_riddle",
        "prompt": "There are 5 apples in a basket. You take away 3 apples. How many apples do you have? Think carefully.",
    },
    {
        "category": "Logic",
        "id": "syllogism",
        "prompt": "Premise 1: All cats are animals. Premise 2: Some animals are black. Question: Does it logically follow that all cats are black? Explain.",
    },

    # 3. Python Code & Syntax Validity
    {
        "category": "Code",
        "id": "is_palindrome",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if a string s is a palindrome and False otherwise.",
    },
    {
        "category": "Code",
        "id": "factorial",
        "prompt": "Write a Python function `factorial(n)` that computes the factorial of a positive integer n.",
    },
    {
        "category": "Code",
        "id": "find_max",
        "prompt": "Write a Python function `find_max(numbers)` that returns the largest number in a list.",
    },

    # 4. Instruction Following & Structured Output
    {
        "category": "Instruct",
        "id": "planets_list",
        "prompt": "List 3 distinct planets in our solar system as numbered bullet points with one key feature for each.",
    },
    {
        "category": "Instruct",
        "id": "photosynthesis",
        "prompt": "Explain the concept of photosynthesis in two clear sentences.",
    },
]


def extract_code(text: str) -> str:
    pattern = r"```python(.*?)```"
    m = re.findall(pattern, text, re.DOTALL)
    if m:
        return m[0].strip()
    pattern_any = r"```(.*?)```"
    m2 = re.findall(pattern_any, text, re.DOTALL)
    if m2:
        return m2[0].strip()
    lines = [l for l in text.split("\n") if "def " in l or "    " in l]
    return "\n".join(lines) if lines else text.strip()


def validate_ast(code: str) -> Tuple[bool, str]:
    if not code or len(code) < 5:
        return False, "Empty or truncated code"
    try:
        ast.parse(code)
        return True, "Valid AST"
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg}"
    except Exception as e:
        return False, f"ParseError: {e}"


class ModelRunner:
    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 120) -> Tuple[str, float, float]:
        raise NotImplementedError


class NoolAlphaRunner(ModelRunner):
    def __init__(self, model_dir: str, device: torch.device):
        super().__init__("Nool-Alpha-100M (Reasoning SFT)", device)
        cfg_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "model.safetensors")

        with open(cfg_path, "r", encoding="utf-8") as f:
            c = json.load(f)

        self.config = NoolAlphaConfig(
            vocab_size=c["vocab_size"],
            d_model=c["d_model"],
            n_layers=c["n_layers"],
            num_heads=c["num_heads"],
            head_dim=c["head_dim"],
            d_c=c["d_c"],
            d_pe=c["d_pe"],
            shared_ffn_dim=c.get("shared_ffn_dim", 1536),
            num_experts=c.get("num_experts", 8),
            top_k_experts=c.get("top_k_experts", 2),
            expert_rank=c.get("expert_rank", 96),
            logit_soft_cap=c.get("logit_soft_cap", 30.0),
        )
        self.model = NoolAlphaForCausalLM(self.config).to(device)
        state = load_file(weights_path)
        self.model.load_state_dict(state)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str, max_new_tokens: int = 120) -> Tuple[str, float, float]:
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        enc = self.tokenizer.encode(formatted, return_tensors="pt").to(self.device)
        prompt_len = enc.shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = enc
            for _ in range(max_new_tokens):
                idx = out[:, -self.config.max_position_embeddings:] if out.shape[1] > self.config.max_position_embeddings else out
                logits, _, _, _ = self.model(idx)
                nxt = logits[:, -1, :] / 0.55

                # Repetition penalty
                for prev in set(out[0].tolist()):
                    if nxt[0, prev] > 0:
                        nxt[0, prev] /= 1.15
                    else:
                        nxt[0, prev] *= 1.15

                probs = F.softmax(nxt, dim=-1)
                sp, si = torch.sort(probs, descending=True)
                cp = torch.cumsum(sp, dim=-1)
                si_rem = cp > 0.85
                si_rem[..., 1:] = si_rem[..., :-1].clone()
                si_rem[..., 0] = 0
                rem = si_rem.scatter(1, si, si_rem)
                probs = probs.masked_fill(rem, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

                tok = torch.multinomial(probs, num_samples=1)
                out = torch.cat([out, tok], dim=-1)
                if tok.item() == self.tokenizer.eos_token_id:
                    break

        elapsed = max(time.time() - t0, 1e-4)
        gen_tokens = out.shape[1] - prompt_len
        tok_speed = gen_tokens / elapsed

        raw_decoded = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        return raw_decoded, elapsed, tok_speed


class HuggingFaceModelRunner(ModelRunner):
    def __init__(self, model_id: str, display_name: str, device: torch.device, is_chat: bool = False):
        super().__init__(display_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model.eval()
        self.is_chat = is_chat

    def generate(self, prompt: str, max_new_tokens: int = 120) -> Tuple[str, float, float]:
        if self.is_chat and hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

        enc = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=0.55,
                top_p=0.85,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = max(time.time() - t0, 1e-4)
        gen_tokens = out.shape[1] - prompt_len
        tok_speed = gen_tokens / elapsed

        completion = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        return completion, elapsed, tok_speed


def run_intelligence_benchmark(nool_dir: str, output_report: str = "BENCHMARK_INTELLIGENCE_100M.md"):
    console = Console()
    console.print(Panel(
        "[bold cyan]Head-to-Head Intelligence & Reasoning Benchmark (100M-Class Models)[/bold cyan]\n"
        "[dim]Empirical comparison on Math, Logic, Python Code, and Instruction Following[/dim]",
        border_style="cyan"
    ))

    device = torch.device("cpu")
    console.print(f"[+] Running on Hardware: [bold]{device}[/bold]")

    console.print("[+] Initializing Nool-Alpha-100M Reasoning Runner...")
    nool_runner = NoolAlphaRunner(nool_dir, device)

    console.print("[+] Initializing GPT-2 Small (124M) Runner...")
    gpt2_runner = HuggingFaceModelRunner("gpt2", "GPT-2 Small (124M)", device, is_chat=False)

    console.print("[+] Initializing SmolLM-135M-Instruct Runner...")
    smol_runner = HuggingFaceModelRunner("HuggingFaceTB/SmolLM-135M-Instruct", "SmolLM-135M-Instruct", device, is_chat=True)

    runners = [nool_runner, gpt2_runner, smol_runner]

    # Metrics storage
    benchmark_results = {r.name: [] for r in runners}
    ast_scores = {r.name: {"passed": 0, "total": 0} for r in runners}
    cot_counts = {r.name: 0 for r in runners}
    speed_records = {r.name: [] for r in runners}

    # Execute tests
    for test in INTELLIGENCE_TEST_SUITE:
        cat = test["category"]
        t_id = test["id"]
        prompt = test["prompt"]

        console.print(f"\n[bold yellow]=== [{cat}] {t_id} ===[/bold yellow]")
        console.print(f"[dim]Prompt:[/dim] {prompt}")

        for r in runners:
            resp, elapsed, speed = r.generate(prompt, max_new_tokens=100)
            speed_records[r.name].append(speed)

            # Check Chain of Thought indicator (<think>, step-by-step)
            has_cot = ("<think>" in resp.lower()) or ("step by step" in resp.lower()) or ("first," in resp.lower()) or ("let's solve" in resp.lower())
            if has_cot:
                cot_counts[r.name] += 1

            # Check AST validity for Code tasks
            ast_status = "N/A"
            if cat == "Code":
                code_snippet = extract_code(resp)
                valid, reason = validate_ast(code_snippet)
                ast_scores[r.name]["total"] += 1
                if valid:
                    ast_scores[r.name]["passed"] += 1
                    ast_status = "[green]PASS[/green]"
                else:
                    ast_status = f"[red]FAIL ({reason[:18]})[/red]"

            preview = resp[:90].replace("\n", " ")
            console.print(f"  • [bold]{r.name:30s}[/bold] | Speed: {speed:4.1f} t/s | CoT: {'YES' if has_cot else 'NO ':3s} | AST: {ast_status:15s} | Preview: {preview}...")

            benchmark_results[r.name].append({
                "category": cat,
                "id": t_id,
                "prompt": prompt,
                "response": resp,
                "speed": speed,
                "has_cot": has_cot,
                "ast_status": ast_status,
            })

    # Summary Table
    summary_table = Table(title="[bold cyan]Intelligence Benchmark Executive Scorecard[/bold cyan]")
    summary_table.add_column("Evaluation Dimension", style="bold")
    summary_table.add_column("GPT-2 Small (124M)", justify="center", style="red")
    summary_table.add_column("SmolLM-135M-Instruct", justify="center", style="yellow")
    summary_table.add_column("Nool-Alpha-100M-Reasoning", justify="center", style="bold green")

    # Parameters
    summary_table.add_row("Total Parameters", "124.4M", "134.5M", "111.2M")
    summary_table.add_row("Active Parameters / Token", "124.4M (100%)", "134.5M (100%)", "97.9M (88.0%)")
    summary_table.add_row("KV-Cache Scaling", "Dense MHA (1536 f/t)", "Dense GQA (384 f/t)", "GSLA (224 f/t, -87.8%)")

    # Average Speed
    gpt2_spd = sum(speed_records["GPT-2 Small (124M)"]) / len(speed_records["GPT-2 Small (124M)"])
    smol_spd = sum(speed_records["SmolLM-135M-Instruct"]) / len(speed_records["SmolLM-135M-Instruct"])
    nool_spd = sum(speed_records["Nool-Alpha-100M (Reasoning SFT)"]) / len(speed_records["Nool-Alpha-100M (Reasoning SFT)"])
    summary_table.add_row("Average CPU Throughput", f"{gpt2_spd:.1f} tok/s", f"{smol_spd:.1f} tok/s", f"{nool_spd:.1f} tok/s")

    # CoT / Reasoning Emergence Rate
    tot_tasks = len(INTELLIGENCE_TEST_SUITE)
    gpt2_cot_rate = (cot_counts["GPT-2 Small (124M)"] / tot_tasks) * 100
    smol_cot_rate = (cot_counts["SmolLM-135M-Instruct"] / tot_tasks) * 100
    nool_cot_rate = (cot_counts["Nool-Alpha-100M (Reasoning SFT)"] / tot_tasks) * 100
    summary_table.add_row("CoT Reasoning Induction Rate", f"{gpt2_cot_rate:.1f}%", f"{smol_cot_rate:.1f}%", f"{nool_cot_rate:.1f}%")

    # AST Code Validation
    def fmt_ast(sc):
        t = sc["total"]
        p = sc["passed"]
        pct = (p / max(t, 1)) * 100
        return f"{p}/{t} ({pct:.1f}%)"

    summary_table.add_row("Python Code AST Pass Rate",
        fmt_ast(ast_scores["GPT-2 Small (124M)"]),
        fmt_ast(ast_scores["SmolLM-135M-Instruct"]),
        fmt_ast(ast_scores["Nool-Alpha-100M (Reasoning SFT)"])
    )

    console.print("\n")
    console.print(summary_table)

    # Write Markdown Report
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(f"""# Nool-Alpha-100M vs. 100M-Class Peer Intelligence & Reasoning Benchmark

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Scope:** Cognitive & Intelligence Benchmark across 100M-parameter models on Math, Logic, Python Code, and Instruction Following.  
**Hardware Platform:** CPU Execution (Direct head-to-head on identical hardware).

---

## 1. Executive Scorecard

| Evaluation Dimension | GPT-2 Small (124M) | SmolLM-135M-Instruct | Nool-Alpha-100M (Reasoning SFT) |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 124.4M | 134.5M | **111.17M** |
| **Active Parameters / Token** | 124.4M (100%) | 134.5M (100%) | **97.90M (88.0%)** |
| **Attention Architecture** | Dense MHA | Dense GQA (3:1) | **GSLA (Latent Subspace)** |
| **KV-Cache Memory Footprint** | 1536 floats/token/layer | 384 floats/token/layer | **224 floats/token/layer (-87.8%)** |
| **Average Decoding Speed** | {gpt2_spd:.1f} tok/s | {smol_spd:.1f} tok/s | **{nool_spd:.1f} tok/s** |
| **Step-by-Step CoT Emergence Rate** | {gpt2_cot_rate:.1f}% | {smol_cot_rate:.1f}% | **{nool_cot_rate:.1f}%** |
| **Python AST Syntax Pass Rate** | {fmt_ast(ast_scores["GPT-2 Small (124M)"])} | {fmt_ast(ast_scores["SmolLM-135M-Instruct"])} | **{fmt_ast(ast_scores["Nool-Alpha-100M (Reasoning SFT)"])}** |

---

## 2. Qualitative Output Analysis by Domain

""")
        for i, test in enumerate(INTELLIGENCE_TEST_SUITE):
            f.write(f"### Test {i+1} [{test['category']}]: `{test['id']}`\n")
            f.write(f"**Prompt:**\n> {test['prompt']}\n\n")

            for r_name in [nool_runner.name, smol_runner.name, gpt2_runner.name]:
                item = benchmark_results[r_name][i]
                f.write(f"#### {r_name}:\n")
                f.write(f"```text\n{item['response']}\n```\n")
                f.write(f"- Speed: `{item['speed']:.1f} tok/s` | CoT Triggered: `{item['has_cot']}` | AST: `{item['ast_status']}`\n\n")

    console.print(f"[OK] Full comparative report successfully written to: [bold]{output_report}[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Head-to-Head Intelligence Benchmark for 100M Models.")
    parser.add_argument(
        "--nool_dir",
        type=str,
        default=r"C:\Users\Matthew Chen\Downloads\Nool_alpha model\model",
        help="Path to Nool-Alpha exported reasoning model directory",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="BENCHMARK_INTELLIGENCE_100M.md",
        help="Path to markdown output report",
    )
    args = parser.parse_args()

    run_intelligence_benchmark(args.nool_dir, args.report)
