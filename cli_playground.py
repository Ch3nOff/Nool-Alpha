"""
Interactive Terminal Playground for Nool-Alpha-100M.
Features real-time token streaming, parameter adjustment, and checkpoint switching.
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nool_alpha.infer import NoolAlphaInference, DEFAULT_MODEL_DIR

PRESETS = {
    "1": ("English Reasoning", "Artificial intelligence will transform the future of"),
    "2": ("Bahasa Indonesia", "Ibu kota Nusantara (IKN) merupakan pusat pemerintahan baru"),
    "3": ("Python Coding", "def quick_sort(arr):\n    # Implement quicksort in python\n"),
}


def print_banner(console: Console, engine: NoolAlphaInference):
    meta = engine.checkpoint_metadata
    table = Table(title="[bold cyan]Nool-Alpha-100M Architecture Specs[/bold cyan]", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="bold green")

    table.add_row("Active Checkpoint", f"{meta.get('filename', 'unknown')} (Step: {meta.get('step', 'N/A')})")
    table.add_row("Training Loss", f"{meta.get('loss', 'N/A')}")
    table.add_row("Total Parameters", f"{meta.get('total_params_m', 111.2)}M")
    table.add_row("Active Params / Token", f"{meta.get('active_params_m', 97.9)}M")
    table.add_row("Attention Mechanism", "GSLA (Grouped-Subspace Latent Attention, d_c=192, d_pe=32)")
    table.add_row("Feed-Forward", "HFK-MoE (Dense SwiGLU shared + 8 Low-Rank Experts, Top-2)")
    table.add_row("Context Window", f"{engine.config.max_position_embeddings} tokens")
    table.add_row("Device", str(engine.device))

    console.print(table)
    console.print(Panel(
        "[bold yellow]Commands:[/bold yellow]\n"
        "  [cyan]/model <name|num>[/cyan]  - Switch checkpoint (e.g. /model best, /model final)\n"
        "  [cyan]/preset <1|2|3>[/cyan]     - Run multi-domain preset prompt\n"
        "  [cyan]/params[/cyan]              - View or change generation hyperparameters\n"
        "  [cyan]/help[/cyan]                - Show this help message\n"
        "  [cyan]/exit[/cyan]                - Exit playground",
        title="[bold green]Interactive Controls[/bold green]",
        border_style="cyan"
    ))


def run_cli_playground():
    console = Console()
    console.print("[bold green]Starting Nool-Alpha Interactive Playground...[/bold green]")

    engine = NoolAlphaInference()
    print_banner(console, engine)

    # Default sampling parameters
    params = {
        "max_new_tokens": 50,
        "temperature": 0.6,
        "top_p": 0.85,
        "repetition_penalty": 1.2,
    }

    while True:
        try:
            prompt = console.input("\n[bold cyan]Nool-Alpha >> [/bold cyan]").strip()
            if not prompt:
                continue

            if prompt.lower() in ["/exit", "/quit", "exit", "quit"]:
                console.print("[yellow]Exiting playground. Sampai jumpa![/yellow]")
                break

            if prompt.lower() in ["/help", "help"]:
                print_banner(console, engine)
                continue

            if prompt.lower().startswith("/model"):
                parts = prompt.split()
                if len(parts) == 1:
                    ckpts = engine.get_available_checkpoints()
                    console.print("[bold yellow]Available Checkpoints:[/bold yellow]")
                    for idx, c in enumerate(ckpts, 1):
                        current = " [bold green](ACTIVE)[/bold green]" if c["filename"] == engine.current_checkpoint_name else ""
                        console.print(f"  [{idx}] {c['filename']} ({c['size_mb']} MB){current}")
                else:
                    target = parts[1]
                    if target.lower() in ["best", "1"]:
                        engine.load_checkpoint("best_checkpoint.pt")
                    elif target.lower() in ["final", "2"]:
                        engine.load_checkpoint("nool_alpha_100m_final.pt")
                    else:
                        engine.load_checkpoint(target)
                    console.print(f"[bold green]Switched active checkpoint to:[/bold green] {engine.current_checkpoint_name}")
                continue

            if prompt.lower().startswith("/preset"):
                parts = prompt.split()
                preset_id = parts[1] if len(parts) > 1 else "1"
                if preset_id in PRESETS:
                    domain_name, prompt = PRESETS[preset_id]
                    console.print(f"[bold magenta]Running Preset [{preset_id}] - {domain_name}:[/bold magenta]")
                    console.print(f"[italic]{prompt}[/italic]\n")
                else:
                    console.print("[red]Invalid preset id. Use 1, 2, or 3.[/red]")
                    continue

            if prompt.lower() == "/params":
                console.print("[bold yellow]Current Generation Parameters:[/bold yellow]")
                for k, v in params.items():
                    console.print(f"  {k}: [bold green]{v}[/bold green]")
                console.print("[dim]Usage to update: /params <key>=<value>, e.g., /params temperature=0.7[/dim]")
                continue

            if prompt.lower().startswith("/params "):
                assignment = prompt[8:].strip()
                if "=" in assignment:
                    k, v = assignment.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in params:
                        val_type = type(params[k])
                        params[k] = val_type(v)
                        console.print(f"[bold green]Updated {k} -> {params[k]}[/bold green]")
                    else:
                        console.print(f"[red]Unknown parameter: {k}[/red]")
                continue

            # Generate response with streaming output
            console.print("[dim]--- Generated Completion ---[/dim]")
            console.print(f"[bold]{prompt}[/bold]", end="")

            last_len = 0
            stats = None
            for chunk in engine.stream_generate(
                prompt=prompt,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"],
            ):
                console.print(chunk["token"], end="")
                stats = chunk

            console.print("\n")
            if stats:
                console.print(
                    f"[dim]Stats: {stats['token_idx'] + 1} tokens generated in "
                    f"{stats['elapsed_sec']}s ({stats['tok_per_sec']} tok/s)[/dim]"
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error during generation: {e}[/red]")


if __name__ == "__main__":
    run_cli_playground()
