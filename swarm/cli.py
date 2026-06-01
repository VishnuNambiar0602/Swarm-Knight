"""Swarm-Knight CLI - Multi-LLM Swarm Intelligence Tool."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box

app = typer.Typer(
    name="knight",
    help="Swarm-Knight - Multi-LLM Collaboration via Debate/Refinement",
    add_completion=False,
)
console = Console()

# Version
__version__ = "0.1.0"


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]Swarm-Knight[/bold cyan] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit."
    ),
):
    """
    Swarm-Knight - Multi-LLM Collaboration via Debate/Refinement

    Run coding tasks with multiple AI models working together.
    """
    pass


@app.command()
def run(
    task: str = typer.Argument(..., help="The coding task to execute"),
    preset: str = typer.Option("coding", "--preset", "-p", help="Preset: coding, ecommerce, custom"),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Max debate rounds (1-10)"),
    models: Optional[str] = typer.Option(None, "--models", "-m", help="Comma-separated model names"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="OpenRouter API key"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show detailed output"),
):
    """
    Run a swarm task with multiple AI models.

    Examples:

        swarm run "Build a shopping cart component"

        swarm run "Create a REST API" --preset coding --rounds 5

        swarm run "Design an e-commerce site" --preset ecommerce
    """
    asyncio.run(_run_swarm(task, preset, rounds, models, api_key, verbose))


async def _run_swarm(
    task: str,
    preset: str,
    rounds: int,
    models: Optional[str],
    api_key: Optional[str],
    verbose: bool,
):
    """Execute the swarm task."""
    from .providers import OpenRouterProvider
    from .orchestrator import SwarmOrchestrator
    from .models import (
        SwarmParticipant, SwarmConfig, ProviderType, ParticipantRole,
        get_coding_swarm_config, get_ecommerce_swarm_config,
    )

    # Print banner
    _print_banner()

    # Get API key
    if not api_key:
        import os
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            api_key = Prompt.ask(
                "\n[bold cyan]Enter your OpenRouter API key[/bold cyan]",
                password=True
            )
            if not api_key:
                console.print("[red]API key required. Get one at https://openrouter.ai/keys[/red]")
                raise typer.Exit(1)

    # Get participants based on preset
    if preset == "ecommerce":
        participants = get_ecommerce_swarm_config()
    else:
        participants = get_coding_swarm_config()

    # Override models if specified
    if models:
        model_list = [m.strip() for m in models.split(",")]
        participants = []
        for i, model in enumerate(model_list):
            participants.append(SwarmParticipant(
                name=f"Model-{i+1}",
                provider=ProviderType.OPENROUTER,
                model=model,
                role=ParticipantRole.GENERATOR if i < len(model_list) - 1 else ParticipantRole.CRITIC,
            ))

    # Set API key on all participants
    for p in participants:
        p.api_key = api_key

    # Print task info
    _print_task_info(task, participants, rounds)

    # Create config
    config = SwarmConfig(max_rounds=rounds)

    # Create orchestrator
    orchestrator = SwarmOrchestrator()

    # Progress callback
    def on_progress(event: str, data: dict):
        if event == "round_start":
            console.print(f"\n[bold yellow]  Round {data['round']} started...[/bold yellow]")
        elif event == "round_complete":
            score = data.get("consensus_score", 0)
            console.print(f"  [dim]Consensus: {score:.0%}[/dim]")
        elif event == "consensus":
            console.print(f"\n[bold green]  Consensus reached![/bold green]")
        elif event == "completed":
            console.print(f"\n[bold green]  Completed in {data.get('duration', 0):.1f}s[/bold green]")

    orchestrator.set_progress_callback("cli", on_progress)

    # Run with live display
    with Live(console=console, refresh_per_second=4) as live:
        # Create session
        session = await orchestrator.create_session(
            task=task,
            participants=participants,
            config=config,
        )

        # Update display
        live.update(_build_running_panel(session))

        # Run session
        try:
            result = await orchestrator.run_session(session.id)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # Print result
    _print_result(result)


def _print_banner():
    """Print the CLI banner."""
    banner = """
 ███╗   ███╗██╗███╗   ██╗███████╗ ██████╗ █████╗ ███╗   ██╗     ██████╗ ███████╗███╗   ██╗
 ████╗ ████║██║████╗  ██║██╔════╝██╔════╝██╔══██╗████╗  ██║     ██╔══██╗██╔════╝████╗  ██║
 ██╔████╔██║██║██╔██╗ ██║███████╗██║     ███████║██╔██╗ ██║     ██████╔╝█████╗  ██╔██╗ ██║
 ██║╚██╔╝██║██║██║╚██╗██║╚════██║██║     ██╔══██║██║╚██╗██║     ██╔══██╗██╔══╝  ██║╚██╗██║
 ██║ ╚═╝ ██║██║██║ ╚████║███████║╚██████╗██║  ██║██║ ╚████║     ██║  ██║███████╗██║ ╚████║
 ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝"""

    console.print()
    console.print(Align.center(Text(banner, style="bold cyan")))
    console.print(Align.center(Text("Multi-LLM Collaboration via Debate/Refinement", style="dim")))
    console.print()
    console.print(Align.center(banner))
    console.print(Align.center(Text("Multi-LLM Collaboration via Debate/Refinement", style="dim")))
    console.print()


def _print_task_info(task: str, participants, rounds: int):
    """Print task information."""
    table = Table(
        title="Task Configuration",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=False,
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Task", task[:60] + "..." if len(task) > 60 else task)
    table.add_row("Models", str(len(participants)))
    table.add_row("Max Rounds", str(rounds))
    table.add_row("Pattern", "Debate/Refinement")

    console.print()
    console.print(table)

    # Show participants
    ptable = Table(
        title="Participants",
        box=box.ROUNDED,
        border_style="green",
    )
    ptable.add_column("Name", style="bold")
    ptable.add_column("Model")
    ptable.add_column("Role")

    for p in participants:
        ptable.add_row(p.name, p.model, p.role.value)

    console.print()
    console.print(ptable)
    console.print()


def _build_running_panel(session) -> Panel:
    """Build a panel showing running status."""
    content = Text()
    content.append("  Swarm is running...\n\n", style="bold yellow")
    content.append(f"  Session: {session.id[:8]}...\n", style="dim")
    content.append(f"  Status: {session.status.value}\n", style="cyan")

    return Panel(
        content,
        title="[bold]Swarm Status[/bold]",
        border_style="yellow",
        box=box.ROUNDED,
    )


def _print_result(result):
    """Print the final result."""
    console.print()
    console.print(Panel(
        f"[bold green]Swarm Completed![/bold green]\n\n"
        f"Best solution from: [cyan]{result.best_participant_name}[/cyan]\n"
        f"Rounds: {result.rounds_completed}\n"
        f"Consensus: {'Yes' if result.consensus_reached else 'No'}\n"
        f"Time: {result.duration_seconds:.1f}s",
        title="Result",
        border_style="green",
        box=box.ROUNDED,
    ))

    # Show the output
    console.print(Panel(
        result.output,
        title="[bold]Generated Code[/bold]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


@app.command()
def models(
    provider: str = typer.Option("openrouter", "--provider", "-p", help="Provider: openrouter, ollama"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key"),
):
    """
    List available models.
    """
    _print_banner()

    if provider == "openrouter":
        _list_openrouter_models(api_key)
    elif provider == "ollama":
        _list_ollama_models()
    else:
        console.print(f"[red]Unknown provider: {provider}[/red]")


def _list_openrouter_models(api_key: Optional[str]):
    """List free OpenRouter models."""
    table = Table(
        title="Free OpenRouter Models",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Model", style="bold cyan")
    table.add_column("Best For")
    table.add_column("Context")

    models = [
        ("nvidia/nemotron-3-super-49b:free", "Planning, QA", "1M"),
        ("openai/gpt-oss-120b:free", "Architecture, Logic", "128K"),
        ("minimax/minimax-m2.5:free", "Layout, Visual", "1M"),
        ("google/gemma-4-31b:free", "Color, Style", "128K"),
        ("poolside/poolside-laguna-m-1:free", "Coding", "128K"),
        ("poolside/poolside-laguna-xs-2:free", "Fast Iteration", "128K"),
        ("moonshotai/kimi-k2.6:free", "Full-page Assembly", "1M"),
        ("nvidia/nemotron-nano-12b-2-vl:free", "Multimodal, OCR", "128K"),
        ("z-ai/glm-4.5-air:free", "Tagging, Extraction", "128K"),
    ]

    for model, use_case, ctx in models:
        table.add_row(model, use_case, ctx)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Get API key: https://openrouter.ai/keys[/dim]")


def _list_ollama_models():
    """List local Ollama models."""
    import httpx

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            table = Table(
                title="Local Ollama Models",
                box=box.ROUNDED,
                border_style="green",
            )
            table.add_column("Model", style="bold green")
            table.add_column("Size")

            for m in models:
                size_gb = m.get("size", 0) / (1024**3)
                table.add_row(m["name"], f"{size_gb:.1f} GB")

            console.print()
            console.print(table)
        else:
            console.print("[yellow]Ollama is running but returned an error[/yellow]")
    except Exception:
        console.print("[red]Ollama not running. Start it with: ollama serve[/red]")
        console.print("[dim]Then pull a model: ollama pull llama3.2[/dim]")


@app.command()
def init():
    """
    Initialize swarm configuration.
    """
    _print_banner()

    console.print("[bold]Let's set up your swarm![/bold]\n")

    # Get API key
    api_key = Prompt.ask(
        "[cyan]OpenRouter API key[/cyan] (get one at https://openrouter.ai/keys)",
        password=True,
    )

    if not api_key:
        console.print("[red]API key required[/red]")
        raise typer.Exit(1)

    # Save to config file
    config_dir = Path.home() / ".swarm-knight"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "config.json"
    import json

    config = {
        "api_key": api_key,
        "default_preset": "coding",
        "default_rounds": 3,
    }

    config_file.write_text(json.dumps(config, indent=2))
    console.print(f"\n[green]Saved config to {config_file}[/green]")
    console.print("[dim]You can now run: knight run \"your task\"[/dim]")


@app.command()
def presets():
    """
    Show available presets.
    """
    _print_banner()

    table = Table(
        title="Available Presets",
        box=box.ROUNDED,
        border_style="magenta",
    )
    table.add_column("Preset", style="bold magenta")
    table.add_column("Description")
    table.add_column("Models")

    table.add_row(
        "coding",
        "General coding tasks with debate",
        "Laguna M.1, Kimi K2.6, Nemotron",
    )
    table.add_row(
        "ecommerce",
        "Full e-commerce development",
        "6 specialized models",
    )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Usage: swarm run \"task\" --preset coding[/dim]")


if __name__ == "__main__":
    app()
