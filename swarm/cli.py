"""Swarm-Knight CLI - Multi-LLM Swarm Intelligence Tool."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich import box

app = typer.Typer(
    name="knight",
    help="Swarm-Knight - Multi-LLM Collaboration via Debate/Refinement",
    add_completion=False,
)
console = Console()

__version__ = "2.0.0"


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
    preset: str = typer.Option("auto", "--preset", "-p", help="Preset: auto, coding, ecommerce"),
    rounds: int = typer.Option(5, "--rounds", "-r", help="Max debate rounds (1-20)"),
    models: Optional[str] = typer.Option(None, "--models", "-m", help="Comma-separated model names"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="OpenRouter API key"),
    max_agents: int = typer.Option(6, "--agents", "-a", help="Max agents (2-20)"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show detailed output"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable memory system"),
    no_reputation: bool = typer.Option(False, "--no-reputation", help="Disable reputation tracking"),
):
    """
    Run a swarm task with multiple AI models.

    Examples:

        knight run "Build a shopping cart component"

        knight run "Create a REST API" --preset coding --rounds 5

        knight run "Design an e-commerce site" --agents 8
    """
    asyncio.run(_run_swarm(task, preset, rounds, models, api_key, max_agents, verbose, no_memory, no_reputation))


async def _run_swarm(
    task: str, preset: str, rounds: int, models: Optional[str],
    api_key: Optional[str], max_agents: int, verbose: bool,
    no_memory: bool, no_reputation: bool,
):
    from .providers import OpenRouterProvider
    from .orchestrator import SwarmOrchestrator
    from .models import (
        SwarmParticipant, SwarmConfig, ProviderType, ParticipantRole, FREE_MODELS,
    )

    _print_banner()

    if not api_key:
        import os
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            api_key = Prompt.ask("\n[bold cyan]Enter your OpenRouter API key[/bold cyan]", password=True)
            if not api_key:
                console.print("[red]API key required. Get one at https://openrouter.ai/keys[/red]")
                raise typer.Exit(1)

    participants = None
    if models:
        model_list = [m.strip() for m in models.split(",")]
        participants = []
        for i, model in enumerate(model_list):
            participants.append(SwarmParticipant(
                name=f"Model-{i+1}",
                provider=ProviderType.OPENROUTER,
                model=model,
                role=ParticipantRole.GENERATOR if i < len(model_list) - 1 else ParticipantRole.CRITIC,
                api_key=api_key,
            ))

    config = SwarmConfig(
        max_rounds=rounds,
        max_agents=max_agents,
        enable_memory=not no_memory,
        enable_reputation=not no_reputation,
        enable_dynamic_agents=(preset == "auto"),
    )

    _print_task_info(task, participants, config)

    orchestrator = SwarmOrchestrator()

    def on_progress(event: str, data: dict):
        if event == "round_start":
            console.print(f"\n[bold yellow]  Round {data['round']} started...[/bold yellow]")
        elif event == "round_complete":
            score = data.get("consensus_score", 0)
            improvement = data.get("improvement", 0)
            arrow = "[green]+[/green]" if improvement > 0 else "[red]~[/red]"
            console.print(f"  [dim]Consensus: {score:.0%} {arrow}[/dim]")
        elif event == "consensus":
            console.print(f"\n[bold green]  Consensus reached![/bold green]")
        elif event == "merging":
            console.print(f"\n[bold cyan]  Merging solutions...[/bold cyan]")
        elif event == "completed":
            console.print(f"\n[bold green]  Completed in {data.get('duration', 0):.1f}s[/bold green]")

    orchestrator.set_progress_callback("cli", on_progress)

    with Live(console=console, refresh_per_second=4) as live:
        session = await orchestrator.create_session(
            task=task,
            participants=participants,
            config=config,
            api_key=api_key,
        )
        live.update(_build_running_panel(session))
        try:
            result = await orchestrator.run_session(session.id)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            raise typer.Exit(1)

    _print_result(result)

    stats = orchestrator.get_stats()
    if verbose:
        _print_stats(stats)


def _print_banner():
    banner = """
 ███╗   ███╗██╗███╗   ██╗███████╗ ██████╗ █████╗ ███╗   ██╗     ██████╗ ███████╗███╗   ██╗
 ████╗ ████║██║████╗  ██║██╔════╝██╔════╝██╔══██╗████╗  ██║     ██╔══██╗██╔════╝████╗  ██║
 ██╔████╔██║██║██╔██╗ ██║███████╗██║     ███████║██╔██╗ ██║     ██████╔╝█████╗  ██╔██╗ ██║
 ██║╚██╔╝██║██║██║╚██╗██║╚════██║██║     ██╔══██║██║╚██╗██║     ██╔══██╗██╔══╝  ██║╚██╗██║
 ██║ ╚═╝ ██║██║██║ ╚████║███████║╚██████╗██║  ██║██║ ╚████║     ██║  ██║███████╗██║ ╚████║
 ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝"""
    console.print()
    console.print(Align.center(Text(banner, style="bold cyan")))
    console.print(Align.center(Text("v2.0 - Consensus Engine with Memory & Reputation", style="dim")))
    console.print()


def _print_task_info(task: str, participants, config: SwarmConfig):
    from .dynamic_agents import dynamic_generator

    plan = dynamic_generator.analyze_task(task)

    table = Table(title="Task Analysis", box=box.ROUNDED, border_style="cyan", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Task", task[:60] + "..." if len(task) > 60 else task)
    table.add_row("Type", plan.task_type)
    table.add_row("Complexity", f"{plan.complexity:.0%}")
    table.add_row("Max Rounds", str(config.max_rounds))
    table.add_row("Max Agents", str(config.max_agents))
    table.add_row("Pattern", "Debate/Refinement with Consensus")
    table.add_row("Memory", "ON" if config.enable_memory else "OFF")
    table.add_row("Reputation", "ON" if config.enable_reputation else "OFF")

    console.print()
    console.print(table)

    if participants:
        ptable = Table(title="Agents", box=box.ROUNDED, border_style="green")
        ptable.add_column("Name", style="bold")
        ptable.add_column("Model")
        ptable.add_column("Role")
        for p in participants:
            ptable.add_row(p.name, p.model.split("/")[-1], p.role.value)
        console.print()
        console.print(ptable)

    console.print()


def _build_running_panel(session) -> Panel:
    content = Text()
    content.append("  Swarm is running...\n\n", style="bold yellow")
    content.append(f"  Session: {session.id[:8]}...\n", style="dim")
    content.append(f"  Status: {session.status.value}\n", style="cyan")
    content.append(f"  Agents: {len(session.participants)}\n", style="cyan")
    return Panel(content, title="[bold]Swarm Status[/bold]", border_style="yellow", box=box.ROUNDED)


def _print_result(result):
    console.print()
    console.print(Panel(
        f"[bold green]Swarm Completed![/bold green]\n\n"
        f"Best solution from: [cyan]{result.best_participant_name}[/cyan]\n"
        f"Rounds: {result.rounds_completed}\n"
        f"Consensus: {'Yes' if result.consensus_reached else 'No'} ({result.final_consensus_score:.0%})\n"
        f"Time: {result.duration_seconds:.1f}s\n"
        f"Memory: {'Saved' if result.memory_saved else 'Disabled'}",
        title="Result", border_style="green", box=box.ROUNDED,
    ))

    console.print(Panel(
        result.output, title="[bold]Generated Code[/bold]",
        border_style="cyan", box=box.ROUNDED, padding=(1, 2),
    ))


def _print_stats(stats: dict):
    console.print()
    st = Table(title="System Stats", box=box.ROUNDED, border_style="magenta")
    st.add_column("Metric", style="bold")
    st.add_column("Value")
    st.add_row("Cache Hit Rate", stats.get("cache", {}).get("hit_rate", "0%"))
    st.add_row("Cache Size", str(stats.get("cache", {}).get("response_cache_size", 0)))
    st.add_row("Memory Entries", str(stats.get("memory", {}).get("long_term", 0)))
    st.add_row("Reputation Agents", str(stats.get("reputation", {}).get("total_agents", 0)))
    console.print(st)


@app.command()
def models(
    provider: str = typer.Option("openrouter", "--provider", "-p", help="Provider: openrouter, ollama"),
):
    """List available models."""
    _print_banner()
    if provider == "openrouter":
        _list_models()
    else:
        _list_ollama()


def _list_models():
    from .models import FREE_MODELS
    table = Table(title="Free OpenRouter Models", box=box.ROUNDED, border_style="cyan")
    table.add_column("Model", style="bold cyan")
    table.add_column("Best For")
    table.add_column("Context")
    for key, info in FREE_MODELS.items():
        table.add_row(info["model"], key.title(), f"{info['ctx']//1000}K")
    console.print()
    console.print(table)
    console.print("[dim]Get API key: https://openrouter.ai/keys[/dim]")


def _list_ollama():
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if r.status_code == 200:
            models = r.json().get("models", [])
            table = Table(title="Local Ollama Models", box=box.ROUNDED, border_style="green")
            table.add_column("Model", style="bold green")
            table.add_column("Size")
            for m in models:
                size_gb = m.get("size", 0) / (1024**3)
                table.add_row(m["name"], f"{size_gb:.1f} GB")
            console.print()
            console.print(table)
    except Exception:
        console.print("[red]Ollama not running. Start with: ollama serve[/red]")


@app.command()
def init():
    """Initialize Swarm-Knight configuration."""
    _print_banner()
    console.print("[bold]Let's set up Swarm-Knight![/bold]\n")
    api_key = Prompt.ask("[cyan]OpenRouter API key[/cyan]", password=True)
    if not api_key:
        console.print("[red]API key required[/red]")
        raise typer.Exit(1)

    config_dir = Path.home() / ".swarm-knight"
    config_dir.mkdir(exist_ok=True)
    import json
    (config_dir / "config.json").write_text(json.dumps({
        "api_key": api_key,
        "default_rounds": 5,
        "enable_memory": True,
        "enable_reputation": True,
    }, indent=2))
    console.print(f"\n[green]Saved to {config_dir / 'config.json'}[/green]")
    console.print("[dim]Run: knight run \"your task\"[/dim]")


@app.command()
def stats():
    """Show system statistics."""
    _print_banner()
    from .consensus import consensus_engine
    s = consensus_engine.get_stats()
    table = Table(title="System Statistics", box=box.ROUNDED, border_style="magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Cache Hit Rate", s.get("cache", {}).get("hit_rate", "0%"))
    table.add_row("Cache Entries", str(s.get("cache", {}).get("response_cache_size", 0)))
    table.add_row("Memory Entries", str(s.get("memory", {}).get("long_term", 0)))
    table.add_row("Reputation Agents", str(s.get("reputation", {}).get("total_agents", 0)))
    table.add_row("Active Sessions", str(s.get("active_sessions", 0)))
    console.print(table)


@app.command()
def clear_cache():
    """Clear all caches."""
    from .cache import cache_system
    cache_system.clear_all()
    console.print("[green]Cache cleared[/green]")


if __name__ == "__main__":
    app()
