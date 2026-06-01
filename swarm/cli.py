"""Swarm-Knight CLI - Production-ready multi-LLM collaboration."""

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

app = typer.Typer(name="knight", help="Swarm-Knight", add_completion=False)
console = Console()

__version__ = "2.1.0"


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]Swarm-Knight[/bold cyan] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(version: Optional[bool] = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True)):
    pass


@app.command()
def run(
    task: str = typer.Argument(..., help="The task to execute"),
    rounds: int = typer.Option(5, "--rounds", "-r", help="Max rounds"),
    agents: int = typer.Option(6, "--agents", "-a", help="Max agents"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="OpenRouter API key"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Verbose output"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable memory"),
    no_reputation: bool = typer.Option(False, "--no-reputation", help="Disable reputation"),
):
    """Run a swarm task through the Knight loop."""
    asyncio.execute(_run(task, rounds, agents, api_key, verbose, no_memory, no_reputation))


async def _run(task, rounds, agents, api_key, verbose, no_memory, no_reputation):
    from .knight import knight
    from .models import SwarmConfig
    from .retry import retry_queue

    _banner()

    if not api_key:
        import os
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            api_key = Prompt.ask("\n[bold cyan]OpenRouter API key[/bold cyan]", password=True)
            if not api_key:
                console.print("[red]Required. Get one at https://openrouter.ai/keys[/red]")
                raise typer.Exit(1)

    config = SwarmConfig(
        max_rounds=rounds,
        max_agents=agents,
        enable_memory=not no_memory,
        enable_reputation=not no_reputation,
    )

    _info(task, config)

    def on_event(event: str, data: dict):
        icons = {
            "plan_start": "[bold blue]Planning...[/bold blue]",
            "plan_complete": f"  [dim]Type: {data.get('plan', {}).get('task_type', '?')}[/dim]",
            "spawn_start": f"[bold magenta]Spawning {len(data.get('roles', []))} agents...[/bold magenta]",
            "spawn_complete": f"  [dim]{data.get('agents', [])}[/dim]",
            "generate_start": f"[bold cyan]Generating {data.get('agent_count', 0)} solutions...[/bold cyan]",
            "generate_complete": f"  [dim]{data.get('solutions', 0)} solutions ready[/dim]",
            "debate_start": f"[bold yellow]Debate round {data.get('round', '?')}...[/bold yellow]",
            "debate_complete": f"  [dim]{data.get('critiques', 0)} critiques[/dim]",
            "judge_start": "[bold red]Judging...[/bold red]",
            "judge_complete": f"  [dim]Winner: {data.get('winner', '?')}[/dim]",
            "memory_start": "[dim]Updating memory...[/dim]",
            "reputation_start": "[dim]Updating reputation...[/dim]",
            "optimize_start": "[dim]Self-optimizing...[/dim]",
        }
        if event in icons:
            console.print(icons[event])

    knight.on("event", on_event)
    retry_queue.set_callback(lambda aid, res: None)
    await retry_queue.start()

    with Live(console=console, refresh_per_second=4):
        result = await knight.run(task=task, config=config, api_key=api_key)

    await retry_queue.stop()

    _result(result)
    if verbose:
        _stats(result.metrics)


def _banner():
    b = """
 ███╗   ███╗██╗███╗   ██╗███████╗ ██████╗ █████╗ ███╗   ██╗
 ████╗ ████║██║████╗  ██║██╔════╝██╔════╝██╔══██╗████╗  ██║
 ██╔████╔██║██║██╔██╗ ██║███████╗██║     ███████║██╔██╗ ██║
 ██║╚██╔╝██║██║██║╚██╗██║╚════██║██║     ██╔══██║██║╚██╗██║
 ██║ ╚═╝ ██║██║██║ ╚████║███████║╚██████╗██║  ██║██║ ╚████║
 ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝"""
    console.print()
    console.print(Align.center(Text(b, style="bold cyan")))
    console.print(Align.center(Text("v2.1 - Production Loop", style="dim")))
    console.print()


def _info(task: str, config):
    t = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    t.add_column("Key", style="bold")
    t.add_column("Value")
    t.add_row("Task", task[:60] + "..." if len(task) > 60 else task)
    t.add_row("Max Rounds", str(config.max_rounds))
    t.add_row("Max Agents", str(config.max_agents))
    t.add_row("Loop", "Plan -> Agents -> Generate -> Debate -> Judge -> Memory -> Rep -> Optimize")
    console.print()
    console.print(t)
    console.print()


def _result(result):
    console.print()
    console.print(Panel(
        f"[bold green]Completed![/bold green]\n\n"
        f"Winner: [cyan]{result.winner_name}[/cyan]\n"
        f"Rounds: {result.rounds_completed}\n"
        f"Consensus: {result.consensus_score:.0%}\n"
        f"Time: {result.duration_seconds:.1f}s",
        title="Result", border_style="green", box=box.ROUNDED,
    ))
    console.print(Panel(
        result.output, title="[bold]Output[/bold]",
        border_style="cyan", box=box.ROUNDED, padding=(1, 2),
    ))


def _stats(metrics: dict):
    t = Table(title="Metrics", box=box.ROUNDED, border_style="magenta")
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    for k, v in metrics.items():
        if not isinstance(v, (dict, list)):
            t.add_row(k, str(v))
    console.print(t)


@app.command()
def models():
    """List free models."""
    from .models import FREE_MODELS
    t = Table(title="Free Models", box=box.ROUNDED, border_style="cyan")
    t.add_column("Model", style="bold cyan")
    t.add_column("Roles")
    t.add_column("Context")
    for k, v in FREE_MODELS.items():
        t.add_row(v["model"], ", ".join(v["roles"]), f"{v['ctx']//1000}K")
    console.print()
    console.print(t)


@app.command()
def init():
    """Setup API key."""
    _banner()
    api_key = Prompt.ask("[cyan]OpenRouter API key[/cyan]", password=True)
    if not api_key:
        console.print("[red]Required[/red]")
        raise typer.Exit(1)
    config_dir = Path.home() / ".swarm-knight"
    config_dir.mkdir(exist_ok=True)
    import json
    (config_dir / "config.json").write_text(json.dumps({"api_key": api_key}, indent=2))
    console.print(f"[green]Saved to {config_dir / 'config.json'}[/green]")


@app.command()
def stats():
    """Show system stats."""
    from .metrics import metrics as m
    from .reputation import reputation as r
    from .memory import memory as mem
    from .cache import cache as c
    t = Table(title="Stats", box=box.ROUNDED, border_style="magenta")
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    snap = m.get_snapshot()
    for k, v in snap.items():
        if not isinstance(v, (dict, list)):
            t.add_row(k, str(v))
    t.add_row("memory", str(mem.get_stats()))
    t.add_row("reputation", str(r.get_stats()))
    t.add_row("cache", str(c.get_stats()))
    console.print(t)


@app.command()
def clear():
    """Clear all caches."""
    from .cache import cache as c
    c.clear()
    console.print("[green]Cleared[/green]")


if __name__ == "__main__":
    app()
