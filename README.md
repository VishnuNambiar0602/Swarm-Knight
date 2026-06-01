# Swarm-Knight

Multi-LLM collaboration via debate/refinement with consensus engine, memory, and reputation tracking.

## v2.0 - What's New

### Consensus Engine
- **Confidence-based termination** - stops when consensus > 85%
- **Early stopping** - detects when no improvement for 2 rounds
- **Automatic round estimation** - adjusts based on task complexity

### Memory System
- **Long-term memory** - persists across sessions
- **Project memory** - per-project context
- **Agent-specific memory** - tracks what each agent learned
- **Semantic retrieval** - finds relevant past solutions

### Dynamic Agent Generation
- **Auto-detect task type** - analyzes your query to pick the right agents
- **Automatic role assignment** - generator, critic, reviewer based on task
- **Reputation-based selection** - better agents get picked first

### Parallel Execution
- **asyncio orchestration** - all agents run simultaneously
- **Streaming aggregation** - results collected as they complete
- **Timeout protection** - prevents stuck agents

### Agent Reputation
- **Accuracy tracking** - scores based on task success
- **Latency tracking** - faster agents ranked higher
- **Consensus contribution** - agents that help reach consensus get boosted
- **Hallucination detection** - flags unreliable agents

## Quick Start

```bash
# Setup
knight init

# Run with auto-generated agents
knight run "Build a shopping cart"

# Run with specific config
knight run "Create a REST API" --preset coding --rounds 5 --agents 8

# Show stats
knight stats
```

## Commands

| Command | Description |
|---------|-------------|
| `knight run "task"` | Run a swarm task |
| `knight models` | List available models |
| `knight init` | Setup configuration |
| `knight stats` | Show system statistics |
| `knight clear-cache` | Clear all caches |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preset, -p` | auto | Preset: auto, coding, ecommerce |
| `--rounds, -r` | 5 | Max debate rounds (1-20) |
| `--agents, -a` | 6 | Max agents (2-20) |
| `--models, -m` | - | Comma-separated model names |
| `--api-key, -k` | - | OpenRouter API key |
| `--no-memory` | false | Disable memory system |
| `--no-reputation` | false | Disable reputation tracking |
| `--verbose, -V` | false | Show detailed output |

## How It Works

```
User Query
    ↓
[Dynamic Agent Generation] - Analyzes task, picks right agents
    ↓
[Parallel Generation] - All agents generate solutions simultaneously
    ↓
[Cross-Critique] - Each agent critiques others' solutions
    ↓
[Refinement] - Agents improve based on feedback
    ↓
[Consensus Check] - If score > 85%, stop. If no improvement, stop.
    ↓
[Merge] - Combine best parts from all solutions
    ↓
[Memory Store] - Save for future sessions
    ↓
[Reputation Update] - Score all agents
    ↓
Final Output
```

## Architecture

```
swarm/
├── models.py          # Data models (Session, Agent, Memory, etc.)
├── consensus.py       # Core consensus engine
├── memory.py          # Long-term, project, agent memory
├── reputation.py      # Agent reputation tracking
├── dynamic_agents.py  # Auto-generate agents from query
├── parallel.py        # Parallel execution engine
├── cache.py           # Response, embedding, tool cache
├── debate.py          # Debate round logic
├── providers.py       # LLM provider abstraction
├── orchestrator.py    # High-level interface
└── cli.py             # CLI interface
```

## Free Models

All pre-configured models are free on OpenRouter:

| Model | Best For |
|-------|----------|
| Nemotron 3 Super | Planning, QA |
| GPT-OSS 120B | Architecture, Logic |
| MiniMax M2.5 | Layout, Visual |
| Gemma 4 31B | Color, Style |
| Laguna M.1 | Coding |
| Laguna XS.2 | Fast Iteration |
| Kimi K2.6 | Full-page Assembly |
| Nemotron Nano 12B VL | Multimodal, OCR |
| GLM 4.5 Air | Tagging, Extraction |

## Configuration

Config saved to `~/.swarm-knight/config.json`:

```json
{
  "api_key": "your-key",
  "default_rounds": 5,
  "enable_memory": true,
  "enable_reputation": true
}
```
