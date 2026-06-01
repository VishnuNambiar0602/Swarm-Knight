# Swarm Intelligence Module

Multi-LLM collaboration through debate/refinement for code development.

## Overview

This module enables multiple open-source LLMs to work together on coding tasks through an iterative debate and refinement process. Instead of relying on a single model, the swarm leverages the strengths of different models specialized for different aspects of software development.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Swarm Orchestrator                        │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ OpenRouter│  │  Ollama   │  │ Together  │  │   Groq    │   │
│  │ (Free)    │  │  (Local)  │  │    AI     │  │           │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │              │              │              │           │
│        └──────────────┴──────────────┴──────────────┘           │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │  Debate Manager   │                        │
│                    │  - Round tracking │                        │
│                    │  - Critique cycle │                        │
│                    │  - Consensus      │                        │
│                    └─────────┬─────────┘                        │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │  Output Merger    │                        │
│                    │  - Best selection │                        │
│                    │  - Code merging   │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## Debate/Refinement Pattern

1. **Initial Generation**: All participants independently generate solutions
2. **Cross-Critique**: Each participant critiques all other solutions
3. **Refinement**: Participants refine their solutions based on received feedback
4. **Repeat**: Steps 2-3 repeat until consensus or max rounds reached
5. **Selection**: Best solution is selected based on quality metrics

## Pre-configured Free Models (OpenRouter)

| Role | Model | Strengths |
|------|-------|-----------|
| **Planner** | Nemotron 3 Super | Multi-agent planning, long-term reasoning |
| **Architect** | GPT-OSS 120B | High-reasoning, structured outputs |
| **Designer** | MiniMax M2.5 | Layout, UI/UX, visual hierarchy |
| **Stylist** | Gemma 4 31B | Reasoning, multimodal understanding |
| **Coder** | Laguna M.1 | Flagship coding agent model |
| **Fast Iter** | Laguna XS.2 | Rapid iteration, smaller patches |
| **Assembler** | Kimi K2.6 | Long-horizon coding, full-page assembly |
| **Reviewer** | Nemotron 3 Super | QA, edge-case testing |
| **Extractor** | Nemotron Nano 12B VL | Document intelligence, OCR |
| **Tagger** | GLM 4.5 Air | Structured extraction, tagging |

## API Endpoints

### Create Swarm Session
```http
POST /api/swarm/create
Content-Type: application/json

{
  "task": "Build a responsive product card component",
  "preset": "coding",
  "config": {
    "max_rounds": 3,
    "consensus_threshold": 0.7
  }
}
```

### Start Swarm
```http
POST /api/swarm/{session_id}/start
```

### Get Status
```http
GET /api/swarm/{session_id}/status
```

### Get Result
```http
GET /api/swarm/{session_id}/result
```

### WebSocket Updates
```javascript
const ws = new WebSocket('ws://localhost:8324/api/swarm/ws/{session_id}');

ws.onmessage = (event) => {
  const { event: eventType, data } = JSON.parse(event.data);
  
  switch (eventType) {
    case 'round_start':
      console.log(`Round ${data.round} started`);
      break;
    case 'consensus':
      console.log(`Consensus reached: ${data.score}`);
      break;
    case 'completed':
      console.log(`Best solution from: ${data.best_participant}`);
      break;
  }
};
```

## Configuration

### Swarm Config
```python
SwarmConfig(
    max_rounds=3,              # Max debate rounds (1-10)
    consensus_threshold=0.7,   # Score to trigger early consensus
    min_improvement=0.05,      # Min improvement to continue
    enable_parallel_generation=True,
    enable_cross_critique=True,
    timeout_seconds=300,
)
```

### Custom Participants
```python
SwarmParticipant(
    name="My Coder",
    provider=ProviderType.OPENROUTER,
    model="poolside/poolside-laguna-m-1:free",
    role=ParticipantRole.GENERATOR,
    api_key="your-api-key",
)
```

## Usage Example

```python
from backend.apps.agents.swarm import SwarmOrchestrator, SwarmConfig

# Create orchestrator
orchestrator = SwarmOrchestrator()

# Create session with preset
session = await orchestrator.create_session(
    task="Build a shopping cart with Redux",
    preset="coding",
    config=SwarmConfig(max_rounds=3),
)

# Run swarm
result = await orchestrator.run_session(session.id)

print(f"Best solution from {result.best_participant_name}:")
print(result.output)
```

## Local LLM Support (Ollama)

```python
from backend.apps.agents.swarm.models import SwarmParticipant, ProviderType

participant = SwarmParticipant(
    name="Local Llama",
    provider=ProviderType.OLLAMA,
    model="llama3.2",
    base_url="http://localhost:11434",
)
```

## Benefits

1. **No Single Point of Failure**: If one model fails, others continue
2. **Specialized Strengths**: Each model contributes its best capabilities
3. **Quality through Debate**: Multiple perspectives improve output quality
4. **Free Tier**: All pre-configured models are free on OpenRouter
5. **Local Fallback**: Ollama support for fully offline usage
