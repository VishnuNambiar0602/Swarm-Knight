# Swarm-Knight

Multi-LLM collaboration via debate/refinement. Run coding tasks with multiple AI models working together.

## Quick Start

```bash
# First time setup
python swarm.py init

# Run a task
python swarm.py run "Build a shopping cart component"

# With options
python swarm.py run "Create a REST API" --preset coding --rounds 5
```

## Commands

### `knight run`

Run a swarm task with multiple AI models.

```bash
knight run "Build a responsive navbar"
knight run "Create a login page" --preset ecommerce
knight run "Write unit tests" --rounds 5 --verbose
```

**Options:**
- `--preset, -p` - Preset config: `coding` (default), `ecommerce`
- `--rounds, -r` - Max debate rounds (1-10, default: 3)
- `--models, -m` - Comma-separated model names to use
- `--api-key, -k` - OpenRouter API key (or set `OPENROUTER_API_KEY`)
- `--verbose, -V` - Show detailed output

### `knight models`

List available models.

```bash
knight models                    # List free OpenRouter models
knight models --provider ollama  # List local Ollama models
```

### `knight presets`

Show available presets.

### `knight init`

Initialize configuration (saves API key to `~/.swarm-knight/config.json`).

## Presets

### Coding (Default)
Best for general coding tasks:
- **Laguna M.1** - Core coding
- **Kimi K2.6** - Assembly & orchestration  
- **Nemotron 3 Super** - Review & QA

### E-Commerce
Full e-commerce development with 6 specialized models:
- **Nemotron 3 Super** - Planning
- **GPT-OSS 120B** - Architecture
- **MiniMax M2.5** - Design
- **Gemma 4 31B** - Styling
- **Laguna M.1** - Components
- **Kimi K2.6** - Assembly

## Environment Variables

```bash
export OPENROUTER_API_KEY="your-key-here"
```

Get a free key at: https://openrouter.ai/keys

## How It Works

1. **Initial Generation** - All models generate solutions independently
2. **Cross-Critique** - Each model critiques the others' solutions
3. **Refinement** - Models improve based on feedback
4. **Repeat** - Until consensus or max rounds reached
5. **Selection** - Best solution is selected

## Example Session

```
$ knight run "Build a product card component"

 ███╗   ███╗██╗███╗   ██╗███████╗ ██████╗ █████╗ ███╗   ██╗
 ████╗ ████║██║████╗  ██║██╔════╝██╔════╝██╔══██╗████╗  ██║
 ██╔████╔██║██║██╔██╗ ██║███████╗██║     ███████║██╔██╗ ██║
 ██║╚██╔╝██║██║██║╚██╗██║╚════██║██║     ██╔══██║██║╚██╗██║
 ██║ ╚═╝ ██║██║██║ ╚████║███████║╚██████╗██║  ██║██║ ╚████║
 ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝

            Multi-LLM Collaboration

  Task Configuration
  ┌─────────────┬─────────────────────────────────┐
  │ Task        │ Build a product card component  │
  │ Models      │ 3                               │
  │ Max Rounds  │ 3                               │
  │ Pattern     │ Debate/Refinement               │
  └─────────────┴─────────────────────────────────┘

  Participants
  ┌─────────────────┬──────────────────────┬───────────┐
  │ Name            │ Model                │ Role      │
  ├─────────────────┼──────────────────────┼───────────┤
  │ Laguna Coder    │ poolside-laguna-m-1  │ generator │
  │ Kimi Assembler  │ kimi-k2.6            │ generator │
  │ Nemotron Review │ nemotron-3-super     │ critic    │
  └─────────────────┴──────────────────────┴───────────┘

  Round 1 started...
  Consensus: 45%

  Round 2 started...
  Consensus: 72%

  Round 3 started...
  Consensus: 89%

  Completed in 45.2s

  Result
  ┌─────────────────────────────────────────────────────┐
  │ Swarm Completed!                                    │
  │                                                     │
  │ Best solution from: Laguna Coder                    │
  │ Rounds: 3                                           │
  │ Consensus: Yes                                      │
  │ Time: 45.2s                                         │
  └─────────────────────────────────────────────────────┘

  Generated Code
  ┌─────────────────────────────────────────────────────┐
  │ import React from 'react';                          │
  │                                                     │
  │ interface ProductCardProps {                         │
  │   name: string;                                     │
  │   price: number;                                    │
  │   image: string;                                    │
  │ }                                                   │
  │                                                     │
  │ export const ProductCard = ({ name, price, image }) │
  │   => {                                              │
  │   return (                                          │
  │     <div className="product-card">                  │
  │       <img src={image} alt={name} />                │
  │       <h3>{name}</h3>                               │
  │       <p>${price}</p>                               │
  │     </div>                                          │
  │   );                                                │
  │ };                                                  │
  └─────────────────────────────────────────────────────┘
```
