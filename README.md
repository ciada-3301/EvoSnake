# 🐍 EvoSnake

> A multi-agent artificial life simulation that models **natural selection and evolution** using neuroevolution — combining neural network policies with genetic algorithms in a real-time 2D environment.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pygame-2.x-green?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow"/>
  <img src="https://img.shields.io/github/stars/ciada-3301/EvoSnake?style=social"/>
</p>

---

## 📌 What is EvoSnake?

EvoSnake places a population of autonomous agents ("snakes") in a constrained 2D world with limited food. Each agent is controlled by a small neural network that perceives its environment and decides how to move. Agents that eat more food survive longer and reproduce — passing on (and mutating) their neural weights to the next generation.

**No rewards are hand-coded. No gradients are computed. Intelligence emerges purely from selection pressure.**

Over successive generations the population discovers increasingly efficient food-seeking strategies, demonstrating emergent behaviour from simple evolutionary rules.

---

## ✨ Features

- 🧠 **PyTorch neural policy** — each agent runs a real `nn.Module` (8 → 16 → 16 → 2)
- 🧬 **Genetic algorithm** — elite selection, weight mutation, uniform crossover
- 🎮 **Real-time pygame visualisation** — trails, energy rings, direction vectors
- 📊 **Live fitness chart** — best and average fitness plotted across generations
- ⌨️ **Interactive controls** — pause, reset, speed, mutation rate, architecture overlay
- 🔁 **Two versions** — plain NumPy (`evosnake.py`) and PyTorch (`evosnake_pytorch.py`)

---

## 🗂️ Repository Structure

```
EvoSnake/
├── evosnake.py            # Version 1 — pure NumPy neural net
├── evosnake_pytorch.py    # Version 2 — PyTorch nn.Module policy
├── requirements.txt       # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ciada-3301/EvoSnake.git
cd EvoSnake
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pygame numpy torch
```

### 3. Run the simulation

```bash
# NumPy version
python evosnake.py

# PyTorch version (recommended)
python evosnake_pytorch.py
```

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `R` | Reset simulation (new random population) |
| `N` | Force-advance to next generation |
| `↑` / `↓` | Increase / decrease simulation speed |
| `M` | Cycle mutation rate: Low → Med → High |
| `A` | Toggle architecture info overlay *(PyTorch version)* |
| `D` | Toggle debug labels (fitness score per agent) |
| `ESC` / `Q` | Quit |

---

## 🖥️ Visual Guide

| Element | Meaning |
|---------|---------|
| 🟢 Green agent | Elite — top 20% by fitness this generation |
| 🔵 Blue agent | Normal agent |
| ⚫ Faded agent | Dead (energy depleted) |
| 🟡 Yellow dot | Food item |
| 🔴 Red line | Direction vector (current heading) |
| Arc ring | Energy remaining (full ring = full energy) |
| Fading trail | Recent path history |

---

## ⚙️ How It Works

### 1. Environment

A 620 × 620 pixel 2D world with a fixed pool of food items. Food respawns immediately when eaten so competition stays constant throughout a generation.

### 2. Agent sensing (8 inputs)

Each agent observes:

| Input | Description |
|-------|-------------|
| `food_dx` | Normalised X direction to nearest food |
| `food_dy` | Normalised Y direction to nearest food |
| `food_dist` | Distance to nearest food (normalised 0–1) |
| `vx` | Current X velocity (normalised) |
| `vy` | Current Y velocity (normalised) |
| `energy` | Remaining energy (0–1) |
| `wall_dx` | Normalised distance to nearest vertical wall |
| `wall_dy` | Normalised distance to nearest horizontal wall |

### 3. Neural policy (PyTorch version)

```
Input (8)  →  Linear(8→16) + Tanh
           →  Linear(16→16) + Tanh
           →  Linear(16→2) + Tanh
           →  Output (dx, dy) steering vector
```

The output is blended 50/50 with a direct food-seeking vector. This gives even first-generation agents a fighting chance while still leaving room for the network to learn richer strategies.

### 4. Fitness function

```
+1   for each food item eaten
−ε   per timestep (energy drain, encourages efficiency)
```

Agents with zero energy are eliminated mid-generation. This creates pressure not just to eat, but to eat *fast*.

### 5. Evolution (each generation end)

```
1. Rank all agents by fitness (food eaten)
2. Select top 20% as elite
3. Clone top 2 unchanged  →  preserves best solution
4. Crossover pairs from elite  →  ~25% of new population
5. Mutate elite parents  →  fills remaining slots
6. Spawn new food, reset step counter
```

**Mutation** adds sparse Gaussian noise (~30% of weights perturbed per call), controlled by the mutation rate slider.

**Crossover** (PyTorch version) performs uniform crossover — each weight is independently drawn from either parent with 50% probability.

---

## 📈 Emergent Behaviours to Watch For

By generation 5–10 with default settings you will typically observe:

- **Beelining** — agents stop wandering and move directly toward food
- **Wall avoidance** — agents learn not to get trapped at boundaries
- **Energy management** — longer-lived agents consume food more efficiently
- **Competitive clustering** — multiple elite agents converging on the same food patch

Try cranking mutation to `High` to watch exploration chaos, then dropping it to `Low` to watch the population lock in and exploit a good strategy.

---

## 🔧 Configuration

Key constants at the top of each file:

| Constant | Default | Description |
|----------|---------|-------------|
| `POP_SIZE` | 32 | Number of agents per generation |
| `FOOD_COUNT` | 22 | Food items in the world |
| `STEPS_PER_GEN` | 650 | Timesteps before evolution triggers |
| `AGENT_SPEED` | 2.8 | Max movement speed (pixels/tick) |
| `ENERGY_DRAIN` | 0.00065 | Energy cost per timestep |
| `ENERGY_FOOD` | 0.42 | Energy gained per food eaten |
| `ELITE_FRAC` | 0.20 | Fraction of population selected as elite |
| `HIDDEN_DIM` | 16 | Hidden layer size (PyTorch version) |

---

## 🔮 Planned Enhancements

- [ ] Save / load elite neural weights between sessions
- [ ] Predator agents (creates prey-evasion pressure)
- [ ] Multi-species ecosystems
- [ ] RNN / LSTM policy for memory-based agents
- [ ] Agent-to-agent communication signals
- [ ] Fitness landscape visualisation
- [ ] Distributed simulation for large populations

---

## 🧪 Applications & Research Relevance

EvoSnake is a hands-on demonstration of several active research areas:

- **Neuroevolution** — evolving neural network weights without gradient descent (related to OpenAI ES, NEAT)
- **Artificial life** — studying how life-like behaviour arises from simple rules
- **Emergent intelligence** — intelligence without explicit programming
- **Multi-agent systems** — competitive resource dynamics

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

Inspired by the fields of evolutionary computation, artificial life research, and the classic work on neuroevolution of augmenting topologies (NEAT).

---

<p align="center">Built with ❤️ · <a href="https://github.com/ciada-3301/EvoSnake">github.com/ciada-3301/EvoSnake</a></p>
