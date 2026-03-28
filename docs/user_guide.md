# Unicorn User Guide

Complete reference for building networks, altering neuron properties, running simulations, computing layouts, and generating visualizations.

---

## Table of Contents

1. [Building Networks](#1-building-networks)
2. [Altering Neuron Properties](#2-altering-neuron-properties)
3. [Running Simulations](#3-running-simulations)
4. [Computing Layouts](#4-computing-layouts)
5. [Generating Visualizations](#5-generating-visualizations)
6. [Batch Mutation Recipes](#6-batch-mutation-recipes)
7. [CLI Reference](#7-cli-reference)
8. [Config Recipe Reference](#8-config-recipe-reference)
9. [Neuron Property Reference](#9-neuron-property-reference)

---

## 1. Building Networks

All network builder commands share this structure:

```bash
python -m tools.network_builder <path> <subcommand> [options]
```

`<path>` is the target network JSON file (default: `samples/network.json`).

---

### `generate` — one-command network creation

Create a complete N-neuron network with shared property defaults, auto-wired topology, and optional per-neuron exceptions.

```bash
python -m tools.network_builder <path> generate <count> [options] [--force]
```

**Examples:**

```bash
# 46-neuron random network, 9% density, reproducible seed, 40 steps
python -m tools.network_builder samples/net.json generate 46 \
  --topology random --density 0.09 --seed 13 \
  --tau 9 --threshold 1.0 --input-current 0.1 \
  --steps 40 --dt 0.5 --global-refractory 1.0 \
  --neuron-overrides \
    "0:input_current=1.8,threshold=0.7" \
    "7:input_current=1.4,tau=5" \
    "45:input_current=1.7,tau=6" \
  --force
```

| Flag | Default | Description |
|---|---|---|
| `count` | *(required)* | Number of neurons, assigned ids 0…N-1 |
| `--threshold` | `1.0` | Firing threshold (shared default) |
| `--tau` | `10.0` | Membrane time constant τ |
| `--input-current` | `0.0` | Input drive for every neuron |
| `--refractory-period` | `None` | Per-neuron refractory period |
| `--reset-potential` | `None` | Membrane reset value after spike |
| `--neuron-overrides` | — | Per-neuron exceptions (see below) |
| `--topology` | `none` | `none` · `chain` · `ring` · `all-to-all` · `random` |
| `--weight` | `0.5` | Synapse weight for topology presets |
| `--density` | `0.3` | Edge probability for `random` topology |
| `--seed` | `None` | RNG seed for `random` |
| `--steps` | `10` | Simulation steps |
| `--dt` | `1.0` | Time-step size |
| `--global-refractory` | `0.0` | Simulation-level refractory period |
| `--force` | — | Overwrite existing file |

**Neuron override format:** `"<id>:key=val,key=val"` (space-separate multiple entries)

```bash
--neuron-overrides \
  "0:input_current=1.8,threshold=0.7" \
  "7:tau=5,refractory_period=2.0"
```

---

### `init` — minimal fresh network

```bash
python -m tools.network_builder <path> init [--steps N] [--dt F] [--refractory-period F] [--force]
python -m tools.network_builder <path> init --from-config samples/network_config.json --force
```

---

### Incremental editing

```bash
# Add a neuron
python -m tools.network_builder net.json add-neuron 5 \
  --threshold 0.9 --membrane-time-constant 8.0 --input-current 0.5

# Add a synapse
python -m tools.network_builder net.json add-synapse 0 5 0.8

# Set global input current for one neuron
python -m tools.network_builder net.json set-input 3 1.2

# Update simulation settings
python -m tools.network_builder net.json set-config --steps 50 --dt 0.5
```

---

## 2. Altering Neuron Properties

### `set-neuron` — patch any property on one neuron

```bash
python -m tools.network_builder <path> set-neuron <id> [options]
```

| Flag | Property altered |
|---|---|
| `--threshold F` | Firing threshold |
| `--tau F` | Membrane time constant |
| `--refractory-period F` | Refractory period |
| `--reset-potential F` | Membrane reset value |
| `--v-rest F` | Resting potential (membrane leaks toward this) |
| `--input-current F` | Constant external drive |
| `--bias F` | Additive bias current (independent of synapses) |
| `--gain F` | Scales all incoming synaptic current |
| `--noise-std F` | Gaussian noise std injected each step |
| `--dropout F` | Per-step probability of silencing this neuron |
| `--adaptation-rate F` | Threshold rise per spike |
| `--adaptation-decay F` | Per-step decay of adaptation state |
| `--activation lif\|relu\|softplus\|tanh\|sigmoid\|rbf` | Neuron model type |
| `--rbf-centre F` | RBF centre voltage |
| `--rbf-sigma F` | RBF Gaussian width |

**Examples:**

```bash
# Make neuron 5 a sigmoid probabilistic unit with noise and dropout
python -m tools.network_builder net.json set-neuron 5 \
  --activation sigmoid --noise-std 0.05 --dropout 0.25

# Configure neuron 12 as a RBF unit
python -m tools.network_builder net.json set-neuron 12 \
  --activation rbf --rbf-centre 0.5 --rbf-sigma 0.2 --gain 1.5

# Add spike-frequency adaptation to neuron 3
python -m tools.network_builder net.json set-neuron 3 \
  --adaptation-rate 0.4 --adaptation-decay 0.85 --refractory-period 2.0

# Neuron 0: elevated drive with v_rest baseline
python -m tools.network_builder net.json set-neuron 0 \
  --input-current 1.8 --v-rest 0.2 --bias 0.05
```

---

### `mutate` — bulk property changes across a neuron slice

```bash
python -m tools.network_builder <path> mutate \
  --select <selector> \
  [--set   "key=val,..."] \
  [--add   "key=delta,..."] \
  [--scale "key=factor,..."]
```

**Selectors:**

| Selector | Meaning |
|---|---|
| `all` | Every neuron |
| `range:0-9` | Neurons 0 through 9 (inclusive) |
| `every:3` | Every 3rd neuron (0, 3, 6, …) |
| `0,5,12` | Specific ids |

**Examples:**

```bash
# Double gain on all neurons
python -m tools.network_builder net.json mutate --select all --scale gain=2.0

# Drop threshold by 0.1 on neurons 0-9
python -m tools.network_builder net.json mutate \
  --select range:0-9 --add threshold=-0.1

# Set noise+dropout on every 3rd neuron
python -m tools.network_builder net.json mutate \
  --select every:3 --set noise_std=0.05,dropout_prob=0.1

# Push more drive to a specific subgroup
python -m tools.network_builder net.json mutate \
  --select 0,7,15,22 --add input_current=0.5
```

---

### `apply-profile` — named property presets

```bash
python -m tools.network_builder <path> apply-profile <profile> --to <selector>
python -m tools.network_builder <path> apply-profile --from-file profiles.json --name my_type --to all
```

**Built-in profiles:**

| Profile | Properties applied |
|---|---|
| `inhibitory` | `gain=-1`, `threshold=0.8`, `tau=5` |
| `driver` | `input_current=1.5`, `threshold=0.7`, `noise_std=0.02` |
| `adaptive` | `adaptation_rate=0.3`, `adaptation_decay=0.9`, `refractory_period=2.0` |
| `stochastic` | `dropout_prob=0.25`, `noise_std=0.05`, `activation_fn=sigmoid` |
| `silent` | `input_current=0`, `dropout_prob=1.0` |

**Custom profiles** (`profiles.json`):
```json
{
  "excitatory_relay": {
    "gain": 1.5, "threshold": 0.85, "noise_std": 0.01
  }
}
```

```bash
python -m tools.network_builder net.json apply-profile --from-file profiles.json \
  --name excitatory_relay --to range:10-20
```

---

### `set-schedule` — time-varying input current

Attach a dynamic drive schedule to a neuron. Overrides its `input_current` each step.

```bash
python -m tools.network_builder <path> set-schedule <id> \
  --mode <constant|ramp|pulse|sine> \
  [--amplitude F] [--period F] [--offset F] [--duration F]
```

| Mode | Behaviour |
|---|---|
| `constant` | Fixed value `amplitude + offset` |
| `sine` | `amplitude * sin(2π * t / period) + offset` |
| `pulse` | Square wave: high for first half of each period |
| `ramp` | Linear ramp from 0 to `amplitude` over `duration` steps |

**Examples:**

```bash
# Sine wave: neuron 0 with amplitude 1.2, period 12 steps
python -m tools.network_builder net.json set-schedule 0 \
  --mode sine --amplitude 1.2 --period 12

# Ramp: neuron 3 ramps up over 20 steps then holds
python -m tools.network_builder net.json set-schedule 3 \
  --mode ramp --amplitude 2.0 --duration 20

# Pulse: neuron 7 alternates every 5 steps
python -m tools.network_builder net.json set-schedule 7 \
  --mode pulse --amplitude 1.5 --period 10
```

---

### `props` — inspect neuron properties

```bash
python -m tools.network_builder <path> props [id ...]
```

```bash
# Show all neurons
python -m tools.network_builder net.json props

# Show specific neurons
python -m tools.network_builder net.json props 0 5 12
```

Output example:
```
Neuron   0  input_current=+1.8000
  threshold=0.7000  membrane_time_constant=9.0000
  bias=0.0500  v_rest=0.2000  activation_fn=lif
  input_schedule={'mode': 'sine', 'amplitude': 1.2, 'period': 12.0, ...}

Neuron   5  input_current=+0.0000
  threshold=1.0000  activation_fn=sigmoid
  noise_std=0.0500  dropout_prob=0.2500
```

---

## 3. Running Simulations

```bash
python -m tools.animate_preview <network.json> \
  --layout <layout.json> \
  --output output/animation.html \
  --history-output samples/spike_history.json
```

The simulator (`SimpleSNN`) reads all neuron properties automatically — no extra flags needed.

---

## 4. Computing Layouts

```bash
python -m tools.layout_demo <network.json> [--output <layout.json>]
```

Uses a chunked, vectorised force-directed algorithm. Layout positions are stored in 3-D space.

---

## 5. Generating Visualizations

### Static 3D preview (Plotly)

```bash
python -m tools.render_preview <network.json> \
  --layout <layout.json> --output output/preview.html
```

### Animated spike playback (Plotly)

```bash
python -m tools.animate_preview <network.json> \
  --layout <layout.json> --output output/animation.html
```

### WebGPU real-time renderer

```bash
python -m tools.webgpu_preview <network.json> \
  --layout <layout.json> \
  --history <spike_history.json> \
  --output output/preview_webgpu.html
```

**Requires:** Chrome 113+, Edge 113+, or any WebGPU-capable browser.

#### WebGPU Visualizer Features

| Feature | Description |
|---|---|
| **Per-type colouring** | Each activation function type renders in a distinct colour (see legend) |
| **Spike glow** | Neurons flash warm amber on firing, then decay |
| **Noise shimmer** | Neurons with `noise_std > 0` show a subtle brightness oscillation |
| **Adaptation dimming** | High `adaptation_rate` dims the neuron's base colour |
| **Dropout desaturation** | Neurons with `dropout_prob > 0` become grey-tinted |
| **Gain-scaled size** | Larger crosshairs = higher `gain` |
| **Mouse-drag orbit** | Left-drag to rotate the 3D network |
| **Scroll zoom** | Mouse wheel to zoom in/out |
| **Type legend** | HUD shows neuron types present and their modifiers |
| **Speed control** | ½× · 1× · 2× · 4× simulation playback |

**Activation type colour legend:**

| Type | Colour |
|---|---|
| `lif` | 🔵 Cool blue |
| `relu` | 🟢 Green |
| `softplus` | 🩵 Cyan |
| `tanh` | 🟣 Purple |
| `sigmoid` | 🟠 Orange |
| `rbf` | 🟡 Yellow |

---

## 6. Batch Mutation Recipes

Apply sequences of mutations from a YAML or JSON recipe file:

```bash
python -m tools.mutate_network recipe.yaml
```

**Recipe format (YAML):**

```yaml
network: samples/network_46.json
output:  samples/network_46_evolved.json
steps:
  # Make neurons 10-20 probabilistic stochastic units
  - apply-profile:
      name: stochastic
      to: "range:10-20"

  # Globally amplify all synaptic input
  - mutate:
      select: all
      scale:
        gain: 1.5

  # Add sinewave drive to the main input neuron
  - set-schedule:
      id: 0
      mode: sine
      amplitude: 1.5
      period: 12

  # Fine-tune one specific neuron
  - set-neuron:
      id: 22
      activation_fn: rbf
      rbf_centre: 0.6
      rbf_sigma: 0.15
      adaptation_rate: 0.2
```

---

## 7. CLI Reference

### `network_builder` subcommands

| Subcommand | Purpose |
|---|---|
| `init` | Create a fresh network file |
| `generate` | Build an N-neuron network in one step |
| `add-neuron` | Append a neuron |
| `add-synapse` | Append a synapse |
| `set-input` | Set input current for one neuron |
| `set-config` | Update global simulation settings |
| `set-neuron` | Patch any property on an existing neuron |
| `mutate` | Bulk property change across a neuron selector |
| `apply-profile` | Apply a built-in or custom property preset |
| `set-schedule` | Attach a time-varying input schedule |
| `props` | Print all properties of selected neurons |
| `summary` | Print a concise network summary |
| `validate` | Validate neuron/synapse references and numeric ranges |

### Other tools

| Tool | Command |
|---|---|
| Animated preview | `python -m tools.animate_preview` |
| Static render | `python -m tools.render_preview` |
| WebGPU preview | `python -m tools.webgpu_preview` |
| Layout solver | `python -m tools.layout_demo` |
| Batch mutate | `python -m tools.mutate_network <recipe.yaml>` |

---

## 8. Config Recipe Reference

`init --from-config` bootstraps a network from a JSON recipe:

```json
{
  "count": 10,
  "topology": "random",
  "density": 0.25,
  "seed": 42,
  "weight": 0.5,
  "threshold": 1.0,
  "membrane_time_constant": 10.0,
  "refractory_period": 1.0,
  "input_current": 0.3,
  "steps": 30,
  "dt": 0.5
}
```

---

## 9. Neuron Property Reference

Every neuron object in the JSON supports these fields. All are optional (sensible defaults apply).

### Core LIF properties

| Field | Default | Description |
|---|---|---|
| `threshold` | `1.0` | Membrane voltage threshold for firing |
| `membrane_time_constant` | `10.0` | τ in ms — how fast voltage decays |
| `refractory_period` | `0.0` | Silence enforced after each spike |
| `reset_potential` | `0.0` | Voltage set to this after firing |
| `initial_voltage` | `0.0` | Starting membrane voltage |
| `v_rest` | `0.0` | Resting potential; membrane leaks toward this |

### Extended properties

| Field | Default | Description |
|---|---|---|
| `activation_fn` | `"lif"` | Neuron model — `lif`, `relu`, `softplus`, `tanh`, `sigmoid`, `rbf` |
| `bias` | `0.0` | Constant additive current, independent of synapses |
| `gain` | `1.0` | Scales all incoming synaptic current |
| `noise_std` | `0.0` | Gaussian noise std injected per step; 0 = off |
| `dropout_prob` | `0.0` | Per-step silencing probability; 0 = never, 1 = always |
| `adaptation_rate` | `0.0` | Threshold rise per spike (spike-frequency adaptation) |
| `adaptation_decay` | `1.0` | Per-step decay of adaptation back to baseline |
| `rbf_centre` | `0.5` | Voltage at which RBF unit is maximally active |
| `rbf_sigma` | `0.3` | Width of the RBF Gaussian |
| `input_schedule` | `null` | Dict with `mode`, `amplitude`, `period`, `offset`, `duration` |

### Activation function cheat-sheet

| Type | Fire condition / output | Best used for |
|---|---|---|
| `lif` | v ≥ θ → reset | Standard spiking — default biological model |
| `relu` | max(0, v) | Rate-code neurons, non-negative output |
| `softplus` | log(1 + exp(v)) | Smooth relu; always positive |
| `tanh` | tanh(v) ∈ [−1, 1] | Biphasic / inhibitory output |
| `sigmoid` | 1/(1+exp(−v)) ∈ [0,1] | Probabilistic firing (stochastic units) |
| `rbf` | exp(−½(v−centre)²/σ²) | Pattern detector — fires for specific voltage range |

### Plasticity properties (per synapse)

| Field | Default | Description |
|---|---|---|
| `plasticity_rule` | `null` | `"hebbian"` or `"oja"` for online weight updates |
| `plasticity_lr` | `0.01` | Learning rate for the plasticity rule |

---

> **Performance note:** All extended properties use NumPy vectorised operations — there is no per-step Python loop cost, even for 63 000 neurons. New properties add at most a single O(N) array pass each step.
