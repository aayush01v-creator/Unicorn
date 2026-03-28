# Unicorn User Guide

This guide covers everything you need to build networks, run simulations, and generate visualizations with Unicorn.

---

## Table of Contents

1. [Building Networks](#1-building-networks)
   - [generate — one-command network creation](#generate--one-command-network-creation)
   - [init --from-config — recipe-driven setup](#init---from-config--recipe-driven-setup)
   - [Incremental editing](#incremental-editing-add-neuron-add-synapse)
2. [Running Simulations](#2-running-simulations)
3. [Computing Layouts](#3-computing-layouts)
4. [Generating Visualizations](#4-generating-visualizations)
5. [Viewing Output](#5-viewing-output)
6. [CLI Reference](#6-cli-reference)
7. [Config Recipe Reference](#7-config-recipe-reference)

---

## 1. Building Networks

All network builder commands share this structure:

```bash
python -m tools.network_builder <path> <subcommand> [options]
```

`<path>` is the target network JSON file. It defaults to `samples/network.json`.

---

### `generate` — one-command network creation

Create a complete N-neuron network with shared property defaults, auto-wired topology, and optional per-neuron exceptions — all in a single call.

```bash
python -m tools.network_builder <path> generate <count> [options] [--force]
```

**Examples:**

```bash
# 10-neuron ring, everything at defaults
python -m tools.network_builder samples/my.json generate 10 --topology ring --force

# 6-neuron random network, 40% density, reproducible seed
python -m tools.network_builder samples/my.json generate 6 \
  --topology random --density 0.4 --seed 42 --weight 0.6 --force

# 5-neuron chain with mixed neuron properties
python -m tools.network_builder samples/my.json generate 5 \
  --topology chain \
  --tau 8 --threshold 1.0 --input-current 0.1 \
  --steps 30 --dt 0.5 --global-refractory 1.0 \
  --neuron-overrides "0:input_current=1.5,threshold=0.8" "3:tau=4,refractory_period=2.0" \
  --force
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `count` | *(required)* | Number of neurons, assigned ids 0…N-1 |
| `--threshold` | `1.0` | Firing threshold (shared default) |
| `--tau` | `10.0` | Membrane time constant τ (shared default) |
| `--input-current` | `0.0` | Input drive for every neuron |
| `--refractory-period` | `None` | Per-neuron refractory period |
| `--reset-potential` | `None` | Membrane reset value after spike |
| `--neuron-overrides` | — | Per-neuron exceptions (see format below) |
| `--topology` | `none` | Auto-wire preset (see topologies below) |
| `--weight` | `0.5` | Synapse weight for topology presets |
| `--density` | `0.3` | Edge probability for `random` topology |
| `--seed` | `None` | RNG seed for `random` (reproducible) |
| `--steps` | `10` | Simulation steps |
| `--dt` | `1.0` | Time-step size |
| `--global-refractory` | `0.0` | Simulation-level refractory period |
| `--force` | — | Overwrite existing file |

**Per-neuron override format:**

```
"<id>:key=val,key=val,..."
```

Multiple neurons are space-separated (one quoted token per neuron):

```bash
--neuron-overrides "0:threshold=0.9,input_current=1.2" "2:tau=5" "4:refractory_period=2"
```

Supported keys: `threshold`, `tau` (alias for `membrane_time_constant`), `membrane_time_constant`, `refractory_period`, `reset_potential`, `input_current`.

**Topology presets:**

| Value | Wiring |
|---|---|
| `none` | No synapses — add them manually |
| `chain` | 0→1→2→…→N-1 |
| `ring` | Chain + N-1→0 closing the loop |
| `all-to-all` | Every ordered pair (no self-loops) |
| `random` | Each edge independently included with probability `--density` |

---

### `init --from-config` — recipe-driven setup

Seed an entire network from a compact JSON config file. This is the fastest way to version-control reusable topologies.

```bash
python -m tools.network_builder <path> init --from-config <config.json> [--force]
```

**Example:**

```bash
python -m tools.network_builder samples/my.json init \
  --from-config samples/network_config.json --force
```

See [Config Recipe Reference](#7-config-recipe-reference) for all supported fields, and [`samples/network_config.json`](../samples/network_config.json) for a ready-to-use example.

---

### Incremental editing (`add-neuron`, `add-synapse`, …)

Fine-tune an existing network without regenerating it from scratch:

```bash
# Add a neuron
python -m tools.network_builder samples/my.json add-neuron 5 \
  --threshold 0.9 --membrane-time-constant 6 --input-current 0.5

# Add a synapse
python -m tools.network_builder samples/my.json add-synapse 0 5 0.7

# Update an existing synapse weight
python -m tools.network_builder samples/my.json add-synapse 0 5 -0.3 --replace

# Change input current for neuron 2
python -m tools.network_builder samples/my.json set-input 2 1.0

# Update simulation settings
python -m tools.network_builder samples/my.json set-config --steps 50 --dt 0.25

# Verify integrity
python -m tools.network_builder samples/my.json validate

# Human-readable summary
python -m tools.network_builder samples/my.json summary
```

---

## 2. Running Simulations

```bash
python main.py <network.json>
```

`main.py` loads the network and runs a simulation using the best available backend:

- **Simple (default):** pure-Python leaky integrate-and-fire.
- **snnTorch:** used automatically if installed.
- **SpikingJelly:** used automatically if installed.

Simulation output (spike trains, voltages per step) is printed to stdout and also written to `samples/spike_history.json` for later use in animations.

---

## 3. Computing Layouts

The layout solver computes 3D node positions using force-directed equilibrium:

```bash
python -m tools.layout_demo <network.json> --output <layout.json>
```

The output JSON maps neuron ids to `[x, y, z]` coordinates and is consumed by all preview generators.

---

## 4. Generating Visualizations

### Static 3D Preview

```bash
python -m tools.render_preview <network.json> \
  --layout <layout.json> \
  --output output/network_preview.html
```

Produces an interactive Plotly 3D scene showing nodes (sized by degree, colored by input current) and directional synapse cones.

### Animated Spike Playback

```bash
python -m tools.animate_preview <network.json> \
  --layout <layout.json> \
  --spikes <spike_history.json> \
  --output output/spike_animation.html
```

Produces a frame-by-frame animation with speed controls, active-path highlighting, and recent-spike trail rings.

### WebGPU High-Performance Renderer

```bash
python -m tools.webgpu_preview <network.json> \
  --layout <layout.json> \
  --history <spike_history.json> \
  --output output/webgpu_preview.html
```

Generates a GPU-accelerated renderer using WebGPU compute shaders — much faster for large networks (thousands of neurons).

---

## 5. Viewing Output

Open `viewer/index.html` in any modern browser to navigate all generated previews, or open the HTML files directly:

```
output/network_preview.html    — static 3D inspection
output/spike_animation.html    — animated spike playback
output/webgpu_preview.html     — WebGPU high-performance view
```

For mobile/Termux, serve locally:

```bash
cd output && python -m http.server 8000
# then open http://127.0.0.1:8000 in your browser
```

---

## 6. CLI Reference

### `network_builder` subcommands

| Subcommand | Positional args | Key options |
|---|---|---|
| `generate` | `count` | `--topology`, `--tau`, `--threshold`, `--input-current`, `--neuron-overrides`, `--weight`, `--density`, `--seed`, `--steps`, `--dt`, `--force` |
| `init` | — | `--steps`, `--dt`, `--refractory-period`, `--from-config`, `--force` |
| `add-neuron` | `id` | `--threshold`, `--membrane-time-constant`, `--refractory-period`, `--reset-potential`, `--initial-voltage`, `--input-current` |
| `add-synapse` | `source target weight` | `--replace` |
| `set-input` | `id current` | — |
| `set-config` | — | `--steps`, `--dt`, `--refractory-period` |
| `summary` | — | — |
| `validate` | — | — |

---

## 7. Config Recipe Reference

A config recipe is a JSON file that fully describes a network:

```jsonc
{
  // ── Required ───────────────────────────────────────────────────────
  "count": 6,              // Number of neurons (ids 0…N-1)

  // ── Simulation settings ────────────────────────────────────────────
  "steps": 20,             // Simulation time steps
  "dt": 0.5,               // Time-step size
  "global_refractory": 1.0,// Simulation-level refractory period

  // ── Shared neuron defaults (apply to every neuron) ─────────────────
  "threshold": 1.0,        // Firing threshold
  "tau": 8.0,              // Membrane time constant τ
  "input_current": 0.0,    // Input drive
  // "reset_potential": -0.1, (optional)

  // ── Topology preset ────────────────────────────────────────────────
  "topology": "ring",      // none | chain | ring | all-to-all | random
  "weight": 0.6,           // Synapse weight for preset edges
  "density": 0.3,          // (random only) edge probability
  "seed": 42,              // (random only) RNG seed

  // ── Per-neuron exceptions ───────────────────────────────────────────
  "neuron_overrides": [
    { "id": 0, "input_current": 1.5, "threshold": 0.8 },
    { "id": 3, "tau": 4.0, "refractory_period": 2.0 },
    { "id": 5, "input_current": 0.9, "threshold": 1.2 }
  ],

  // ── Explicit extra synapses (stacked on top of preset) ─────────────
  "synapses": [
    { "from": 0, "to": 3, "weight": -0.4 },
    { "from": 5, "to": 1, "weight": 0.3 }
  ]
}
```

All fields except `count` are optional. Missing fields fall back to their defaults.
