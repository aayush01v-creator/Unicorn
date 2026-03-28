# Unicorn: Structured Probabilistic Model Visualizer

Unicorn is a cross-platform tool that visualizes and simulates structured probabilistic models as
interactive graphs. It connects random variables (nodes) with direct interactions (edges),
integrating underlying dynamics like time-stepped decays into readable 3D structures.

## At a Glance

- **Standardized Loading**: Imports from Unicorn JSON, SONATA-style, and NeuroML natively.
- **Physics Layout**: Force-directed placement computes responsive 3D topologies.
- **Model Inspection**: Clear static graphs or animated state-playbacks in any WebGL/WebGPU browser.
- **Flexible Backends**: Ships with a pure-Python simulator; auto-scales to `snntorch` or `spikingjelly` if available.
- **Powerful CLI**: Build networks instantly — set neuron count, properties, and topology in one command.

## Reading the Graph

- **Nodes vs Edges**: Nodes represent random variables; edges represent their direct interactions.
- **Color Coding**: Positive interactions render in Green (excitatory); Negative in Red (inhibitory). High-intensity edges saturate in color.
- **Interactive Telemetry**: Hovering shows per-variable metrics: degree, threshold, τ, refractory period, and input current.

## Quickstart

1. **Install dependencies:**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Build a network** (one command — no manual JSON editing needed):
   ```bash
   # 8-neuron ring network, all defaults
   python -m tools.network_builder samples/my_network.json generate 8 --topology ring --force

   # Custom properties + per-neuron overrides
   python -m tools.network_builder samples/my_network.json generate 6 \
     --topology ring --tau 8 --threshold 1.0 --input-current 0.2 --steps 20 --dt 0.5 \
     --neuron-overrides "0:input_current=1.5,threshold=0.8" "3:tau=4" --force
   ```

3. **Simulate Dynamics & Layout:**
   ```bash
   python main.py samples/my_network.json
   python -m tools.layout_demo samples/my_network.json --output samples/layout_output.json
   ```

4. **Generate Static & Animated Previews:**
   ```bash
   python -m tools.render_preview samples/my_network.json --layout samples/layout_output.json --output output/network_preview.html
   python -m tools.animate_preview samples/my_network.json --layout samples/layout_output.json --spikes samples/spike_history.json --output output/spike_animation.html
   ```

Open the resulting `output/*.html` files in any modern browser.

For full setup notes (including Android/Termux environments), see [`docs/setup.md`](docs/setup.md).  
For a complete CLI reference and config-driven workflow, see [`docs/user_guide.md`](docs/user_guide.md).

---

## `network_builder` — Quick Reference

All commands run as `python -m tools.network_builder <path> <subcommand> [options]`.

### `generate` — build an entire network in one step

```bash
python -m tools.network_builder <path> generate <count> [options] [--force]
```

| Option | Default | Description |
|---|---|---|
| `count` | *(required)* | Number of neurons (ids 0…N-1) |
| `--threshold` | `1.0` | Firing threshold for all neurons |
| `--tau` | `10.0` | Membrane time constant τ |
| `--input-current` | `0.0` | Input drive applied to every neuron |
| `--refractory-period` | `None` | Per-neuron refractory period |
| `--reset-potential` | `None` | Membrane reset value after spike |
| `--neuron-overrides` | — | Space-separated `id:key=val,...` overrides |
| `--topology` | `none` | `none` · `chain` · `ring` · `all-to-all` · `random` |
| `--weight` | `0.5` | Synapse weight for topology presets |
| `--density` | `0.3` | Fraction of edges to create for `random` topology |
| `--seed` | `None` | RNG seed for `random` topology (reproducible) |
| `--steps` | `10` | Number of simulation steps |
| `--dt` | `1.0` | Time step size |
| `--global-refractory` | `0.0` | Simulation-level refractory period |
| `--force` | — | Overwrite an existing file |

**Per-neuron override format:** `"<id>:key=val,key=val"` — supported keys are `threshold`, `tau` (or `membrane_time_constant`), `refractory_period`, `reset_potential`, `input_current`.

### `init --from-config` — seed a network from a JSON recipe

```bash
python -m tools.network_builder <path> init --from-config samples/network_config.json [--force]
```

See [`samples/network_config.json`](samples/network_config.json) for a fully annotated example.

### Other subcommands

| Subcommand | Description |
|---|---|
| `init` | Create an empty network file |
| `add-neuron <id>` | Add a single neuron |
| `add-synapse <src> <tgt> <weight>` | Add a synapse (use `--replace` to update) |
| `set-input <id> <current>` | Set input current for a neuron |
| `set-config` | Update `steps`, `dt`, or `refractory_period` |
| `summary` | Print a human-readable network summary |
| `validate` | Check structural integrity |

---

## Project Layout

```
Unicorn/
├── main.py                   # Simulation entry-point
├── backend/                  # Data loading & SNN stepping
│   ├── data_loader/          # JSON / SONATA / NeuroML parsers
│   └── neuron_sim/           # Pure-Python, snnTorch & SpikingJelly runners
├── physics_engine/
│   └── force_layout/         # Force-directed 3D layout solver
├── tools/                    # CLI utilities (run via python -m tools.<name>)
│   ├── network_builder.py    # Build / edit / generate networks
│   ├── layout_demo.py        # Compute & export 3D layout
│   ├── render_preview.py     # Static 3D HTML preview
│   ├── animate_preview.py    # Animated spike-playback HTML
│   └── webgpu_preview.py     # High-performance WebGPU renderer
├── viewer/
│   └── index.html            # Browser launcher for all previews
├── samples/                  # Example networks and config recipes
├── docs/                     # Setup guide, user guide, architecture
└── tests/                    # pytest test suite
```

