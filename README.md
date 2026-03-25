Unicorn is an early-stage cross-platform neural network visualization project focused on three ideas:

- simulate simple spiking neural networks
- generate automatic 3D layouts using force-directed placement
- preview the network in an interactive 3D browser view

Right now, the repo is a working prototype with a minimal backend, a simple force-layout engine, and Plotly-based 3D previews for both static inspection and spike animation.


## Near-Term Roadmap

The current priority order is:

1. strengthen the simulation layer
2. improve visualization and playback quality
3. add collaboration features only after those foundations stabilize

A shared collaboration layer is intentionally deferred for now so we do not lock in sync behavior around a prototype whose data model and UI are still changing quickly.

## Current Features

- JSON-based network loading
- leaky integrate-and-fire simulation with membrane decay, refractory periods, and configurable timesteps
- optional framework-backed simulation selection (`simple`, `snntorch`, `spikingjelly`, or `auto`)
- 3D force-directed layout generation
- interactive 3D HTML preview with synapse direction arrows
- synapse weight labels plus color intensity mapped to edge strength
- excitatory vs inhibitory synapse coloring for easier circuit inspection
- animated spike playback in the browser
- sample network and generated layout output
- Termux-friendly development workflow

## Project Structure

```text
Unicorn/
├── backend/
│   ├── data_loader/
│   │   └── json_loader.py
│   └── neuron_sim/
│       ├── framework_runner.py
│       ├── simple_snn.py
│       └── torch_snn.py
├── physics_engine/
│   └── force_layout/
│       └── simple_layout.py
├── samples/
│   ├── network.json
│   ├── layout_output.json
│   └── spike_history.json
├── viewer/
│   ├── network_preview.html
│   └── spike_animation.html
├── layout_demo.py
├── main.py
├── render_preview.py
├── animate_preview.py
└── README.md
```

## What It Does

1. **Simulation**

   `main.py` loads a sample network and runs a lightweight leaky integrate-and-fire simulation with timestep-scaled membrane decay and refractory handling.

2. **Layout**

   `layout_demo.py` computes 3D neuron positions using a simple force-directed layout where neurons repel each other and synapses act like springs.

3. **Preview**

   `render_preview.py` generates a static interactive 3D HTML visualization, while `animate_preview.py` generates a time-based spike playback view.

## Reading the 3D Preview

The browser preview now exposes the main connectivity cues directly in 3D:

- **Arrow direction:** each synapse renders with a cone marker near the target neuron so you can see signal flow at a glance.
- **Weight labels:** every synapse midpoint shows a signed numeric label such as `+0.70` or `-0.40`.
- **Weight intensity:** stronger weights appear with more saturated edge coloring.
- **Excitatory vs inhibitory colors:** positive weights render in the green side of the diverging scale, while negative weights render in the red side.
- **Animated spikes:** neurons still pulse in the animation so you can correlate structure with activity over time.

## Simulation Configuration

Each neuron can override a few lightweight LIF parameters in `samples/network.json` or any other network config:

- `threshold`: firing threshold for that neuron
- `membrane_time_constant`: controls how quickly the membrane voltage decays back toward rest
- `refractory_period`: amount of simulated time the neuron stays reset after a spike
- `reset_potential`: optional post-spike reset voltage
- `initial_voltage`: optional starting membrane voltage

Global simulation settings include:

- `dt`: simulation timestep used for integration and refractory countdowns
- `steps`: number of simulation steps to run
- `input_current`: constant external drive per neuron



## Framework-Backed Simulation

Unicorn now supports selecting a simulation backend via `simulator` in your network JSON:

- `"simple"`: existing pure-Python reference simulator
- `"snntorch"`: requires `snntorch` (and `torch`)
- `"spikingjelly"`: requires `spikingjelly` (and `torch`)
- `"auto"` (default): tries `snntorch`, then `spikingjelly`, then falls back to `simple`

Example:

```json
{
  "simulator": "snntorch",
  "dt": 0.5,
  "steps": 20
}
```

If the chosen framework is not installed, Unicorn raises an import error so missing dependencies are explicit.

### Optional framework installs

Install one of the framework backends if you want tensor-accelerated simulation:

```bash
# snnTorch backend
pip install torch snntorch

# SpikingJelly backend
pip install torch spikingjelly
```

> Note: both framework modes currently run Unicorn's LIF update loop through the Torch-backed simulator while validating that your selected framework package is installed.

## CLI Network Builder

If editing raw JSON by hand gets tedious, use `network_builder.py` to create or update network files directly from the terminal. It works with `samples/network.json` by default, but you can point it at any other path first in the command.

```bash
python network_builder.py samples/network.json summary
python network_builder.py samples/network.json add-neuron 3 --threshold 1.1 --membrane-time-constant 5.0 --input-current 0.4
python network_builder.py samples/network.json add-synapse 2 3 0.8
python network_builder.py samples/network.json set-config --steps 20 --dt 0.25
python network_builder.py samples/network.json validate
```

Supported commands:

- `init`: create a fresh network JSON file
- `add-neuron`: add a neuron and optionally seed its input current
- `add-synapse`: add or replace a connection
- `set-input`: update per-neuron external current
- `set-config`: update global simulation settings
- `summary` / `validate`: inspect the resulting network

## Tutorial: Generate and Inspect the Browser Preview

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Review or edit the sample network

Open `samples/network.json` and define neurons plus synapses. Use positive weights for excitatory connections and negative weights for inhibitory ones. For example:

```json
{
  "from": 2,
  "to": 0,
  "weight": -0.5
}
```

That single negative weight is enough to make the preview render the edge in the inhibitory color range.

### 3. Generate a layout

```bash
python layout_demo.py
```

This produces `samples/layout_output.json`, which stores the 3D coordinates used by the preview renderers.

### 4. Build the static 3D preview

```bash
python render_preview.py
```

Open `viewer/network_preview.html` in a browser and inspect:

- cones showing synapse direction
- signed text labels at synapse midpoints
- diverging edge colors indicating inhibitory vs excitatory strength

### 5. Build the animated spike preview

```bash
python animate_preview.py
```

Open `viewer/spike_animation.html` to replay spikes with the same connectivity overlays preserved. The animation now includes recent-spike trail rings, per-step timestamps in the slider and title, multiple play-speed buttons, and active-path highlighting for synapses driven by the current spikes. The script also writes `samples/spike_history.json` so you can inspect the simulation output separately.

### 6. Iterate on readability

A practical workflow is:

1. edit `samples/network.json`
2. rerun `python layout_demo.py`
3. rerun `python render_preview.py` and/or `python animate_preview.py`
4. refresh the browser tab

If your preview looks too dense, reduce the number of edges temporarily or scale the network into smaller subcircuits before rendering.
