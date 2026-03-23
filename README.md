Unicorn is an early-stage cross-platform neural network visualization project focused on three ideas:

- simulate simple spiking neural networks
- generate automatic 3D layouts using force-directed placement
- preview the network in an interactive 3D browser view

Right now, the repo is a working prototype with a minimal backend, a simple force-layout engine, and a Plotly-based 3D preview.

## Current Features

- JSON-based network loading
- leaky integrate-and-fire simulation with membrane decay, refractory periods, and configurable timesteps
- 3D force-directed layout
- interactive 3D HTML preview
- sample network and generated layout output
- Termux-friendly development workflow

## Project Structure

```text
Unicorn/
├── backend/
│   ├── data_loader/
│   │   └── json_loader.py
│   └── neuron_sim/
│       └── simple_snn.py
├── physics_engine/
│   └── force_layout/
│       └── simple_layout.py
├── samples/
│   ├── network.json
│   └── layout_output.json
├── viewer/
│   └── network_preview.html
├── layout_demo.py
├── main.py
├── render_preview.py
├── requirements.txt
└── README.md
```

What It Does

1. Simulation

main.py loads a sample network and runs a lightweight leaky integrate-and-fire simulation with timestep-scaled membrane decay and refractory handling.

2. Layout

layout_demo.py computes 3D neuron positions using a simple force-directed layout:

neurons repel each other

synapses act like springs


3. Preview

render_preview.py generates an interactive 3D HTML visualization that can be opened in a browser.


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
