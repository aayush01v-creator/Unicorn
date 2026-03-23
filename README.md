Unicorn is an early-stage cross-platform neural network visualization project focused on three ideas:

- simulate simple spiking neural networks
- generate automatic 3D layouts using force-directed placement
- preview the network in an interactive 3D browser view

Right now, the repo is a working prototype with a minimal backend, a simple force-layout engine, and a Plotly-based 3D preview.

## Current Features

- JSON-based network loading
- simple spiking neural simulation
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

main.py loads a sample network and runs a very small spiking simulation.

2. Layout

layout_demo.py computes 3D neuron positions using a simple force-directed layout:

neurons repel each other

synapses act like springs


3. Preview

render_preview.py generates an interactive 3D HTML visualization that can be opened in a browser.
