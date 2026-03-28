# Unicorn: Structured Probabilistic Model Visualizer

Unicorn is a cross-platform tool that visualizes and simulates structured probabilistic models as interactive graphs. It connects random variables (nodes) with direct interactions (edges), integrating underlying dynamics like time-stepped decays into readable 3D structures.

## At a Glance

- **Standardized Loading**: Imports from Unicorn JSON, SONATA-style, and NeuroML natively.
- **Physics Layout**: Force-directed placement computes responsive 3D topologies.
- **Model Inspection**: Clear static graphs or animated state-playbacks in any WebGL/WebGPU browser.
- **Flexible Backends**: Ships with a pure-Python simulator, and automatically scales to `snntorch` or `spikingjelly` if available.
- **CLI Utility**: Rapidly edit configuration, nodes, and weights via command-line helpers.

## Reading the Graph

- **Nodes vs Edges**: Nodes represent random variables, and edges represent their direct interactions.
- **Color Coding**: Positive interactions are rendered in Green, reflecting excitatory flow; Negative interactions in Red. High intensity edges saturate in color.
- **Interactive Telemetry**: Hovering provides localized variable metrics, degrees, thresholds, and current input loads.

## Quickstart

1. **Install dependencies:**  
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Simulate Dynamics & Layout:**  
   ```bash
   python main.py samples/network.json
   python -m tools.layout_demo samples/network.json --output samples/layout_output.json
   ```
3. **Generate Static & Animated Previews:**  
   ```bash
   python -m tools.render_preview samples/network.json --layout samples/layout_output.json --output output/network_preview.html
   python -m tools.animate_preview samples/network.json --layout samples/layout_output.json --spikes samples/spike_history.json --output output/spike_animation.html
   ```

Open the resulting `output/*.html` files in any modern browser to inspect the model. 

For full setup notes (including Android/Termux environments), see `docs/setup.md`.
