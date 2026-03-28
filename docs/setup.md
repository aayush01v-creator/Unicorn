# Unicorn Setup Guide (including Android / Termux)

This guide provides a concise setup path for Unicorn across standard desktop environments and mobile terminals (like Termux).

## Installation

1. **System Dependencies:**
   - **Desktop**: Ensure standard build tools and Python 3.9+ are installed. 
   - **Termux**: `pkg update && pkg upgrade -y && pkg install -y git python clang make pkg-config libjpeg-turbo libpng`

2. **Clone and Virtual Environment:**
   Run these in your preferred directory (or `~/storage/shared/dev` for Android):
   ```bash
   git clone <YOUR-UNICORN-REPO-URL> Unicorn
   cd Unicorn
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel
   ```

3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *(On Termux, prefer the default simple Python backend as PyTorch wheels are frequently unavailable).*

## Generating Previews

From the repository root with your virtual environment activated:

```bash
# 1. Run simulation to get dynamics history
python main.py samples/network.json

# 2. Compute 3D forces layout
python -m tools.layout_demo samples/network.json --output samples/layout_output.json

# 3. Render previews to the output/ directory
python -m tools.render_preview samples/network.json --layout samples/layout_output.json --output output/network_preview.html
python -m tools.animate_preview samples/network.json --layout samples/layout_output.json --spikes samples/spike_history.json --output output/spike_animation.html
```

## Viewing the Output

Open the generated files from your `output/` directory:
- **Desktop**: Simply open `output/network_preview.html` in Chrome/Firefox/Edge.
- **Termux**: Run a local HTTP server:
  ```bash
  cd output
  python -m http.server 8000
  ```
  Then navigate to `http://127.0.0.1:8000/network_preview.html` in your mobile web browser.

## Next Steps

### Build networks instantly with `generate`

Instead of adding neurons one at a time, create an entire network in a single command:

```bash
# 8-neuron ring — all neurons share the same defaults
python -m tools.network_builder samples/network.json generate 8 \
  --topology ring --tau 8 --steps 20 --dt 0.5 --force

# 5-neuron random network with per-neuron overrides
python -m tools.network_builder samples/network.json generate 5 \
  --topology random --density 0.4 --seed 42 \
  --neuron-overrides "0:input_current=1.2,threshold=0.8" "3:tau=4" \
  --force
```

### Seed a network from a config file

Version-control your network topology as a compact JSON recipe:

```bash
python -m tools.network_builder samples/network.json init \
  --from-config samples/network_config.json --force
```

See [`samples/network_config.json`](../samples/network_config.json) for a fully annotated example.

### Incremental editing

Fine-tune an existing network without regenerating it:

```bash
python -m tools.network_builder samples/network.json add-neuron 3 --input-current 0.4
python -m tools.network_builder samples/network.json add-synapse 2 3 0.8
python -m tools.network_builder samples/network.json set-input 0 1.5
python -m tools.network_builder samples/network.json summary
```

For a complete CLI reference covering all subcommands and options, see [`docs/user_guide.md`](user_guide.md).
