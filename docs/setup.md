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
python layout_demo.py samples/network.json --output samples/layout_output.json

# 3. Render previews to the output/ directory
python render_preview.py samples/network.json --layout samples/layout_output.json --output output/network_preview.html
python animate_preview.py samples/network.json --layout samples/layout_output.json --spikes samples/spike_history.json --output output/spike_animation.html
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

Use the included helper script to rapidly trace out new structured probability models or custom topological parameters from the command line:

```bash
python ./network_builder.py samples/network.json add-neuron 3 --input-current 0.4
python ./network_builder.py samples/network.json add-synapse 2 3 0.8
python ./network_builder.py samples/network.json summary
```
