# Unicorn Setup Guide for Termux (Android)

This guide walks through a full, from-scratch setup of Unicorn on an Android phone or tablet using Termux.

---

## 1) Prerequisites

- Android device with internet access
- At least ~1.5 GB free storage (recommended 3+ GB)
- A modern browser on the same device (Chrome/Edge/Firefox) to open generated HTML previews

> Tip: If you are using battery saver, disable it during setup so long installs do not get interrupted.

---

## 2) Install Termux

1. Install **Termux** from **F-Droid** (recommended) or GitHub releases.
2. Open Termux once and let it initialize.

---

## 3) Update packages and grant storage access

Run:

```bash
pkg update && pkg upgrade -y
termux-setup-storage
```

- Approve the Android storage prompt when asked.

---

## 4) Install required system packages

Run:

```bash
pkg install -y git python clang make pkg-config libjpeg-turbo libpng
```

Notes:
- `python` is required for all Unicorn scripts.
- `clang/make/pkg-config` help with building wheels when prebuilt wheels are unavailable.
- `libjpeg-turbo` and `libpng` help image/plot-related dependencies build cleanly.

---

## 5) Clone the Unicorn repository

Pick a working folder and clone:

```bash
cd ~/storage/shared
mkdir -p dev && cd dev
git clone <YOUR-UNICORN-REPO-URL> Unicorn
cd Unicorn
```

If you already cloned it, just enter it:

```bash
cd ~/storage/shared/dev/Unicorn
```

---

## 6) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

You should now see `(.venv)` in the prompt.

---

## 7) Install Python dependencies

```bash
pip install -r requirements.txt
```

If installation fails for a package, retry once after upgrading tooling:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 8) Run tests (recommended health check)

```bash
pip install pytest
pytest -q
```

If a specific test fails because of a platform/browser limitation on Android, you can continue with the core runtime steps below.

---

## 9) Run the core simulation

Generate spike history from sample data:

```bash
python main.py samples/network.json
```

Expected output artifact:

- `samples/spike_history.json`

---

## 10) Generate 3D layout

```bash
python layout_demo.py samples/network.json --output samples/layout_output.json
```

Expected output artifact:

- `samples/layout_output.json`

---

## 11) Generate static HTML preview

```bash
python render_preview.py samples/network.json --layout samples/layout_output.json --output viewer/network_preview.html
```

---

## 12) Generate animated spike preview

```bash
python animate_preview.py samples/network.json --layout samples/layout_output.json --spikes samples/spike_history.json --output viewer/spike_animation.html
```

---

## 13) Open previews on Android

You have two easy options.

### Option A: Open file directly

Use Android file manager to open:

- `viewer/network_preview.html`
- `viewer/spike_animation.html`

### Option B: Serve locally from Termux (more reliable)

```bash
cd viewer
python -m http.server 8000
```

Then open in browser:

- `http://127.0.0.1:8000/network_preview.html`
- `http://127.0.0.1:8000/spike_animation.html`

Stop server with `Ctrl+C`.

---

## 14) Run with alternate input formats

Unicorn supports multiple network formats.

```bash
python main.py samples/network.sonata.json
python main.py samples/network.nml
```

You can also generate layout/preview from those files similarly.

---

## 15) Optional framework-backed simulation notes (Termux)

Unicorn supports optional `snntorch` / `spikingjelly` backends, both of which depend on PyTorch.

On Android Termux, PyTorch wheels are often unavailable or difficult to build. If that happens, use the default `simple` backend (recommended on Termux).

In your network JSON, keep:

```json
{
  "simulator": "simple"
}
```

or omit the field to let Unicorn choose fallback behavior.

---

## 16) Daily workflow cheat sheet

From repo root:

```bash
source .venv/bin/activate
python main.py samples/network.json
python layout_demo.py samples/network.json --output samples/layout_output.json
python render_preview.py samples/network.json --layout samples/layout_output.json --output viewer/network_preview.html
python animate_preview.py samples/network.json --layout samples/layout_output.json --spikes samples/spike_history.json --output viewer/spike_animation.html
```

---

## 17) Troubleshooting

### `pip install -r requirements.txt` fails

- Run `pkg update && pkg upgrade -y`
- Ensure you are inside `.venv`
- Upgrade packaging tools:
  - `pip install --upgrade pip setuptools wheel`

### Browser opens a blank page

- Use local server mode (`python -m http.server 8000`) instead of opening raw file paths.
- Try Chrome if your current browser blocks local JS resources.

### `No module named ...`

- Ensure virtual environment is active: `source .venv/bin/activate`
- Re-run: `pip install -r requirements.txt`

### Permission/path issues with shared storage

- Re-run `termux-setup-storage`
- Prefer repo location under `~/storage/shared/dev`

---

## 18) Keeping your local copy up to date

```bash
cd ~/storage/shared/dev/Unicorn
git pull
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 19) What to do next

- Replace `samples/network.json` with your own network file.
- Tune simulation settings (`dt`, `steps`, `input_current`) and layout settings under `layout`.
- Add more neurons quickly with the built-in CLI instead of hand-editing JSON.
- From the **repo root** (`~/storage/shared/dev/Unicorn`), run `network_builder.py` from the top level (not `samples/network_builder.py`):

```bash
cd ~/storage/shared/dev/Unicorn
python ./network_builder.py samples/network.json add-neuron 3 --threshold 1.1 --membrane-time-constant 5.0 --initial-voltage 0.2 --input-current 0.4
python ./network_builder.py samples/network.json add-synapse 2 3 0.8
python ./network_builder.py samples/network.json summary
```

- Customize voltage behavior easily by setting per-neuron `initial_voltage`, `reset_potential`, and `threshold` values (via `network_builder.py` flags or direct JSON edits), then rerun `python main.py samples/network.json` to compare spike/voltage output.
- Generate new previews and compare behavior.
