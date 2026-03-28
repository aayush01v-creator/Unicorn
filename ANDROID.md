# NeuroGen — Android App Documentation

<p align="center">
  <b>A real-time 3D neural network simulation engine for Android</b><br/>
  <i>Designed for research, education, and the pure beauty of computational neuroscience</i>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [Simulation Engine](#simulation-engine)
- [Rendering Engine](#rendering-engine)
- [Visual Topologies & Mathematical Shapes](#visual-topologies--mathematical-shapes)
- [Sound Design](#sound-design)
- [Settings & Customization](#settings--customization)
- [Performance Architecture](#performance-architecture)
- [Tech Stack](#tech-stack)
- [Download & Install](#download--install)
- [Minimum Requirements](#minimum-requirements)

---

## Overview

**NeuroGen** is a high-performance, interactive neural network simulator that runs entirely on an Android device. No internet connection, no server, no cloud — just raw computational neuroscience in the palm of your hand.

Built around a **Leaky Integrate-and-Fire (LIF)** spiking neural network model, NeuroGen lets you watch hundreds of artificial neurons fire, recover, and pass signals to one another in real time — rendered in a fully interactive 3D space.

Whether you're exploring how neural topologies affect collective behavior, demonstrating neuroscience concepts, or simply mesmerized by the electrical symphony of a firing network, NeuroGen delivers an experience that is equal parts scientific tool and living artwork.

---

## Key Capabilities

| Capability | Details |
|---|---|
| **Live 3D Simulation** | Runs a full Spiking Neural Network (SNN) in real-time at 60 fps |
| **Interactive Camera** | Freely rotate, pitch, and yaw the network in 3D via touch |
| **Gyroscope Control** | Tilt your device to orbit the neural field naturally |
| **Dynamic Node Jitter** | Each neuron drifts in organic, sinusoidal micro-motion for a "living" feel |
| **Synaptic Edge Glow** | Active connections emit colorful, animated electric halos |
| **Mathematical Topologies** | Position neurons in Brain, Torus, Galaxy, Astroid shapes — or define your own equations |
| **Neural Sound Effects** | Synthesized electrical SFX react to network firing activity in real-time |
| **Full Settings Persistence** | Every configuration auto-saves and restores across app restarts |
| **Color Palette Presets** | Six distinct visual themes: Neuro, Plasma, Fire, Ice, Matrix, Monochrome |
| **Simulation History** | Computes 300 time-steps of SNN history rendered as an animated playback |

---

## Simulation Engine

NeuroGen's simulation core implements a **Leaky Integrate-and-Fire (LIF)** spiking neural network.

### How It Works

Each neuron maintains a **membrane potential** `V(t)`. At every time step `dt`:

```
V(t+dt) = V(t) * exp(-dt/τ) + I_input + noise + Σ(w_ij * spike_j)
```

- **τ (tau)**: Membrane time constant — controls how fast voltage decays.
- **I_input**: External drive current injected into the neuron.
- **noise**: Optional Gaussian noise for stochastic behavior.
- **w_ij**: Synaptic weight from neuron `j` to neuron `i`.

When `V(t)` exceeds the **firing threshold**, the neuron emits a **spike** and immediately enters a **refractory period**, during which it cannot fire again. After the refractory window, the membrane resets and the cycle continues.

### Configurable Parameters

| Parameter | Range | Effect |
|---|---|---|
| Neuron Count | 50 – 500 | Number of simulated neurons |
| Connectivity | 1 – 20 | Average synaptic connections per neuron |
| Membrane τ | 1 – 50 ms | Voltage leak rate |
| Refractory Period | 0 – 20 ms | Post-spike silence window |
| Input Current | 0.0 – 1.0 | External excitation strength |
| Synaptic Gain | 0.5 – 5.0 | Signal amplification per synapse |
| Noise Std Dev | 0.0 – 1.0 | Stochastic membrane noise |

---

## Rendering Engine

The 3D renderer, `_VulkanGraphPainter`, is a custom high-performance `CustomPainter` built specifically around zero-allocation rendering.

### Projection Pipeline

All 3D neuron coordinates are projected to 2D screen space through a **software rotation matrix** applied live every frame:

```
[x']   [cos(yaw)  -sin(yaw)  0 ] [x]
[y'] = [sin(yaw)   cos(yaw)  0 ] [y]  * pitch_rotation
[z']   [0          0         1 ] [z]
```

The visible depth (`z'`) drives **perspective division** (`scale = fov / (fov - z')`), making near neurons appear larger and far neurons smaller — a classic pinhole camera model.

### Node Disturbance (Jitter)

Each neuron oscillates within a small local region using sinusoidal offsets:

```dart
dx = sin(t * freq + phase_x) * amplitude
dy = cos(t * freq + phase_y) * amplitude
```

Phase seeds are unique per neuron, so each one moves independently — creating the effect of a field of living, breathing nodes.

### Edge Glow

Synaptic edges between neurons that have recently fired are rendered with a multi-layer **radial glow** effect: a wide, translucent backdrop stroke is drawn first, followed by a bright, narrow core — simulating an electric arc. The glow intensity fades linearly from source to target neuron.

---

## Visual Topologies & Mathematical Shapes

Neurons can be distributed spatially in several preset topologies or in any shape described by a custom mathematical equation.

### Built-in Presets

| Preset | Description |
|---|---|
| **Sphere** | Uniformly random distribution in 3D spherical volume |
| **Cube** | Uniformly random distribution in a rectangular 3D box |
| **Brain** | Parametric surface approximating cerebral hemisphere geometry |
| **Torus** | Neurons arranged on the surface of a 3D torus (donut) |
| **Astroid** | A compact star-shaped 3D parametric curve |
| **Galaxy** | Logarithmic spiral arms + disk distribution |

### Custom Equations

NeuroGen includes a built-in **code editor** for custom mathematical topology. Type any expression using these variables:

| Variable | Meaning |
|---|---|
| `i` | Neuron index (0 to n-1) |
| `n` | Total neuron count |
| `u`, `v` | Random uniform samples in [0, 1] |
| `r` | Random sample for radius/scale |
| `pi` | π = 3.14159… |
| `e` | Euler's number ≈ 2.718… |

**Example — Cardioid XZ plane:**
```
X = (1 - sin(2*pi*i/n)) * cos(2*pi*i/n) * 4
Y = u * 2 - 1
Z = (1 - sin(2*pi*i/n)) * sin(2*pi*i/n) * 4
```

**Example — Lissajous curve:**
```
X = sin(3 * 2*pi*i/n) * 4
Y = cos(2 * 2*pi*i/n) * 4
Z = sin(2*pi*i/n) * 2
```

The expressions are parsed at runtime using the `math_expressions` Dart library. Any valid algebraic or trigonometric expression is supported.

---

## Sound Design

NeuroGen generates a live **electrical soundscape** that reacts to the neural network's firing activity.

- **Background Hum**: A continuous low-frequency drone. Volume scales with how many neurons are currently active — a quiet network hums softly, a hyperactive one buzzes loudly.
- **Spike Pops**: Short percussive electrical pops fire in sync with individual neuron spikes. The rate and density of pops mirrors the network's firing frequency in real-time.

All audio assets are pre-synthesized `.wav` files, played back through the `audioplayers` package with minimal CPU overhead. **Disabled by default** — enable via the Settings drawer.

---

## Settings & Customization

All settings are accessible via the **side drawer** and **auto-saved** using `shared_preferences`. Every preference is restored exactly as you left it, across app restarts.

### Network Settings
- Neuron count (min 50)
- Connectivity (edges per neuron)
- Membrane time constant (τ)
- Refractory period
- Synaptic gain
- Input current level
- Noise standard deviation

### Render Settings
- Color palette (6 presets)
- Edge glow toggle
- Node size
- Node jitter amplitude and frequency

### Shape Settings
- Topology preset (Sphere, Cube, Brain, Torus, Astroid, Galaxy, Custom)
- Custom equation editor (X, Y, Z expressions)

### Motion Settings
- Gyroscope control (on/off)

### Audio Settings
- SFX master toggle (off by default)

---

## Performance Architecture

NeuroGen was engineered from the ground up for high-throughput performance on mobile hardware.

### Memory-Contiguous Data Structures

The core simulation and rendering pipelines operate on **hardware-native typed arrays** instead of object graphs:

| Structure | Previous | Optimized |
|---|---|---|
| Neuron positions | `Map<int, List<double>>` | `Float32List` (stride-3) |
| Synapse connectivity | `List<Map<String, int>>` | `Int32List` (stride-2) |
| Spike history | `List<double>` | `Float32List` |
| Refractory states | `List<int>` | `Int32List` |

This eliminates **Garbage Collector pauses** during the 60 fps render loop — one of the primary causes of jank in Dart/Flutter applications.

### Zero-Allocation Render Loop

The `paint()` method inside `_VulkanGraphPainter` performs **no heap allocations** per frame. All projection computations work directly against the `Float32List` buffer. There are no intermediate `List<Offset>` or `Map` objects created during drawing.

### R8 Minification & Tree-Shaking

The Android release build applies aggressive dead-code elimination:

- **R8 minification** (`isMinifyEnabled = true`): Removes unused code from Dart AOT output and Java SDK.
- **Resource shrinking** (`isShrinkResources = true`): Strips unused assets, drawables, and strings.
- **Icon tree-shaking**: Flutter's build pipeline reduces `MaterialIcons-Regular.otf` from **1.6 MB → 3 KB** (99.8% reduction).

### Resulting APK Sizes (split-per-ABI)

| Architecture | APK Size |
|---|---|
| armeabi-v7a (32-bit ARM) | **16.7 MB** |
| arm64-v8a (64-bit ARM) | **19.3 MB** |
| x86_64 (emulators/x86 devices) | **20.6 MB** |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI Framework** | Flutter 3.x (Dart) |
| **Rendering** | Flutter `CustomPainter` with software 3D projection |
| **Simulation** | Custom LIF Spiking Neural Network in pure Dart |
| **Math Parser** | [`math_expressions`](https://pub.dev/packages/math_expressions) ^2.5.0 |
| **Settings Persistence** | [`shared_preferences`](https://pub.dev/packages/shared_preferences) ^2.3.1 |
| **Audio Playback** | [`audioplayers`](https://pub.dev/packages/audioplayers) ^5.2.1 |
| **Sensor Input** | [`sensors_plus`](https://pub.dev/packages/sensors_plus) (gyroscope/accelerometer) |
| **Build Toolchain** | Gradle 8.x + Kotlin DSL + Android R8 |
| **Minimum SDK** | Android 5.0 (API 21) |
| **Target SDK** | Android 13 (API 33) |
| **Data Layout** | `Float32List` / `Int32List` (dart:typed_data) |

---

## Download & Install

Pre-built APKs are available in the [`releases/`](releases/) directory. Download the correct APK for your device:

| File | Architecture | Best For |
|---|---|---|
| [`app-arm64-v8a-release.apk`](releases/app-arm64-v8a-release.apk) | 64-bit ARM | **Most modern Android phones (recommended)** |
| [`app-armeabi-v7a-release.apk`](releases/app-armeabi-v7a-release.apk) | 32-bit ARM | Older Android devices |
| [`app-x86_64-release.apk`](releases/app-x86_64-release.apk) | x86_64 | Android emulators / Intel-based tablets |

### Installation Steps

1. Download the correct `.apk` for your device.
2. On your Android device, go to **Settings → Security → Unknown Sources** and enable installation from unknown sources (or follow your device's specific prompt).
3. Open the downloaded `.apk` file and tap **Install**.
4. Launch **NeuroGen** from your app drawer.

> **Note:** These APKs are signed with a debug key. For production deployment to the Play Store, a release signing key would be required.

---

## Minimum Requirements

| Requirement | Minimum |
|---|---|
| Android Version | 5.0 Lollipop (API 21) |
| RAM | 2 GB recommended |
| CPU | ARMv7 or ARM64 |
| GPU | OpenGL ES 2.0 capable |
| Storage | ~25 MB free space |
| Sensors | Gyroscope (optional, for gyro control) |

---

<p align="center">
  Built with ❤️ for computational neuroscience enthusiasts, developers, and curious minds.
</p>
