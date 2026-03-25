# Unicorn Architecture

## Current Runtime Split

Unicorn currently keeps simulation and layout generation in Python, then renders precomputed results in browser-based viewers.

- `backend/`: network loading and SNN stepping
- `physics_engine/`: force-directed layout solver
- `viewer/`: HTML-based 3D preview and spike playback

## High-Performance Graphics Rendering Plan

Today, browser previews are built with Plotly and are excellent for inspection-scale workloads. For very large simulations (thousands of neurons and millions of spikes), Unicorn should move toward a GPU-first rendering stack with explicit data-oriented pipelines.

### Why move beyond traditional WebGL wrappers

Open-source efforts such as SNNtrainer3D show that WebGL + Three.js can deliver compelling 3D neural visualization in the browser. That remains a good baseline for compatibility and rapid iteration.

At Unicorn's target scale, however, the bottleneck shifts to:

- per-frame CPU orchestration
- draw-call overhead
- limited control of memory transfers and synchronization

### WebGPU as the primary high-scale path

Adopt **WebGPU** as the advanced renderer path across modern Chromium, WebKit, and Gecko-based browsers. WebGPU exposes lower-level GPU primitives and explicit command submission, which allows:

- persistent GPU buffers for neuron positions, morphology metadata, and spike event streams
- batched/instanced rendering for dense node and edge populations
- compute shaders for spike pulse propagation, bloom masks, and temporal decay fields
- reduced CPU-GPU synchronization points during playback

This model keeps spike animation logic on the GPU, where millions of simultaneous traveling signals can be processed in parallel.

### Proposed render pipeline

1. **Upload phase:** serialize network geometry once into device-local buffers.
2. **Streaming phase:** append per-timestep spike events into ring buffers.
3. **Compute pass:** update pulse state (position, intensity, lifetime) in parallel.
4. **Render pass:** draw neurons/synapses using instancing and GPU-side color ramps.
5. **Post-process pass (optional):** apply glow/halo effects for active spikes.

### Data layout guidance

- Prefer struct-of-arrays layouts for coalesced memory access in compute kernels.
- Use compact index buffers for synapse traversal and source/target lookup.
- Keep spike events in fixed-size chunks to avoid reallocations during bursts.
- Encode static vs dynamic attributes separately to minimize upload bandwidth.

### Capability tiers

To preserve broad compatibility without blocking high-end performance:

- **Tier A (default modern path):** WebGPU renderer.
- **Tier B (fallback):** existing WebGL/Plotly preview for unsupported environments.
- **Tier C (headless export):** server/offline frame generation for datasets too large for interactive sessions.

### Incremental migration strategy

1. Keep current viewer flow as a stable baseline.
2. Add a WebGPU prototype for neuron point-cloud rendering.
3. Introduce compute-driven spike pulse animation and benchmark CPU/GPU frame time.
4. Port synapse rendering and visual effects.
5. Gate advanced effects behind runtime capability checks and quality presets.


## Real-Time Collaboration Architecture (Experimental)

Unicorn's collaboration runtime now splits synchronization by data type:

### 1) Structural Sync with CRDTs (Yjs)

- The declarative scene graph (`neurons`, spatial coordinates, `synapses`, and weights) is represented as a shared Yjs map.
- Peers join a shared room via `y-webrtc`, which handles eventual-consistent merge semantics for concurrent edits.
- Result: two collaborators can edit different hidden layers at the same time without merge conflicts or lock coordination.

### 2) Live Spike Sync with WebRTC Snapshot Interpolation

- High-frequency spike activity is intentionally **not** synchronized through CRDT operations.
- A host streams compact spike snapshots (`step`, timestamp, spike vector) over direct WebRTC data channels.
- Receiving peers interpolate between snapshots and apply local temporal decay in the render loop to preserve smooth pulse animation under network jitter.

### Session model

- `?session=<id>&role=host` initializes shared structure and broadcasts snapshots.
- `?session=<id>&role=peer` consumes CRDT structure updates + spike snapshots and renders interpolated activity.

This split keeps structural collaboration conflict-free while avoiding CRDT overhead for per-frame simulation data.

## Native Physics Engine Integration Plan

To support large networks and mobile/desktop UI clients without blocking UI rendering, integrate a native headless physics module for equilibrium solving.

### Engine choices

- **Rust + Rapier** for memory safety and predictable performance.
- **C++ + Jolt** when C++ toolchains or existing native ecosystems are preferred.

Both options should expose the same stable C ABI so UI clients can swap engines without changing higher-level app logic.

### Cross-platform bridge

- **Flutter:** call native entry points through `dart:ffi`.
- **React Native:** call native entry points through **JSI** host functions.

Target platforms:

- Android
- iOS
- Windows
- Linux

### Threading model (non-blocking UI)

1. UI serializes node/edge state into a compact native input buffer.
2. Bridge dispatches solve work to a background worker thread (or thread pool).
3. Native engine runs iterative force equilibrium steps to convergence.
4. Engine returns coordinates via callback/future/promise to UI runtime.
5. UI applies the new layout on the next render frame.

This keeps the main UI thread responsive while the solver runs at native speed.

### ABI contract

Use a narrow C ABI with explicit ownership rules:

- `create_solver(config) -> solver_handle`
- `solve_step(handle, graph_ptr, graph_len) -> status`
- `read_positions(handle, out_ptr, out_len)`
- `destroy_solver(handle)`

Guidelines:

- Keep all heap ownership on one side of the boundary.
- Use POD structs and contiguous arrays for low marshalling overhead.
- Return error codes plus optional diagnostic strings.

### Determinism and safety

- Fixed timestep, bounded iteration counts, and convergence thresholds.
- Seeded initialization for reproducible layouts.
- Validate indices and lengths at boundary crossings.
- Keep native state opaque to the UI layer.

### Migration path

1. Preserve current Python solver as reference behavior.
2. Introduce a native adapter behind a shared `LayoutSolver` interface.
3. Add parity tests on small sample networks (position tolerance based).
4. Benchmark throughput/latency on all target OSes.
5. Promote native engine to default for production UI clients.

### Performance expectations

With native solver execution and low-copy boundary formats, layout equilibrium computation can run near bare-metal speed while keeping frame rendering and gesture handling smooth.
