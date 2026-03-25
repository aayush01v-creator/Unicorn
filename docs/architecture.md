# Unicorn Architecture

## Current Runtime Split

Unicorn currently keeps simulation and layout generation in Python, then renders precomputed results in browser-based viewers.

- `backend/`: network loading and SNN stepping
- `physics_engine/`: force-directed layout solver
- `viewer/`: HTML-based 3D preview and spike playback

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
