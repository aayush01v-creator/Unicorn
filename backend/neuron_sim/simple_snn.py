"""Fast NumPy-vectorised leaky integrate-and-fire simulator.

Replaces all per-neuron and per-synapse Python loops with NumPy array
operations so that large networks (thousands to tens-of-thousands of
neurons) run orders of magnitude faster than the original pure-Python
implementation.

Complexity per step:
  - synaptic current : O(spikes * fan-out)  — sparse spike-based matmul
  - voltage update   : O(N)                 — element-wise array ops
  - spike detection  : O(N)                 — boolean mask + argwhere
  - refractory       : O(N)                 — integer countdown clamp

The public interface (constructor + .run()) is identical to SimpleSNN
so framework_runner.py requires no changes.
"""

import math

import numpy as np


class SimpleSNN:
    def __init__(self, config: dict):
        self.neurons = config["neurons"]
        self.synapses = config["synapses"]
        self.steps = config.get("steps", 10)
        self.dt = float(config.get("dt", 1.0))
        if self.dt <= 0:
            raise ValueError("dt must be greater than zero")

        n = len(self.neurons)
        self.n = n

        # ── per-neuron parameters (one NumPy array each) ────────────────
        self.v = np.array(
            [float(neu.get("initial_voltage", 0.0)) for neu in self.neurons],
            dtype=np.float64,
        )
        self.thresholds = np.array(
            [float(neu.get("threshold", 1.0)) for neu in self.neurons],
            dtype=np.float64,
        )
        self.reset_potentials = np.array(
            [float(neu.get("reset_potential", 0.0)) for neu in self.neurons],
            dtype=np.float64,
        )

        default_tau = float(config.get("membrane_time_constant", 10.0))
        taus = np.array(
            [float(neu.get("membrane_time_constant", default_tau)) for neu in self.neurons],
            dtype=np.float64,
        )
        if np.any(taus <= 0):
            raise ValueError("membrane_time_constant must be greater than zero")
        self.decay = np.exp(-self.dt / taus)  # shape (N,)

        default_refractory = float(config.get("refractory_period", 0.0))
        refractory_periods = np.array(
            [float(neu.get("refractory_period", default_refractory)) for neu in self.neurons],
            dtype=np.float64,
        )
        if np.any(refractory_periods < 0):
            raise ValueError("refractory_period must be non-negative")
        self.refractory_steps = np.ceil(refractory_periods / self.dt).astype(np.int32)
        self.refractory_countdown = np.zeros(n, dtype=np.int32)

        configured_current = config.get("input_current", [0.0] * n)
        ic = np.array([float(c) for c in configured_current], dtype=np.float64)
        if len(ic) < n:
            ic = np.concatenate([ic, np.zeros(n - len(ic))])
        elif len(ic) > n:
            ic = ic[:n]
        self.input_current = ic  # shape (N,)

        # ── weight matrix as a sparse COO-style structure ───────────────
        # For N ≤ ~8 000 we use a dense NxN float32 matrix (fast matmul).
        # For larger networks we store (pre, post, weight) arrays and use
        # a sparse dot so we never allocate an N²-element matrix.
        if n <= 8_000:
            W = np.zeros((n, n), dtype=np.float32)
            for syn in self.synapses:
                pre = int(syn["from"])
                post = int(syn["to"])
                if pre < 0 or post < 0 or pre >= n or post >= n:
                    raise ValueError(
                        f"Synapse {pre} -> {post} references an invalid neuron index"
                        f" for network size {n}"
                    )
                W[pre, post] = float(syn["weight"])
            self._W = W
            self._sparse = False
        else:
            pres, posts, weights = [], [], []
            for syn in self.synapses:
                pre = int(syn["from"])
                post = int(syn["to"])
                if pre < 0 or post < 0 or pre >= n or post >= n:
                    raise ValueError(
                        f"Synapse {pre} -> {post} references an invalid neuron index"
                        f" for network size {n}"
                    )
                pres.append(pre)
                posts.append(post)
                weights.append(float(syn["weight"]))
            self._pre = np.array(pres, dtype=np.int32)
            self._post = np.array(posts, dtype=np.int32)
            self._wvals = np.array(weights, dtype=np.float64)
            self._sparse = True

    # ── internal helpers ─────────────────────────────────────────────────

    def _synaptic_current(self, spikes: np.ndarray) -> np.ndarray:
        """Compute incoming synaptic current for every neuron.

        spikes : bool/float array of shape (N,) — 1.0 where neuron fired.
        returns : float64 array of shape (N,).
        """
        if not self._sparse:
            # spikes @ W  →  sum over firing pre-synaptic neurons
            return spikes.astype(np.float32) @ self._W  # shape (N,)
        else:
            # Sparse accumulation: only iterate over synapses whose pre fired
            firing = np.where(spikes)[0]
            if len(firing) == 0:
                return np.zeros(self.n, dtype=np.float64)
            # mask synapses from firing pre-neurons
            keep = np.isin(self._pre, firing)
            result = np.zeros(self.n, dtype=np.float64)
            np.add.at(result, self._post[keep], self._wvals[keep])
            return result

    # ── public interface ─────────────────────────────────────────────────

    def run(self) -> list[dict]:
        history = []
        spikes = np.zeros(self.n, dtype=np.float64)

        for t in range(self.steps):
            # 1. Synaptic drive from previous step's spikes
            syn_current = self._synaptic_current(spikes)

            # 2. Active neurons only (not in refractory)
            active = self.refractory_countdown == 0  # bool (N,)

            # 3. Voltage update
            self.v = np.where(
                active,
                self.v * self.decay + (self.input_current + syn_current) * self.dt,
                self.reset_potentials,
            )

            # 4. Spike detection
            spikes = np.where(active & (self.v >= self.thresholds), 1.0, 0.0)

            # 5. Reset spiking neurons and start refractory countdown
            fired = spikes.astype(bool)
            self.v = np.where(fired, self.reset_potentials, self.v)
            self.refractory_countdown = np.where(
                fired,
                self.refractory_steps,
                np.maximum(0, self.refractory_countdown - 1),
            )

            history.append(
                {
                    "step": t,
                    "time": round(t * self.dt, 10),
                    "voltages": self.v.tolist(),
                    "spikes": spikes.astype(int).tolist(),
                    "refractory_remaining": (
                        self.refractory_countdown * self.dt
                    ).tolist(),
                }
            )

        return history
