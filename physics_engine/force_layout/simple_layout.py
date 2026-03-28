"""Vectorised force-directed layout solver.

The pairwise Coulomb repulsion is computed via NumPy broadcasting instead
of a Python double-loop, reducing complexity from O(N²) Python iterations
to a single NumPy kernel call.  For very large N (> ~30 000) a chunked
strategy avoids materialising an N×N matrix in memory.

The public interface (constructor + .run()) is unchanged.
"""

import numpy as np


class ForceDirectedLayout:
    """Physics-driven force-directed layout for SNN topology visualization.

    Neurons are modeled as charged particles (Coulomb repulsion) and synapses
    as springs (Hooke attraction), then integrated with simple velocity damping.
    """

    # Chunk size for pairwise distance computation when N is very large.
    # Each chunk processes CHUNK rows at a time, keeping peak memory at
    # O(CHUNK * N) rather than O(N²).
    _CHUNK = 2_048

    def __init__(self, config: dict, dim: int = 3, seed: int = 42):
        self.neurons = config["neurons"]
        self.synapses = config["synapses"]
        self.n = len(self.neurons)
        self.dim = dim

        layout_cfg = config.get("layout", {})
        self.k_repulsion = float(layout_cfg.get("k_repulsion", 0.05))
        self.k_spring = float(layout_cfg.get("k_spring", 0.08))
        self.rest_length = float(layout_cfg.get("rest_length", 1.0))
        self.dt = float(layout_cfg.get("dt", 0.05))
        self.damping = float(layout_cfg.get("damping", 0.85))
        self.max_step = float(layout_cfg.get("max_step", 0.15))

        rng = np.random.default_rng(seed)
        self.positions = rng.uniform(-1.0, 1.0, size=(self.n, self.dim))
        self.velocities = np.zeros_like(self.positions)

        # Pre-build spring index arrays once for O(1) per-step spring forces
        if self.synapses:
            self._spring_i = np.array([s["from"] for s in self.synapses], dtype=np.int32)
            self._spring_j = np.array([s["to"] for s in self.synapses], dtype=np.int32)
            self._spring_k = np.array(
                [self.k_spring * abs(float(s.get("weight", 1.0))) for s in self.synapses],
                dtype=np.float64,
            )
        else:
            self._spring_i = np.empty(0, dtype=np.int32)
            self._spring_j = np.empty(0, dtype=np.int32)
            self._spring_k = np.empty(0, dtype=np.float64)

    # ── internal helpers ────────────────────────────────────────────────

    def _repulsion_forces(self) -> np.ndarray:
        """Vectorised Coulomb repulsion, chunked to control memory."""
        n, dim = self.n, self.dim
        forces = np.zeros((n, dim), dtype=np.float64)
        pos = self.positions

        for start in range(0, n, self._CHUNK):
            end = min(start + self._CHUNK, n)
            chunk = pos[start:end]              # (C, dim)

            # delta[c, j, d] = pos[start+c] - pos[j]
            delta = chunk[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (C, N, dim)

            dist2 = np.einsum("cnd,cnd->cn", delta, delta)           # (C, N)
            dist2 = np.maximum(dist2, 1e-12)                          # avoid /0
            dist = np.sqrt(dist2)                                     # (C, N)

            # zero out self-interaction
            idx = np.arange(start, end)
            dist2[np.arange(end - start), idx] = 1.0                 # dummy
            dist[np.arange(end - start), idx] = 1.0

            mag = self.k_repulsion / dist2                            # (C, N)
            # zero self
            mag[np.arange(end - start), idx] = 0.0

            # unit direction
            unit = delta / dist[:, :, np.newaxis]                     # (C, N, dim)

            # chunk contribution to forces[start:end]
            forces[start:end] += np.einsum("cn,cnd->cd", mag, unit)

            # Newton 3rd law: accumulate reaction onto all j
            # forces[j] -= sum_c(mag[c,j] * unit[c,j])
            reaction = -np.einsum("cn,cnd->nd", mag, unit)            # (N, dim)
            forces += reaction

        # Each pair was counted twice (once as i->j, once as j->i via reaction)
        # The reaction already negates Newton's 3rd, but the self-pair diagonal
        # contribution is zeroed so the only double-count is the cross-pairs.
        # Divide by 2 to correct.
        return forces / 2.0

    def _spring_forces(self) -> np.ndarray:
        """Vectorised spring (Hooke) forces for all synapses."""
        forces = np.zeros((self.n, self.dim), dtype=np.float64)
        if len(self._spring_i) == 0:
            return forces

        pi = self.positions[self._spring_i]   # (S, dim)
        pj = self.positions[self._spring_j]   # (S, dim)
        delta = pj - pi                        # (S, dim)
        dist = np.linalg.norm(delta, axis=1, keepdims=True).clip(min=1e-8)  # (S, 1)
        unit = delta / dist                    # (S, dim)
        extension = dist.squeeze(1) - self.rest_length    # (S,)
        force = unit * (self._spring_k * extension)[:, np.newaxis]  # (S, dim)

        np.add.at(forces, self._spring_i, force)
        np.add.at(forces, self._spring_j, -force)
        return forces

    # ── public interface ────────────────────────────────────────────────

    def step(self):
        if self.n == 0:
            return

        forces = self._repulsion_forces() + self._spring_forces()

        self.velocities = (self.velocities + self.dt * forces) * self.damping
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True).clip(min=1e-8)
        self.velocities = np.where(
            speed > self.max_step,
            self.velocities * (self.max_step / speed),
            self.velocities,
        )

        self.positions += self.dt * self.velocities
        self.positions -= self.positions.mean(axis=0, keepdims=True)

    def run(self, iterations: int = 200):
        for _ in range(iterations):
            self.step()

        return [
            {"id": i, "position": self.positions[i].round(4).tolist()}
            for i in range(self.n)
        ]
