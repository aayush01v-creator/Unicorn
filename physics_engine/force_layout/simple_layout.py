import numpy as np


class ForceDirectedLayout:
    """Physics-driven force-directed layout for SNN topology visualization.

    Neurons are modeled as charged particles (Coulomb repulsion) and synapses
    as springs (Hooke attraction), then integrated with simple velocity damping.
    """

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

    def step(self):
        if self.n == 0:
            return

        forces = np.zeros_like(self.positions)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                delta = self.positions[i] - self.positions[j]
                dist = np.linalg.norm(delta)
                if dist < 1e-6:
                    direction = np.zeros(self.dim)
                    direction[0] = 1.0
                    dist = 1e-6
                else:
                    direction = delta / dist

                force_mag = self.k_repulsion / (dist * dist)
                force = direction * force_mag

                forces[i] += force
                forces[j] -= force

        for syn in self.synapses:
            i = syn["from"]
            j = syn["to"]
            spring_strength = self.k_spring * abs(float(syn.get("weight", 1.0)))

            delta = self.positions[j] - self.positions[i]
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                continue

            direction = delta / dist
            extension = dist - self.rest_length
            force = direction * (spring_strength * extension)

            forces[i] += force
            forces[j] -= force

        self.velocities = (self.velocities + self.dt * forces) * self.damping
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        speed = np.maximum(speed, 1e-8)
        capped_velocity = np.where(
            speed > self.max_step,
            self.velocities * (self.max_step / speed),
            self.velocities,
        )

        self.positions += self.dt * capped_velocity
        self.positions -= self.positions.mean(axis=0, keepdims=True)

    def run(self, iterations=200):
        for _ in range(iterations):
            self.step()

        return [
            {
                "id": i,
                "position": self.positions[i].round(4).tolist()
            }
            for i in range(self.n)
        ]
