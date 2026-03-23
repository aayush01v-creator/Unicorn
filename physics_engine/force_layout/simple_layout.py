import numpy as np

class ForceDirectedLayout:
    def __init__(self, config: dict, dim: int = 3, seed: int = 42):
        self.neurons = config["neurons"]
        self.synapses = config["synapses"]
        self.n = len(self.neurons)
        self.dim = dim

        rng = np.random.default_rng(seed)
        self.positions = rng.uniform(-1.0, 1.0, size=(self.n, self.dim))

    def step(self, k_repulsion=0.05, k_spring=0.08, rest_length=1.0, dt=0.05):
        forces = np.zeros_like(self.positions)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                delta = self.positions[i] - self.positions[j]
                dist = np.linalg.norm(delta) + 1e-6
                direction = delta / dist
                force_mag = k_repulsion / (dist * dist)
                force = direction * force_mag

                forces[i] += force
                forces[j] -= force

        for syn in self.synapses:
            i = syn["from"]
            j = syn["to"]
            w = float(syn.get("weight", 1.0))

            delta = self.positions[j] - self.positions[i]
            dist = np.linalg.norm(delta) + 1e-6
            direction = delta / dist
            force_mag = k_spring * (dist - rest_length) * w
            force = direction * force_mag

            forces[i] += force
            forces[j] -= force

        self.positions += dt * forces
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
