import numpy as np

class SimpleSNN:
    def __init__(self, config: dict):
        self.neurons = config["neurons"]
        self.synapses = config["synapses"]
        self.steps = config.get("steps", 10)

        self.n = len(self.neurons)
        self.v = np.zeros(self.n, dtype=float)
        self.thresholds = np.array([n.get("threshold", 1.0) for n in self.neurons], dtype=float)
        self.input_current = np.array(config.get("input_current", [0.0] * self.n), dtype=float)

        self.weights = np.zeros((self.n, self.n), dtype=float)
        for s in self.synapses:
            self.weights[s["from"], s["to"]] = s["weight"]

    def run(self):
        history = []

        spikes = np.zeros(self.n, dtype=float)

        for t in range(self.steps):
            self.v += self.input_current
            self.v += spikes @ self.weights

            spikes = (self.v >= self.thresholds).astype(float)

            self.v[spikes == 1] = 0.0

            history.append({
                "step": t,
                "voltages": self.v.copy().tolist(),
                "spikes": spikes.astype(int).tolist()
            })

        return history
