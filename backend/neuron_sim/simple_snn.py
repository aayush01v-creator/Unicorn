import math


class SimpleSNN:
    def __init__(self, config: dict):
        self.neurons = config["neurons"]
        self.synapses = config["synapses"]
        self.steps = config.get("steps", 10)
        self.dt = float(config.get("dt", 1.0))
        if self.dt <= 0:
            raise ValueError("dt must be greater than zero")

        self.n = len(self.neurons)
        self.v = [float(n.get("initial_voltage", 0.0)) for n in self.neurons]
        self.thresholds = [float(n.get("threshold", 1.0)) for n in self.neurons]
        self.reset_potentials = [float(n.get("reset_potential", 0.0)) for n in self.neurons]

        default_tau = float(config.get("membrane_time_constant", 10.0))
        self.membrane_time_constants = [
            float(n.get("membrane_time_constant", default_tau)) for n in self.neurons
        ]
        if any(tau <= 0 for tau in self.membrane_time_constants):
            raise ValueError("membrane_time_constant must be greater than zero")
        self.decay = [math.exp(-self.dt / tau) for tau in self.membrane_time_constants]

        default_refractory = float(config.get("refractory_period", 0.0))
        self.refractory_periods = [
            float(n.get("refractory_period", default_refractory)) for n in self.neurons
        ]
        if any(period < 0 for period in self.refractory_periods):
            raise ValueError("refractory_period must be non-negative")
        self.refractory_steps = [math.ceil(period / self.dt) for period in self.refractory_periods]
        self.refractory_countdown = [0] * self.n

        configured_current = config.get("input_current", [0.0] * self.n)
        self.input_current = [float(current) for current in configured_current]

        self.weights = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        for synapse in self.synapses:
            self.weights[synapse["from"]][synapse["to"]] = float(synapse["weight"])

    def run(self):
        history = []
        spikes = [0.0] * self.n

        for t in range(self.steps):
            synaptic_current = [0.0] * self.n
            for pre_idx, pre_spike in enumerate(spikes):
                if pre_spike:
                    for post_idx in range(self.n):
                        synaptic_current[post_idx] += pre_spike * self.weights[pre_idx][post_idx]

            active_mask = [countdown == 0 for countdown in self.refractory_countdown]
            for idx, is_active in enumerate(active_mask):
                if is_active:
                    self.v[idx] = (
                        self.v[idx] * self.decay[idx]
                        + (self.input_current[idx] + synaptic_current[idx]) * self.dt
                    )
                else:
                    self.v[idx] = self.reset_potentials[idx]

            spikes = [0.0] * self.n
            for idx, is_active in enumerate(active_mask):
                if is_active and self.v[idx] >= self.thresholds[idx]:
                    spikes[idx] = 1.0
                    self.v[idx] = self.reset_potentials[idx]
                    self.refractory_countdown[idx] = self.refractory_steps[idx]

            history.append(
                {
                    "step": t,
                    "time": round(t * self.dt, 10),
                    "voltages": list(self.v),
                    "spikes": [int(spike) for spike in spikes],
                    "refractory_remaining": [countdown * self.dt for countdown in self.refractory_countdown],
                }
            )

            self.refractory_countdown = [
                max(0, countdown - 1) if not spikes[idx] and countdown > 0 else countdown
                for idx, countdown in enumerate(self.refractory_countdown)
            ]

        return history
