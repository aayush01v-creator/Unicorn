import math
from typing import Any


class TorchSNN:
    def __init__(self, config: dict[str, Any], device):
        import torch

        self.torch = torch
        self.device = device
        self.neurons = config["neurons"]
        self.synapses = config["synapses"]
        self.steps = int(config.get("steps", 10))
        self.dt = float(config.get("dt", 1.0))
        if self.dt <= 0:
            raise ValueError("dt must be greater than zero")

        self.n = len(self.neurons)
        self.v = torch.tensor(
            [float(n.get("initial_voltage", 0.0)) for n in self.neurons],
            dtype=torch.float32,
            device=self.device,
        )
        self.thresholds = torch.tensor(
            [float(n.get("threshold", 1.0)) for n in self.neurons],
            dtype=torch.float32,
            device=self.device,
        )
        self.reset_potentials = torch.tensor(
            [float(n.get("reset_potential", 0.0)) for n in self.neurons],
            dtype=torch.float32,
            device=self.device,
        )

        default_tau = float(config.get("membrane_time_constant", 10.0))
        taus = [
            float(n.get("membrane_time_constant", default_tau)) for n in self.neurons
        ]
        if any(tau <= 0 for tau in taus):
            raise ValueError("membrane_time_constant must be greater than zero")
        self.decay = torch.tensor(
            [math.exp(-self.dt / tau) for tau in taus],
            dtype=torch.float32,
            device=self.device,
        )

        default_refractory = float(config.get("refractory_period", 0.0))
        refractory_periods = [
            float(n.get("refractory_period", default_refractory)) for n in self.neurons
        ]
        if any(period < 0 for period in refractory_periods):
            raise ValueError("refractory_period must be non-negative")
        self.refractory_steps = torch.tensor(
            [math.ceil(period / self.dt) for period in refractory_periods],
            dtype=torch.int32,
            device=self.device,
        )
        self.refractory_countdown = torch.zeros(
            self.n, dtype=torch.int32, device=self.device
        )

        configured_current = config.get("input_current", [0.0] * self.n)
        input_current = [float(current) for current in configured_current]
        if len(input_current) < self.n:
            input_current.extend([0.0] * (self.n - len(input_current)))
        elif len(input_current) > self.n:
            input_current = input_current[: self.n]
        self.input_current = torch.tensor(
            input_current, dtype=torch.float32, device=self.device
        )

        self.weights = torch.zeros(
            (self.n, self.n), dtype=torch.float32, device=self.device
        )
        for synapse in self.synapses:
            pre = int(synapse["from"])
            post = int(synapse["to"])
            if pre < 0 or post < 0 or pre >= self.n or post >= self.n:
                raise ValueError(
                    f"Synapse {pre} -> {post} references an invalid neuron index for network size {self.n}"
                )
            self.weights[pre, post] = float(synapse["weight"])

    def run(self):
        history = []
        spikes = self.torch.zeros(self.n, dtype=self.torch.float32, device=self.device)

        for step in range(self.steps):
            synaptic_current = spikes @ self.weights

            active_mask = self.refractory_countdown == 0
            integrated = (
                self.v * self.decay + (self.input_current + synaptic_current) * self.dt
            )
            self.v = self.torch.where(active_mask, integrated, self.reset_potentials)

            spikes = self.torch.where(
                active_mask & (self.v >= self.thresholds),
                self.torch.tensor(1.0, dtype=self.torch.float32, device=self.device),
                self.torch.tensor(0.0, dtype=self.torch.float32, device=self.device),
            )
            self.v = self.torch.where(spikes > 0, self.reset_potentials, self.v)
            self.refractory_countdown = self.torch.where(
                spikes > 0, self.refractory_steps, self.refractory_countdown
            )

            history.append(
                {
                    "step": step,
                    "time": round(step * self.dt, 10),
                    "voltages": self.v.detach().cpu().tolist(),
                    "spikes": [int(value) for value in spikes.detach().cpu().tolist()],
                    "refractory_remaining": (
                        self.refractory_countdown.to(self.torch.float32) * self.dt
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                }
            )

            decrement_mask = (spikes == 0) & (self.refractory_countdown > 0)
            self.refractory_countdown = self.torch.where(
                decrement_mask,
                self.refractory_countdown - 1,
                self.refractory_countdown,
            )

        return history
