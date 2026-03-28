"""Fast NumPy-vectorised leaky integrate-and-fire simulator — extended.

New per-neuron properties (all optional, default = off):
  activation_fn   : "lif" | "relu" | "softplus" | "tanh" | "sigmoid" | "rbf"
  bias            : constant additive drive (float, default 0.0)
  gain            : scales incoming synaptic current (float, default 1.0)
  noise_std       : Gaussian noise std injected each step (float, default 0.0)
  dropout_prob    : per-step silencing probability, 0 = off (float, default 0.0)
  adaptation_rate : threshold rise per spike (float, default 0.0)
  adaptation_decay: per-step decay of adaptation variable (float, default 1.0)
  v_rest          : resting potential the membrane leaks toward (float, default 0.0)
  rbf_centre      : centre voltage for RBF activation (float, default 0.5)
  rbf_sigma       : width of RBF Gaussian (float, default 0.3)
  input_schedule  : dict with keys mode/amplitude/period/offset/duration
                    overrides input_current per step when present

Synaptic plasticity (optional, per synapse):
  plasticity_rule : "hebbian" | "oja" | None
  plasticity_lr   : learning rate (float, default 0.01)

All computation is vectorised with NumPy — no Python loops per step.
"""

import math

import numpy as np


# ── Activation function helpers (vectorised) ─────────────────────────────────

def _apply_activation(v: np.ndarray, fn_ids: np.ndarray,
                      rbf_centres: np.ndarray, rbf_sigmas: np.ndarray,
                      thresholds: np.ndarray) -> np.ndarray:
    """Return post-activation output for each neuron given membrane voltage v.

    For LIF neurons the returned value is a spike-binary (0/1) decided
    by the threshold comparison done in the main loop.  For non-spiking
    activation types (relu/softplus/tanh/sigmoid/rbf) the returned value
    IS the output — spikes are not used.

    fn_ids: integer array (N,)
        0 = lif, 1 = relu, 2 = softplus, 3 = tanh, 4 = sigmoid, 5 = rbf
    """
    # LIF: output controlled externally — return v unchanged
    out = v.copy()
    relu    = fn_ids == 1
    softpl  = fn_ids == 2
    tanh_m  = fn_ids == 3
    sig     = fn_ids == 4
    rbf_m   = fn_ids == 5

    if relu.any():
        out = np.where(relu, np.maximum(0.0, v), out)
    if softpl.any():
        # log1p(exp(v)), clamped for stability
        out = np.where(softpl, np.log1p(np.exp(np.clip(v, -80, 80))), out)
    if tanh_m.any():
        out = np.where(tanh_m, np.tanh(v), out)
    if sig.any():
        out = np.where(sig, 1.0 / (1.0 + np.exp(-np.clip(v, -80, 80))), out)
    if rbf_m.any():
        rbf_val = np.exp(-0.5 * ((v - rbf_centres) / np.maximum(rbf_sigmas, 1e-8)) ** 2)
        out = np.where(rbf_m, rbf_val, out)
    return out


_ACTIVATION_IDS = {"lif": 0, "relu": 1, "softplus": 2, "tanh": 3, "sigmoid": 4, "rbf": 5}
_SCHEDULE_MODES = {"constant", "ramp", "pulse", "sine"}


def _schedule_current(sched: dict, step: int, dt: float) -> float:
    """Evaluate a per-neuron input schedule at a given simulation step."""
    t = step * dt
    mode = sched.get("mode", "constant")
    amp  = float(sched.get("amplitude", 0.0))
    per  = float(sched.get("period", 10.0))
    off  = float(sched.get("offset", 0.0))
    dur  = sched.get("duration")

    if mode == "constant":
        return amp + off
    if mode == "ramp":
        if dur is not None and t > float(dur) * dt:
            return amp + off
        frac = min(t / max(float(dur or per) * dt, 1e-9), 1.0)
        return amp * frac + off
    if mode == "pulse":
        return amp + off if (step % max(int(per), 1)) < max(int(per) // 2, 1) else off
    if mode == "sine":
        return amp * math.sin(2 * math.pi * t / max(per * dt, 1e-9)) + off
    return off


class SimpleSNN:
    def __init__(self, config: dict):
        self.neurons  = config["neurons"]
        self.synapses = config["synapses"]
        self.steps    = config.get("steps", 10)
        self.dt       = float(config.get("dt", 1.0))
        if self.dt <= 0:
            raise ValueError("dt must be greater than zero")

        n = len(self.neurons)
        self.n = n

        # ── Core LIF parameters ──────────────────────────────────────────────
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
        self.v_rest = np.array(
            [float(neu.get("v_rest", 0.0)) for neu in self.neurons],
            dtype=np.float64,
        )

        default_tau = float(config.get("membrane_time_constant", 10.0))
        taus = np.array(
            [float(neu.get("membrane_time_constant", default_tau)) for neu in self.neurons],
            dtype=np.float64,
        )
        if np.any(taus <= 0):
            raise ValueError("membrane_time_constant must be greater than zero")
        self.decay = np.exp(-self.dt / taus)

        default_refractory = float(config.get("refractory_period", 0.0))
        refractory_periods = np.array(
            [float(neu.get("refractory_period", default_refractory)) for neu in self.neurons],
            dtype=np.float64,
        )
        if np.any(refractory_periods < 0):
            raise ValueError("refractory_period must be non-negative")
        self.refractory_steps     = np.ceil(refractory_periods / self.dt).astype(np.int32)
        self.refractory_countdown = np.zeros(n, dtype=np.int32)

        configured_current = config.get("input_current", [0.0] * n)
        ic = np.array([float(c) for c in configured_current], dtype=np.float64)
        if len(ic) < n:
            ic = np.concatenate([ic, np.zeros(n - len(ic))])
        elif len(ic) > n:
            ic = ic[:n]
        self.input_current = ic

        # ── New extended properties ──────────────────────────────────────────
        self.bias = np.array(
            [float(neu.get("bias", 0.0)) for neu in self.neurons], dtype=np.float64
        )
        self.gain = np.array(
            [float(neu.get("gain", 1.0)) for neu in self.neurons], dtype=np.float64
        )
        self.noise_std = np.array(
            [float(neu.get("noise_std", 0.0)) for neu in self.neurons], dtype=np.float64
        )
        self.dropout_prob = np.array(
            [float(neu.get("dropout_prob", 0.0)) for neu in self.neurons], dtype=np.float64
        )
        self.adaptation_rate = np.array(
            [float(neu.get("adaptation_rate", 0.0)) for neu in self.neurons], dtype=np.float64
        )
        self.adaptation_decay = np.array(
            [float(neu.get("adaptation_decay", 1.0)) for neu in self.neurons], dtype=np.float64
        )
        self.adaptation        = np.zeros(n, dtype=np.float64)  # running adaptation state
        self.effective_thresh  = self.thresholds.copy()          # thresholds + adaptation

        raw_fns = [neu.get("activation_fn", "lif") for neu in self.neurons]
        self.fn_ids = np.array(
            [_ACTIVATION_IDS.get(fn, 0) for fn in raw_fns], dtype=np.int32
        )
        self.rbf_centres = np.array(
            [float(neu.get("rbf_centre", 0.5)) for neu in self.neurons], dtype=np.float64
        )
        self.rbf_sigmas = np.array(
            [float(neu.get("rbf_sigma", 0.3)) for neu in self.neurons], dtype=np.float64
        )

        # Schedules: list of (neuron_index, schedule_dict | None)
        self._schedules: list[tuple[int, dict]] = []
        for idx, neu in enumerate(self.neurons):
            sched = neu.get("input_schedule")
            if sched and isinstance(sched, dict):
                self._schedules.append((idx, sched))

        # ── Weight matrix / sparse structures ────────────────────────────────
        if n <= 8_000:
            W = np.zeros((n, n), dtype=np.float32)
            for syn in self.synapses:
                pre, post = int(syn["from"]), int(syn["to"])
                if not (0 <= pre < n and 0 <= post < n):
                    raise ValueError(
                        f"Synapse {pre} -> {post} out of range for N={n}"
                    )
                W[pre, post] = float(syn["weight"])
            self._W      = W
            self._sparse = False
        else:
            pres, posts, weights = [], [], []
            for syn in self.synapses:
                pre, post = int(syn["from"]), int(syn["to"])
                if not (0 <= pre < n and 0 <= post < n):
                    raise ValueError(
                        f"Synapse {pre} -> {post} out of range for N={n}"
                    )
                pres.append(pre); posts.append(post); weights.append(float(syn["weight"]))
            self._pre    = np.array(pres,    dtype=np.int32)
            self._post   = np.array(posts,   dtype=np.int32)
            self._wvals  = np.array(weights, dtype=np.float64)
            self._sparse = True

        # ── Plasticity ───────────────────────────────────────────────────────
        self._plasticity: list[tuple[int, int, str, float]] = []  # (pre, post, rule, lr)
        if not self._sparse:
            for syn in self.synapses:
                rule = syn.get("plasticity_rule")
                if rule in ("hebbian", "oja"):
                    self._plasticity.append((
                        int(syn["from"]), int(syn["to"]),
                        rule, float(syn.get("plasticity_lr", 0.01)),
                    ))

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _synaptic_current(self, spikes: np.ndarray) -> np.ndarray:
        if not self._sparse:
            return spikes.astype(np.float32) @ self._W
        firing = np.where(spikes)[0]
        if len(firing) == 0:
            return np.zeros(self.n, dtype=np.float64)
        keep   = np.isin(self._pre, firing)
        result = np.zeros(self.n, dtype=np.float64)
        np.add.at(result, self._post[keep], self._wvals[keep])
        return result

    def _apply_plasticity(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """Update weight matrix in-place (dense path only)."""
        for pre, post, rule, lr in self._plasticity:
            ps = pre_spikes[pre]; qs = post_spikes[post]
            if rule == "hebbian":
                self._W[pre, post] += lr * ps * qs
            elif rule == "oja":
                w = float(self._W[pre, post])
                self._W[pre, post] += lr * (ps * qs - qs * qs * w)

    # ── Public interface ─────────────────────────────────────────────────────

    def run(self) -> list[dict]:
        history: list[dict] = []
        spikes = np.zeros(self.n, dtype=np.float64)

        for t in range(self.steps):
            # ── 1. Build base input current (with schedule overrides) ────────
            ic = self.input_current.copy()
            for idx, sched in self._schedules:
                ic[idx] = _schedule_current(sched, t, self.dt)

            # ── 2. Synaptic current + gain + noise ───────────────────────────
            syn = self._synaptic_current(spikes)
            syn = syn * self.gain
            if np.any(self.noise_std > 0):
                syn += np.random.normal(0.0, self.noise_std)

            # ── 3. Dropout mask (silence neuron for this step) ───────────────
            if np.any(self.dropout_prob > 0):
                alive = np.random.random(self.n) >= self.dropout_prob
            else:
                alive = np.ones(self.n, dtype=bool)

            # ── 4. Refractory gate ───────────────────────────────────────────
            active = alive & (self.refractory_countdown == 0)

            # ── 5. Voltage update (leak toward v_rest) ───────────────────────
            drive = (ic + syn + self.bias) * self.dt
            self.v = np.where(
                active,
                self.v * self.decay + self.v_rest * (1.0 - self.decay) + drive,
                self.reset_potentials,
            )

            # ── 6. Activation function output ────────────────────────────────
            act_out = _apply_activation(
                self.v, self.fn_ids, self.rbf_centres, self.rbf_sigmas,
                self.effective_thresh,
            )

            # ── 7. Spike detection (LIF: threshold compare; others: pass-through) ──
            lif_mask    = self.fn_ids == 0
            non_lif     = ~lif_mask
            lif_spikes  = active & lif_mask & (self.v >= self.effective_thresh)

            # Non-LIF neurons emit their activation output as a continuous value
            # (clipped to [0,1] for compatibility with the spike history format).
            non_lif_out = np.where(non_lif & active, np.clip(act_out, 0.0, 1.0), 0.0)
            spikes      = np.where(lif_spikes, 1.0, non_lif_out)

            # ── 8. Reset spiking LIF neurons ─────────────────────────────────
            self.v = np.where(lif_spikes, self.reset_potentials, self.v)

            # ── 9. Spike-frequency adaptation ────────────────────────────────
            fired = spikes > 0
            self.adaptation      = self.adaptation * self.adaptation_decay
            self.adaptation     += self.adaptation_rate * fired
            self.effective_thresh = self.thresholds + self.adaptation

            # ── 10. Refractory countdown ─────────────────────────────────────
            self.refractory_countdown = np.where(
                lif_spikes,
                self.refractory_steps,
                np.maximum(0, self.refractory_countdown - 1),
            )

            # ── 11. Synaptic plasticity ───────────────────────────────────────
            if self._plasticity:
                self._apply_plasticity(spikes, spikes)

            history.append({
                "step": t,
                "time": round(t * self.dt, 10),
                "voltages":             self.v.tolist(),
                "spikes":               spikes.astype(int).tolist(),
                "refractory_remaining": (self.refractory_countdown * self.dt).tolist(),
                "adaptation":           self.adaptation.tolist(),
            })

        return history
