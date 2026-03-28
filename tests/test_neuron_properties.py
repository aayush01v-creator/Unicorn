"""Tests for extended neuron properties — activation functions, bias, gain,
noise, dropout, adaptation, schedules, and CLI subcommands."""

import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.neuron_sim.simple_snn import SimpleSNN, _schedule_current
from tools.network_builder import (
    BUILTIN_PROFILES,
    _parse_kv,
    _parse_selector,
    load_network,
    save_network,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_network(n: int = 2, **neuron_kwargs) -> dict:
    """Return a minimal network dict with n identical neurons."""
    neurons = [
        {"id": i, "threshold": 1.0, "membrane_time_constant": 10.0, **neuron_kwargs}
        for i in range(n)
    ]
    return {
        "neurons": neurons,
        "synapses": [],
        "input_current": [0.0] * n,
        "steps": 5,
        "dt": 1.0,
        "refractory_period": 0.0,
    }


def _run(config: dict) -> list[dict]:
    return SimpleSNN(config).run()


# ── Activation functions ──────────────────────────────────────────────────────

def test_lif_fires_when_above_threshold():
    cfg = _minimal_network(1, threshold=0.5)
    cfg["input_current"] = [1.0]
    history = _run(cfg)
    spikes = [h["spikes"][0] for h in history]
    assert any(s > 0 for s in spikes), "LIF neuron should fire with sufficient drive"


def test_relu_output_nonnegative():
    cfg = _minimal_network(1, activation_fn="relu", threshold=99.0)
    cfg["input_current"] = [0.5]
    for step in _run(cfg):
        assert step["spikes"][0] >= 0, "ReLU output must be non-negative"


def test_softplus_gt_zero():
    cfg = _minimal_network(1, activation_fn="softplus", threshold=99.0)
    cfg["input_current"] = [0.5]
    for step in _run(cfg):
        assert step["spikes"][0] >= 0, "Softplus must be non-negative"


def test_tanh_bounded():
    cfg = _minimal_network(1, activation_fn="tanh", threshold=99.0)
    cfg["input_current"] = [1.0]
    for step in _run(cfg):
        assert -1.0 <= step["spikes"][0] <= 1.0, "tanh output must be in [-1, 1]"


def test_sigmoid_bounded():
    cfg = _minimal_network(1, activation_fn="sigmoid", threshold=99.0)
    cfg["input_current"] = [2.0]
    for step in _run(cfg):
        assert 0.0 <= step["spikes"][0] <= 1.0, "Sigmoid output must be in [0, 1]"


def test_rbf_peaks_near_centre():
    """RBF output should be highest when membrane voltage is near the centre.

    We compare the first-step output directly: neuron initialised at 0.5
    (centre) vs one initialised far away at 5.0.  The activation is applied
    before any spike reset so the initial_voltage is visible.
    """
    from backend.neuron_sim.simple_snn import _apply_activation
    import numpy as np
    centres = np.array([0.5])
    sigmas  = np.array([0.2])
    thresholds = np.array([99.0])
    fn_ids = np.array([5])  # rbf

    near = _apply_activation(np.array([0.5]), fn_ids, centres, sigmas, thresholds)
    far  = _apply_activation(np.array([5.0]), fn_ids, centres, sigmas, thresholds)
    assert near[0] > far[0], "RBF should be larger when voltage is near centre"


# ── Bias and gain ─────────────────────────────────────────────────────────────

def test_bias_increases_drive():
    base  = _minimal_network(1)
    biased = _minimal_network(1, bias=0.5)
    base["input_current"]  = [0.0]
    biased["input_current"] = [0.0]
    base_v   = _run(base)[0]["voltages"][0]
    biased_v = _run(biased)[0]["voltages"][0]
    assert biased_v > base_v, "Bias should push voltage higher"


def test_gain_scales_synaptic_input():
    """With gain=2, neuron 1 should accumulate more voltage from the same synapse."""
    def _net_with_gain(g):
        cfg = {
            "neurons":  [
                # Neuron 0: very low threshold so it fires immediately
                {"id": 0, "threshold": 0.1, "membrane_time_constant": 5.0},
                {"id": 1, "threshold": 99.0, "membrane_time_constant": 10.0, "gain": g},
            ],
            "synapses": [{"from": 0, "to": 1, "weight": 0.5}],
            "input_current": [2.0, 0.0],
            "steps": 5,
            "dt": 1.0,
            "refractory_period": 0.0,
        }
        return cfg

    v_g1 = max(h["voltages"][1] for h in _run(_net_with_gain(1.0)))
    v_g2 = max(h["voltages"][1] for h in _run(_net_with_gain(2.0)))
    assert v_g2 > v_g1, "Higher gain should produce larger post-synaptic voltage"


# ── Noise ─────────────────────────────────────────────────────────────────────

def test_noise_produces_variance():
    """Running the same config twice with noise_std>0 should give different voltages."""
    cfg = _minimal_network(1, noise_std=0.5)
    cfg["input_current"] = [0.3]
    cfg["steps"] = 20
    v1 = [h["voltages"][0] for h in _run(cfg)]
    v2 = [h["voltages"][0] for h in _run(cfg)]
    assert v1 != v2, "Noisy neurons should produce different runs"


# ── Dropout ───────────────────────────────────────────────────────────────────

def test_silent_neuron_never_fires():
    """dropout_prob=1.0 should silence the neuron completely."""
    cfg = _minimal_network(1, dropout_prob=1.0)
    cfg["input_current"] = [5.0]
    cfg["steps"] = 20
    for step in _run(cfg):
        assert step["spikes"][0] == 0, "Silent neuron must not fire"


def test_dropout_zero_behaves_normally():
    cfg = _minimal_network(1, dropout_prob=0.0, threshold=0.5)
    cfg["input_current"] = [1.0]
    cfg["steps"] = 5
    spikes = [h["spikes"][0] for h in _run(cfg)]
    assert any(s > 0 for s in spikes), "dropout=0 should behave like normal LIF"


# ── v_rest ────────────────────────────────────────────────────────────────────

def test_v_rest_sets_leak_target():
    """Neuron with v_rest=0.5 should settle above 0 at equilibrium with no drive."""
    cfg = _minimal_network(1, v_rest=0.5)
    cfg["input_current"] = [0.0]
    cfg["steps"] = 50
    history = _run(cfg)
    final_v = history[-1]["voltages"][0]
    assert final_v > 0.0, "Membrane should leak toward v_rest=0.5, not 0"


# ── Spike-frequency adaptation ────────────────────────────────────────────────

def test_adaptation_reduces_late_spikes():
    """With strong adaptation, firing rate should decrease over time."""
    cfg = _minimal_network(1, adaptation_rate=0.5, adaptation_decay=0.8, threshold=0.5)
    cfg["input_current"] = [2.0]
    cfg["steps"] = 30
    history = _run(cfg)
    spikes = [h["spikes"][0] for h in history]
    early = sum(spikes[:10])
    late  = sum(spikes[20:30])
    assert early >= late, "Adaptation should reduce firing rate over time"


def test_adaptation_history_in_output():
    cfg = _minimal_network(1, adaptation_rate=0.2)
    cfg["input_current"] = [2.0]
    for step in _run(cfg):
        assert "adaptation" in step, "Adaptation state must be in history output"
        assert len(step["adaptation"]) == 1


# ── Input schedules ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,expected_min,expected_max", [
    ("constant", 1.0,  1.0),
    ("sine",    -1.0,  1.0),
    ("pulse",    0.0,  1.0),
    ("ramp",     0.0,  1.0),
])
def test_schedule_modes(mode, expected_min, expected_max):
    sched = {"mode": mode, "amplitude": 1.0, "period": 10.0, "offset": 0.0}
    vals = [_schedule_current(sched, t, 1.0) for t in range(20)]
    assert min(vals) >= expected_min - 1e-9
    assert max(vals) <= expected_max + 1e-9


def test_schedule_overrides_input_current():
    cfg = _minimal_network(1, threshold=99.0)
    cfg["neurons"][0]["input_schedule"] = {
        "mode": "constant", "amplitude": 0.0, "period": 10.0, "offset": 0.0,
    }
    cfg["input_current"] = [5.0]  # should be ignored for neuron 0
    cfg["steps"] = 5
    # Voltage should stay near 0 despite input_current=5.0
    history = _run(cfg)
    for step in history:
        assert abs(step["voltages"][0]) < 0.1, \
            "Schedule amplitude=0 should override large input_current"


# ── CLI helpers ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("selector,n,expected", [
    ("all",       5, [0, 1, 2, 3, 4]),
    ("range:1-3", 5, [1, 2, 3]),
    ("every:2",   6, [0, 2, 4]),
    ("0,3",       5, [0, 3]),
])
def test_parse_selector(selector, n, expected):
    ids = list(range(n))
    assert _parse_selector(selector, ids) == expected


def test_parse_kv():
    result = _parse_kv("noise_std=0.05,gain=2.0")
    assert result == {"noise_std": 0.05, "gain": 2.0}


def test_parse_kv_none():
    assert _parse_kv(None) == {}


# ── CLI subcommand integration ────────────────────────────────────────────────

def _make_tmp_net(n: int = 3) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "net.json"
    cfg = _minimal_network(n)
    save_network(tmp, cfg)
    return tmp


def test_cmd_set_neuron_cli(tmp_path):
    from tools.network_builder import cmd_set_neuron
    import argparse
    net = _make_tmp_net(3)
    args = argparse.Namespace(
        path=str(net), id=1,
        threshold=0.5, membrane_time_constant=5.0, refractory_period=None,
        reset_potential=None, v_rest=0.1, input_current=1.2,
        bias=0.3, gain=2.0, noise_std=0.05, dropout_prob=0.1,
        adaptation_rate=0.2, adaptation_decay=0.9,
        activation_fn="relu", rbf_centre=None, rbf_sigma=None,
    )
    msg = cmd_set_neuron(args)
    assert "1" in msg
    saved = load_network(net)
    neu = next(n for n in saved["neurons"] if n["id"] == 1)
    assert neu["threshold"] == 0.5
    assert neu["activation_fn"] == "relu"
    assert neu["gain"] == 2.0
    assert saved["input_current"][1] == pytest.approx(1.2)


def test_cmd_mutate_set(tmp_path):
    from tools.network_builder import cmd_mutate
    import argparse
    net = _make_tmp_net(4)
    args = argparse.Namespace(
        path=str(net), select="range:0-1",
        set="noise_std=0.08,bias=0.1", add=None, scale=None,
    )
    msg = cmd_mutate(args)
    assert "2 neurons" in msg
    saved = load_network(net)
    for nid in [0, 1]:
        neu = next(n for n in saved["neurons"] if n["id"] == nid)
        assert neu["noise_std"] == pytest.approx(0.08)
        assert neu["bias"] == pytest.approx(0.1)
    # neuron 2 should be untouched
    neu2 = next(n for n in saved["neurons"] if n["id"] == 2)
    assert neu2.get("noise_std", 0.0) == 0.0


def test_cmd_mutate_scale(tmp_path):
    from tools.network_builder import cmd_mutate
    import argparse
    net = _make_tmp_net(3)
    # First set gain=1.0 on all neurons explicitly
    net_data = load_network(net)
    for n in net_data["neurons"]:
        n["gain"] = 1.0
    save_network(net, net_data)

    args = argparse.Namespace(
        path=str(net), select="all", set=None, add=None, scale="gain=3.0",
    )
    cmd_mutate(args)
    saved = load_network(net)
    for neu in saved["neurons"]:
        assert neu["gain"] == pytest.approx(3.0)


def test_cmd_apply_profile(tmp_path):
    from tools.network_builder import cmd_apply_profile
    import argparse
    net = _make_tmp_net(5)
    args = argparse.Namespace(
        path=str(net), profile="stochastic", to="range:0-2",
        from_file=None, name=None,
    )
    msg = cmd_apply_profile(args)
    assert "stochastic" in msg
    saved = load_network(net)
    for nid in [0, 1, 2]:
        neu = next(n for n in saved["neurons"] if n["id"] == nid)
        assert neu.get("activation_fn") == "sigmoid"
        assert neu.get("dropout_prob") == pytest.approx(0.25)
    # neuron 4 should be unchanged
    neu4 = next(n for n in saved["neurons"] if n["id"] == 4)
    assert neu4.get("activation_fn", "lif") == "lif"


def test_cmd_set_schedule(tmp_path):
    from tools.network_builder import cmd_set_schedule
    import argparse
    net = _make_tmp_net(2)
    args = argparse.Namespace(
        path=str(net), id=0, mode="sine", amplitude=1.5, period=8.0,
        offset=0.2, duration=None,
    )
    msg = cmd_set_schedule(args)
    assert "sine" in msg
    saved = load_network(net)
    sched = saved["neurons"][0]["input_schedule"]
    assert sched["mode"] == "sine"
    assert sched["amplitude"] == pytest.approx(1.5)


def test_cmd_props(tmp_path):
    from tools.network_builder import cmd_props
    import argparse
    net = _make_tmp_net(3)
    net_data = load_network(net)
    net_data["neurons"][1]["bias"] = 0.42
    net_data["neurons"][1]["activation_fn"] = "softplus"
    save_network(net, net_data)

    args = argparse.Namespace(path=str(net), ids=[1])
    output = cmd_props(args)
    assert "bias=0.4200" in output
    assert "softplus" in output
    assert "Neuron   1" in output


# ── Builtin profiles completeness ─────────────────────────────────────────────

def test_all_builtin_profiles_are_dicts():
    for name, profile in BUILTIN_PROFILES.items():
        assert isinstance(profile, dict), f"Profile '{name}' must be a dict"
        assert len(profile) > 0, f"Profile '{name}' must not be empty"


# ── Plasticity ────────────────────────────────────────────────────────────────

def test_hebbian_updates_weight():
    """Hebbian rule should change the weight matrix after correlated firing."""
    cfg = {
        "neurons": [
            {"id": 0, "threshold": 0.3, "membrane_time_constant": 5.0},
            {"id": 1, "threshold": 0.3, "membrane_time_constant": 5.0},
        ],
        "synapses": [
            {"from": 0, "to": 1, "weight": 0.5, "plasticity_rule": "hebbian", "plasticity_lr": 0.1}
        ],
        "input_current": [1.0, 1.0],
        "steps": 10,
        "dt": 1.0,
        "refractory_period": 0.0,
    }
    snn = SimpleSNN(cfg)
    initial_w = float(snn._W[0, 1])
    snn.run()
    final_w = float(snn._W[0, 1])
    assert final_w > initial_w, "Hebbian rule should strengthen correlated synapse"
