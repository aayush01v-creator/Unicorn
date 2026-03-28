import importlib
import importlib.util
from typing import Any

from backend.neuron_sim.simple_snn import SimpleSNN


SUPPORTED_SIMULATORS = {"auto", "simple", "snntorch", "spikingjelly"}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def select_simulator(config: dict[str, Any]) -> str:
    simulator = str(config.get("simulator", "auto")).lower()
    if simulator not in SUPPORTED_SIMULATORS:
        raise ValueError(
            f"Unsupported simulator '{simulator}'. Expected one of: {sorted(SUPPORTED_SIMULATORS)}"
        )

    if simulator == "auto":
        if _module_available("snntorch"):
            return "snntorch"
        if _module_available("spikingjelly"):
            return "spikingjelly"
        return "simple"
    return simulator


def _run_torch_backend(config: dict[str, Any], framework_name: str):
    import torch

    from backend.neuron_sim.torch_snn import TorchSNN

    # Require the selected framework package explicitly so callers know
    # whether they are running with snnTorch or SpikingJelly support installed.
    importlib.import_module(framework_name)

    device = torch.device("cpu")
    return TorchSNN(config, device=device).run()


def run_simulation(config: dict[str, Any]):
    simulator = select_simulator(config)
    if simulator == "simple":
        return SimpleSNN(config).run()
    return _run_torch_backend(config, simulator)
