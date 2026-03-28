"""Batch mutation runner — apply sequences of network mutations from a YAML recipe.

Usage:
    python -m tools.mutate_network recipe.yaml

Recipe format (YAML):
    network: samples/net.json        # source network (read-only)
    output:  samples/net_evolved.json # destination (written after all steps)
    steps:
      - set-neuron:    {id: 0, activation_fn: softplus, bias: 0.2}
      - apply-profile: {name: stochastic, to: "range:10-20"}
      - mutate:        {select: all, scale: {gain: 1.5}}
      - set-schedule:  {id: 5, mode: pulse, amplitude: 2.0, period: 8}

If 'output' is omitted the source network is modified in-place.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


from tools.network_builder import (
    BUILTIN_PROFILES,
    _parse_selector,
    _parse_kv,
    ensure_input_current_shape,
    get_neuron_index,
    load_network,
    save_network,
    validate_network,
)


def _apply_step(network: dict[str, Any], step: dict[str, Any]) -> str:
    """Apply one mutation step in-place; return a log line."""
    if "set-neuron" in step:
        cfg = step["set-neuron"]
        nid = int(cfg["id"])
        idx = get_neuron_index(network, nid)
        neu = network["neurons"][idx]
        changed = []
        for k, v in cfg.items():
            if k == "id":
                continue
            if k == "input_current":
                ensure_input_current_shape(network)
                network["input_current"][idx] = float(v)
            else:
                neu[k] = v
            changed.append(f"{k}={v}")
        return f"set-neuron {nid}: {', '.join(changed)}"

    if "mutate" in step:
        cfg = step["mutate"]
        all_ids = [n["id"] for n in network["neurons"]]
        targets = _parse_selector(cfg.get("select", "all"), all_ids)
        ensure_input_current_shape(network)
        id_to_idx = {n["id"]: i for i, n in enumerate(network["neurons"])}

        def _kv(raw):
            if isinstance(raw, dict):
                return {k: float(v) for k, v in raw.items()}
            return _parse_kv(raw)

        sets   = _kv(cfg.get("set",   {}))
        adds   = _kv(cfg.get("add",   {}))
        scales = _kv(cfg.get("scale", {}))

        for nid in targets:
            idx = id_to_idx[nid]
            neu = network["neurons"][idx]
            for k, v in sets.items():
                if k == "input_current":
                    network["input_current"][idx] = v
                else:
                    neu[k] = v
            for k, delta in adds.items():
                if k == "input_current":
                    network["input_current"][idx] = float(network["input_current"][idx]) + delta
                else:
                    neu[k] = float(neu.get(k, 0.0)) + delta
            for k, factor in scales.items():
                if k == "input_current":
                    network["input_current"][idx] = float(network["input_current"][idx]) * factor
                else:
                    neu[k] = float(neu.get(k, 1.0)) * factor
        return f"mutate {len(targets)} neurons (select={cfg.get('select','all')})"

    if "apply-profile" in step:
        cfg = step["apply-profile"]
        pname = cfg.get("name") or cfg.get("profile")
        if pname in BUILTIN_PROFILES:
            profile = BUILTIN_PROFILES[pname]
        elif "from_file" in cfg:
            with open(cfg["from_file"]) as fh:
                profile = json.load(fh)[pname]
        else:
            raise ValueError(f"Unknown profile '{pname}'")
        all_ids = [n["id"] for n in network["neurons"]]
        targets = _parse_selector(cfg.get("to", "all"), all_ids)
        ensure_input_current_shape(network)
        id_to_idx = {n["id"]: i for i, n in enumerate(network["neurons"])}
        for nid in targets:
            idx = id_to_idx[nid]
            neu = network["neurons"][idx]
            for k, v in profile.items():
                if k == "input_current":
                    network["input_current"][idx] = v
                else:
                    neu[k] = v
        return f"apply-profile '{pname}' → {len(targets)} neurons"

    if "set-schedule" in step:
        cfg = step["set-schedule"]
        nid = int(cfg["id"])
        idx = get_neuron_index(network, nid)
        sched: dict[str, Any] = {
            "mode":      cfg.get("mode", "sine"),
            "amplitude": float(cfg.get("amplitude", 1.0)),
            "period":    float(cfg.get("period", 10.0)),
            "offset":    float(cfg.get("offset", 0.0)),
        }
        if "duration" in cfg:
            sched["duration"] = float(cfg["duration"])
        network["neurons"][idx]["input_schedule"] = sched
        return f"set-schedule neuron {nid}: {sched}"

    raise ValueError(f"Unknown step key: {list(step.keys())}")


def run_recipe(recipe_path: Path) -> None:
    if not _HAS_YAML:
        # Fall back to JSON if PyYAML is not installed
        with recipe_path.open() as fh:
            recipe = json.load(fh)
    else:
        with recipe_path.open() as fh:
            recipe = yaml.safe_load(fh)

    src  = Path(recipe["network"])
    dst  = Path(recipe.get("output", recipe["network"]))
    steps: list[dict] = recipe.get("steps", [])

    # Work on a copy so we never corrupt the original mid-run
    network = load_network(src)

    print(f"Recipe: {recipe_path}")
    print(f"Source: {src}  →  Output: {dst}")
    print(f"Steps : {len(steps)}")
    print("-" * 50)

    for i, step in enumerate(steps, 1):
        log = _apply_step(network, step)
        print(f"  [{i:02d}] {log}")

    validate_network(network)
    save_network(dst, network)
    print("-" * 50)
    print(f"Saved → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a YAML/JSON mutation recipe to a Unicorn network."
    )
    parser.add_argument("recipe", help="Path to the YAML or JSON recipe file.")
    args = parser.parse_args()
    run_recipe(Path(args.recipe))


if __name__ == "__main__":
    main()
