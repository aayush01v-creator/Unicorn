import argparse
import json
import math
import random as _random
from pathlib import Path
from typing import Any


DEFAULT_NETWORK = {
    "neurons": [],
    "synapses": [],
    "input_current": [],
    "steps": 10,
    "dt": 1.0,
    "refractory_period": 0.0,
}


def load_network(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r") as f:
            return json.load(f)
    return json.loads(json.dumps(DEFAULT_NETWORK))


def save_network(path: Path, network: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(network, f, indent=2)
        f.write("\n")


def ensure_input_current_shape(network: dict[str, Any]) -> None:
    neuron_count = len(network["neurons"])
    currents = [float(value) for value in network.get("input_current", [])]
    if len(currents) < neuron_count:
        currents.extend([0.0] * (neuron_count - len(currents)))
    elif len(currents) > neuron_count:
        currents = currents[:neuron_count]
    network["input_current"] = currents


def sorted_network(network: dict[str, Any]) -> dict[str, Any]:
    network["neurons"] = sorted(network["neurons"], key=lambda neuron: neuron["id"])
    network["synapses"] = sorted(
        network["synapses"], key=lambda synapse: (synapse["from"], synapse["to"])
    )
    ensure_input_current_shape(network)
    return network


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or edit Unicorn network JSON files from the command line."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="samples/network.json",
        help="Network JSON path to read and modify.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a fresh network file.")
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )
    init_parser.add_argument("--steps", type=int, default=10)
    init_parser.add_argument("--dt", type=float, default=1.0)
    init_parser.add_argument("--refractory-period", type=float, default=0.0)
    init_parser.add_argument(
        "--from-config",
        metavar="CONFIG",
        help=(
            "Bootstrap a complete network from a JSON config file. "
            "See samples/network_config.json for an example."
        ),
    )

    neuron_parser = subparsers.add_parser(
        "add-neuron", help="Append a neuron definition."
    )
    neuron_parser.add_argument("id", type=int)
    neuron_parser.add_argument("--threshold", type=float, default=1.0)
    neuron_parser.add_argument("--membrane-time-constant", type=float, default=10.0)
    neuron_parser.add_argument("--refractory-period", type=float, default=None)
    neuron_parser.add_argument("--reset-potential", type=float, default=None)
    neuron_parser.add_argument("--initial-voltage", type=float, default=None)
    neuron_parser.add_argument("--input-current", type=float, default=0.0)

    # ------------------------------------------------------------------ generate
    gen_parser = subparsers.add_parser(
        "generate",
        help=(
            "Generate a complete N-neuron network in one step. "
            "Shared defaults apply to every neuron; per-neuron overrides "
            "are given as comma-separated key=value lists."
        ),
    )
    gen_parser.add_argument(
        "count", type=int, help="Number of neurons to create (ids 0…N-1)."
    )
    # — shared neuron defaults —
    gen_parser.add_argument(
        "--threshold", type=float, default=1.0, help="Firing threshold (default 1.0)."
    )
    gen_parser.add_argument(
        "--tau",
        dest="membrane_time_constant",
        type=float,
        default=10.0,
        help="Membrane time constant τ (default 10.0).",
    )
    gen_parser.add_argument(
        "--refractory-period",
        type=float,
        default=None,
        help="Per-neuron refractory period (overrides global).",
    )
    gen_parser.add_argument(
        "--reset-potential",
        type=float,
        default=None,
        help="Reset potential after spike.",
    )
    gen_parser.add_argument(
        "--input-current",
        type=float,
        default=0.0,
        help="Input current applied to every neuron (default 0.0).",
    )
    # — per-neuron overrides (CSV key=value lists) —
    gen_parser.add_argument(
        "--neuron-overrides",
        metavar="id:key=val,...",
        default="",
        help=(
            "Comma-separated per-neuron overrides, e.g. "
            "\"0:threshold=1.5,input_current=0.8 2:tau=5\"  "
            "(space-separated entries, colon separates id from key=val pairs)."
        ),
        nargs="*",
    )
    # — simulation settings —
    gen_parser.add_argument("--steps", type=int, default=10)
    gen_parser.add_argument("--dt", type=float, default=1.0)
    gen_parser.add_argument(
        "--global-refractory", type=float, default=0.0, dest="global_refractory"
    )
    # — topology —
    gen_parser.add_argument(
        "--topology",
        choices=["none", "chain", "ring", "all-to-all", "random"],
        default="none",
        help=(
            "Auto-wire synapses: "
            "'chain' (0→1→…→N-1), "
            "'ring' (chain + N-1→0), "
            "'all-to-all' (every pair), "
            "'random' (use --density and --seed). "
            "Default: none."
        ),
    )
    gen_parser.add_argument(
        "--weight",
        type=float,
        default=0.5,
        help="Default synapse weight for topology presets (default 0.5).",
    )
    gen_parser.add_argument(
        "--density",
        type=float,
        default=0.3,
        help="Fraction of possible synapses to create for 'random' topology (0–1).",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for 'random' topology (for reproducibility).",
    )
    gen_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )

    # -------------------------------------------------------------------
    synapse_parser = subparsers.add_parser(
        "add-synapse", help="Append a synapse definition."
    )
    synapse_parser.add_argument("source", type=int)
    synapse_parser.add_argument("target", type=int)
    synapse_parser.add_argument("weight", type=float)
    synapse_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing synapse between the same source and target.",
    )

    current_parser = subparsers.add_parser(
        "set-input", help="Set external drive for an existing neuron by id."
    )
    current_parser.add_argument("id", type=int)
    current_parser.add_argument("current", type=float)

    config_parser = subparsers.add_parser(
        "set-config", help="Update global simulation settings."
    )
    config_parser.add_argument("--steps", type=int)
    config_parser.add_argument("--dt", type=float)
    config_parser.add_argument("--refractory-period", type=float)

    subparsers.add_parser("summary", help="Print a concise network summary.")
    subparsers.add_parser("validate", help="Validate references and numeric shapes.")

    # ── set-neuron ──────────────────────────────────────────────────────────
    sn = subparsers.add_parser(
        "set-neuron",
        help="Patch any property on an existing neuron.",
    )
    sn.add_argument("id", type=int, help="Neuron id to modify.")
    sn.add_argument("--threshold",         type=float)
    sn.add_argument("--tau",               type=float, dest="membrane_time_constant")
    sn.add_argument("--refractory-period", type=float)
    sn.add_argument("--reset-potential",   type=float)
    sn.add_argument("--v-rest",            type=float)
    sn.add_argument("--input-current",     type=float)
    sn.add_argument("--bias",              type=float)
    sn.add_argument("--gain",              type=float)
    sn.add_argument("--noise-std",         type=float, dest="noise_std")
    sn.add_argument("--dropout",           type=float, dest="dropout_prob")
    sn.add_argument("--adaptation-rate",   type=float, dest="adaptation_rate")
    sn.add_argument("--adaptation-decay",  type=float, dest="adaptation_decay")
    sn.add_argument(
        "--activation",
        choices=["lif", "relu", "softplus", "tanh", "sigmoid", "rbf"],
        dest="activation_fn",
    )
    sn.add_argument("--rbf-centre", type=float, dest="rbf_centre")
    sn.add_argument("--rbf-sigma",  type=float, dest="rbf_sigma")

    # ── mutate ──────────────────────────────────────────────────────────────
    mt = subparsers.add_parser(
        "mutate",
        help="Bulk property delta across a slice of neurons.",
    )
    mt.add_argument(
        "--select",
        default="all",
        help=(
            "Which neurons to target: 'all', 'range:0-9', 'every:3', "
            "or comma-separated ids like '0,3,7'."
        ),
    )
    mt.add_argument(
        "--set",
        metavar="key=val,...",
        help="Set properties to exact values, e.g. 'noise_std=0.05,dropout_prob=0.1'.",
    )
    mt.add_argument(
        "--add",
        metavar="key=delta,...",
        help="Add delta to properties, e.g. 'threshold=-0.1,gain=0.2'.",
    )
    mt.add_argument(
        "--scale",
        metavar="key=factor,...",
        help="Multiply properties by factor, e.g. 'gain=2.0,noise_std=0.5'.",
    )

    # ── apply-profile ───────────────────────────────────────────────────────
    ap = subparsers.add_parser(
        "apply-profile",
        help="Apply a named property preset to a set of neurons.",
    )
    ap.add_argument(
        "profile",
        nargs="?",
        choices=["inhibitory", "driver", "adaptive", "stochastic", "silent"],
        help="Built-in profile name.",
    )
    ap.add_argument(
        "--from-file",
        metavar="FILE",
        help="JSON file containing custom profiles.",
    )
    ap.add_argument("--name", help="Profile name when using --from-file.")
    ap.add_argument(
        "--to",
        default="all",
        help="Target neurons: 'all', 'range:0-9', 'every:3', or '0,3,7'.",
    )

    # ── set-schedule ────────────────────────────────────────────────────────
    ss = subparsers.add_parser(
        "set-schedule",
        help="Attach a time-varying input-current schedule to a neuron.",
    )
    ss.add_argument("id", type=int)
    ss.add_argument(
        "--mode",
        choices=["constant", "ramp", "pulse", "sine"],
        default="sine",
    )
    ss.add_argument("--amplitude", type=float, default=1.0)
    ss.add_argument("--period",    type=float, default=10.0,
                    help="Steps per cycle (pulse/sine) or ramp duration.")
    ss.add_argument("--offset",    type=float, default=0.0,
                    help="Constant baseline added to schedule output.")
    ss.add_argument("--duration",  type=float, default=None,
                    help="Limit schedule to this many steps (ramp only).")

    # ── props ───────────────────────────────────────────────────────────────
    pr = subparsers.add_parser(
        "props",
        help="Print all properties of selected neurons.",
    )
    pr.add_argument(
        "ids", nargs="*", type=int, help="Neuron ids to inspect (default: all)."
    )

    return parser


def get_neuron_index(network: dict[str, Any], neuron_id: int) -> int:
    for index, neuron in enumerate(network["neurons"]):
        if neuron["id"] == neuron_id:
            return index
    raise ValueError(f"Neuron {neuron_id} does not exist")


def validate_network(network: dict[str, Any]) -> None:
    neuron_ids = [neuron["id"] for neuron in network["neurons"]]
    if len(neuron_ids) != len(set(neuron_ids)):
        raise ValueError("Neuron ids must be unique")
    if sorted(neuron_ids) != list(range(len(neuron_ids))):
        raise ValueError("Neuron ids must be contiguous and start at 0")

    ensure_input_current_shape(network)

    for synapse in network["synapses"]:
        if synapse["from"] not in neuron_ids or synapse["to"] not in neuron_ids:
            raise ValueError(
                f"Synapse {synapse['from']} -> {synapse['to']} references a missing neuron"
            )

    dt = float(network.get("dt", 1.0))
    if dt <= 0:
        raise ValueError("dt must be greater than zero")

    if int(network.get("steps", 1)) <= 0:
        raise ValueError("steps must be greater than zero")


def cmd_init(args: argparse.Namespace) -> str:
    path = Path(args.path)
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} already exists; rerun with --force to overwrite")

    # Bootstrap from an external config file if provided
    if getattr(args, "from_config", None):
        config_path = Path(args.from_config)
        with config_path.open("r") as fh:
            cfg = json.load(fh)
        network = _network_from_config(cfg)
        validate_network(network)
        save_network(path, network)
        return f"Initialized {path} from {config_path}"

    network = json.loads(json.dumps(DEFAULT_NETWORK))
    network["steps"] = args.steps
    network["dt"] = args.dt
    network["refractory_period"] = args.refractory_period
    save_network(path, network)
    return f"Initialized {path}"


def cmd_add_neuron(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    if any(neuron["id"] == args.id for neuron in network["neurons"]):
        raise ValueError(f"Neuron {args.id} already exists")

    neuron = {
        "id": args.id,
        "threshold": args.threshold,
        "membrane_time_constant": args.membrane_time_constant,
    }
    if args.refractory_period is not None:
        neuron["refractory_period"] = args.refractory_period
    if args.reset_potential is not None:
        neuron["reset_potential"] = args.reset_potential
    if args.initial_voltage is not None:
        neuron["initial_voltage"] = args.initial_voltage

    network["neurons"].append(neuron)
    sorted_network(network)
    index = get_neuron_index(network, args.id)
    network["input_current"][index] = float(args.input_current)
    validate_network(network)
    save_network(path, network)
    return f"Added neuron {args.id} to {path}"


def cmd_add_synapse(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    validate_network(network)

    replacement_index = None
    for index, synapse in enumerate(network["synapses"]):
        if synapse["from"] == args.source and synapse["to"] == args.target:
            replacement_index = index
            break

    new_synapse = {"from": args.source, "to": args.target, "weight": args.weight}
    if replacement_index is not None:
        if not args.replace:
            raise ValueError(
                f"Synapse {args.source} -> {args.target} already exists; use --replace to update it"
            )
        network["synapses"][replacement_index] = new_synapse
    else:
        network["synapses"].append(new_synapse)

    sorted_network(network)
    validate_network(network)
    save_network(path, network)
    return f"Saved synapse {args.source} -> {args.target} in {path}"


def cmd_set_input(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    sorted_network(network)
    index = get_neuron_index(network, args.id)
    network["input_current"][index] = float(args.current)
    validate_network(network)
    save_network(path, network)
    return f"Set input current for neuron {args.id} to {args.current}"


def cmd_set_config(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    if args.steps is not None:
        network["steps"] = args.steps
    if args.dt is not None:
        network["dt"] = args.dt
    if args.refractory_period is not None:
        network["refractory_period"] = args.refractory_period
    validate_network(network)
    save_network(path, network)
    return f"Updated simulation config in {path}"


def cmd_summary(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = sorted_network(load_network(path))
    validate_network(network)
    lines = [
        f"Path: {path}",
        f"Neurons: {len(network['neurons'])}",
        f"Synapses: {len(network['synapses'])}",
        f"dt: {network.get('dt', 1.0)}",
        f"steps: {network.get('steps', 10)}",
    ]
    for neuron, current in zip(network["neurons"], network["input_current"]):
        lines.append(
            "  - neuron {id}: threshold={threshold}, tau={tau}, input_current={current}".format(
                id=neuron["id"],
                threshold=neuron.get("threshold", 1.0),
                tau=neuron.get("membrane_time_constant", 10.0),
                current=current,
            )
        )
    for synapse in network["synapses"]:
        lines.append(
            f"  - synapse {synapse['from']} -> {synapse['to']} weight={synapse['weight']:+.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers for generate / from-config
# ---------------------------------------------------------------------------


def _parse_neuron_overrides(override_list: list[str]) -> dict[int, dict[str, Any]]:
    """Parse override tokens like '0:threshold=1.5,input_current=0.8'.

    Returns {neuron_id: {field: value, ...}, ...}.
    """
    result: dict[int, dict[str, Any]] = {}
    float_fields = {
        "threshold",
        "membrane_time_constant",
        "tau",
        "refractory_period",
        "reset_potential",
        "initial_voltage",
        "input_current",
    }
    for token in override_list or []:
        if ":" not in token:
            raise ValueError(
                f"Invalid override '{token}': expected format 'id:key=val,...'"
            )
        id_str, kv_str = token.split(":", 1)
        nid = int(id_str)
        overrides: dict[str, Any] = {}
        for kv in kv_str.split(","):
            k, v = kv.strip().split("=", 1)
            k = k.strip()
            # normalise tau alias
            if k == "tau":
                k = "membrane_time_constant"
            overrides[k] = float(v) if k in float_fields else v
        result[nid] = overrides
    return result


def _wire_topology(
    n: int, topology: str, weight: float, density: float, seed: int | None
) -> list[dict[str, Any]]:
    """Return a synapse list for the requested auto-wire topology."""
    synapses: list[dict[str, Any]] = []
    if topology == "none":
        return synapses
    if topology in ("chain", "ring"):
        for i in range(n - 1):
            synapses.append({"from": i, "to": i + 1, "weight": weight})
        if topology == "ring" and n > 1:
            synapses.append({"from": n - 1, "to": 0, "weight": weight})
    elif topology == "all-to-all":
        for src in range(n):
            for tgt in range(n):
                if src != tgt:
                    synapses.append({"from": src, "to": tgt, "weight": weight})
    elif topology == "random":
        rng = _random.Random(seed)
        for src in range(n):
            for tgt in range(n):
                if src != tgt and rng.random() < density:
                    synapses.append({"from": src, "to": tgt, "weight": weight})
    return synapses


def _network_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a full network dict from a compact config recipe.

    Example config shape::

        {
          "count": 5,
          "steps": 20,
          "dt": 0.5,
          "global_refractory": 1.0,
          "threshold": 1.0,
          "tau": 10.0,
          "input_current": 0.0,
          "topology": "ring",
          "weight": 0.6,
          "density": 0.3,
          "seed": 42,
          "neuron_overrides": [
            {"id": 0, "input_current": 1.2, "threshold": 0.8},
            {"id": 2, "tau": 5.0, "refractory_period": 2.0}
          ],
          "synapses": [
            {"from": 0, "to": 3, "weight": -0.4}
          ]
        }
    """
    n = int(cfg["count"])
    threshold = float(cfg.get("threshold", 1.0))
    tau = float(cfg.get("tau", 10.0))
    default_refractory = cfg.get("refractory_period", cfg.get("global_refractory"))
    reset_potential = cfg.get("reset_potential")
    default_current = float(cfg.get("input_current", 0.0))

    # index overrides by id
    overrides_by_id: dict[int, dict] = {}
    for entry in cfg.get("neuron_overrides", []):
        overrides_by_id[int(entry["id"])] = entry

    neurons = []
    currents = []
    for nid in range(n):
        ov = overrides_by_id.get(nid, {})
        neuron: dict[str, Any] = {
            "id": nid,
            "threshold": float(ov.get("threshold", threshold)),
            "membrane_time_constant": float(ov.get("tau", ov.get("membrane_time_constant", tau))),
        }
        ref = ov.get("refractory_period", default_refractory)
        if ref is not None:
            neuron["refractory_period"] = float(ref)
        rp = ov.get("reset_potential", reset_potential)
        if rp is not None:
            neuron["reset_potential"] = float(rp)
        iv = ov.get("initial_voltage")
        if iv is not None:
            neuron["initial_voltage"] = float(iv)
        neurons.append(neuron)
        currents.append(float(ov.get("input_current", default_current)))

    topology = cfg.get("topology", "none")
    weight = float(cfg.get("weight", 0.5))
    density = float(cfg.get("density", 0.3))
    seed = cfg.get("seed")
    synapses = _wire_topology(n, topology, weight, density, seed)
    # append any explicit synapse overrides
    for syn in cfg.get("synapses", []):
        synapses.append({"from": int(syn["from"]), "to": int(syn["to"]), "weight": float(syn["weight"])})

    network: dict[str, Any] = {
        "neurons": neurons,
        "synapses": sorted(synapses, key=lambda s: (s["from"], s["to"])),
        "input_current": currents,
        "steps": int(cfg.get("steps", 10)),
        "dt": float(cfg.get("dt", 1.0)),
        "refractory_period": float(cfg.get("global_refractory", cfg.get("refractory_period", 0.0))),
    }
    return network


def cmd_generate(args: argparse.Namespace) -> str:
    """Build a full N-neuron network in a single command."""
    path = Path(args.path)
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} already exists; rerun with --force to overwrite")

    n = args.count
    if n <= 0:
        raise ValueError("count must be a positive integer")

    # parse per-neuron overrides
    override_tokens = args.neuron_overrides or []
    overrides = _parse_neuron_overrides(override_tokens)

    neurons = []
    currents = []
    for nid in range(n):
        ov = overrides.get(nid, {})
        neuron: dict[str, Any] = {
            "id": nid,
            "threshold": float(ov.get("threshold", args.threshold)),
            "membrane_time_constant": float(
                ov.get(
                    "membrane_time_constant",
                    ov.get("tau", args.membrane_time_constant),
                )
            ),
        }
        ref = ov.get("refractory_period", args.refractory_period)
        if ref is not None:
            neuron["refractory_period"] = float(ref)
        rp = ov.get("reset_potential", args.reset_potential)
        if rp is not None:
            neuron["reset_potential"] = float(rp)
        neurons.append(neuron)
        currents.append(float(ov.get("input_current", args.input_current)))

    synapses = _wire_topology(
        n, args.topology, args.weight, args.density, args.seed
    )

    network: dict[str, Any] = {
        "neurons": neurons,
        "synapses": sorted(synapses, key=lambda s: (s["from"], s["to"])),
        "input_current": currents,
        "steps": args.steps,
        "dt": args.dt,
        "refractory_period": args.global_refractory,
    }
    validate_network(network)
    save_network(path, network)

    syn_count = len(synapses)
    return (
        f"Generated {n}-neuron network ({args.topology} topology, "
        f"{syn_count} synapse{'' if syn_count == 1 else 's'}) → {path}"
    )


# ── Helpers shared by new commands ───────────────────────────────────────────

def _parse_selector(select_str: str, all_ids: list[int]) -> list[int]:
    """Return the list of neuron ids matching a selector expression."""
    s = select_str.strip()
    if s == "all":
        return list(all_ids)
    if s.startswith("range:"):
        lo, hi = s[6:].split("-")
        return [i for i in all_ids if int(lo) <= i <= int(hi)]
    if s.startswith("every:"):
        step = int(s[6:])
        return all_ids[::step]
    # Comma-separated ids
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_kv(raw: str | None) -> dict[str, float]:
    """Parse 'key=val,key=val' string into a {key: float} dict."""
    if not raw:
        return {}
    result = {}
    for token in raw.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        result[k.strip()] = float(v.strip())
    return result


_NEURON_PROP_KEYS = {
    "threshold", "membrane_time_constant", "refractory_period",
    "reset_potential", "initial_voltage", "v_rest",
    "bias", "gain", "noise_std", "dropout_prob",
    "adaptation_rate", "adaptation_decay",
    "activation_fn", "rbf_centre", "rbf_sigma",
}

BUILTIN_PROFILES: dict[str, dict] = {
    "inhibitory":  {"gain": -1.0,  "threshold": 0.8,   "membrane_time_constant": 5.0},
    "driver":      {"input_current": 1.5, "threshold": 0.7, "noise_std": 0.02},
    "adaptive":    {"adaptation_rate": 0.3, "adaptation_decay": 0.9, "refractory_period": 2.0},
    "stochastic":  {"dropout_prob": 0.25, "noise_std": 0.05, "activation_fn": "sigmoid"},
    "silent":      {"input_current": 0.0, "dropout_prob": 1.0},
}


# ── set-neuron ────────────────────────────────────────────────────────────────

def cmd_set_neuron(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    idx = get_neuron_index(network, args.id)
    neu = network["neurons"][idx]

    _DIRECT_PROPS = [
        "threshold", "membrane_time_constant", "refractory_period",
        "reset_potential", "v_rest", "bias", "gain", "noise_std",
        "dropout_prob", "adaptation_rate", "adaptation_decay",
        "activation_fn", "rbf_centre", "rbf_sigma",
    ]
    changed = []
    for prop in _DIRECT_PROPS:
        val = getattr(args, prop, None)
        if val is not None:
            neu[prop] = val
            changed.append(f"{prop}={val}")

    # input_current lives in the top-level array
    ic = getattr(args, "input_current", None)
    if ic is not None:
        ensure_input_current_shape(network)
        network["input_current"][idx] = ic
        changed.append(f"input_current={ic}")

    save_network(path, network)
    return f"Updated neuron {args.id}: {', '.join(changed) if changed else '(no changes)'}"


# ── mutate ────────────────────────────────────────────────────────────────────

def cmd_mutate(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    all_ids = [n["id"] for n in network["neurons"]]
    targets = _parse_selector(args.select, all_ids)
    ensure_input_current_shape(network)

    sets   = _parse_kv(args.set)
    adds   = _parse_kv(args.add)
    scales = _parse_kv(args.scale)

    id_to_idx = {n["id"]: i for i, n in enumerate(network["neurons"])}

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

    save_network(path, network)
    ops = []
    if sets:   ops.append(f"set {sets}")
    if adds:   ops.append(f"add {adds}")
    if scales: ops.append(f"scale {scales}")
    return (
        f"Mutated {len(targets)} neurons "
        f"(selector='{args.select}'): {'; '.join(ops) or '(nothing)'}"
    )


# ── apply-profile ─────────────────────────────────────────────────────────────

def cmd_apply_profile(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    all_ids = [n["id"] for n in network["neurons"]]
    targets = _parse_selector(args.to, all_ids)
    ensure_input_current_shape(network)

    # Resolve profile dict
    if getattr(args, "from_file", None):
        with open(args.from_file) as fh:
            custom = json.load(fh)
        pname = args.name or args.profile
        if pname not in custom:
            raise ValueError(f"Profile '{pname}' not found in {args.from_file}")
        profile = custom[pname]
    elif args.profile:
        profile = BUILTIN_PROFILES[args.profile]
    else:
        raise ValueError("Provide a built-in profile name or --from-file + --name.")

    id_to_idx = {n["id"]: i for i, n in enumerate(network["neurons"])}
    for nid in targets:
        idx = id_to_idx[nid]
        neu = network["neurons"][idx]
        for k, v in profile.items():
            if k == "input_current":
                network["input_current"][idx] = v
            else:
                neu[k] = v

    save_network(path, network)
    pname = args.profile or args.name
    return f"Applied profile '{pname}' to {len(targets)} neurons (selector='{args.to}')"


# ── set-schedule ──────────────────────────────────────────────────────────────

def cmd_set_schedule(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    idx = get_neuron_index(network, args.id)
    sched: dict[str, Any] = {
        "mode":      args.mode,
        "amplitude": args.amplitude,
        "period":    args.period,
        "offset":    args.offset,
    }
    if args.duration is not None:
        sched["duration"] = args.duration
    network["neurons"][idx]["input_schedule"] = sched
    save_network(path, network)
    return (
        f"Attached {args.mode} schedule (A={args.amplitude}, T={args.period}) "
        f"to neuron {args.id}"
    )


# ── props ─────────────────────────────────────────────────────────────────────

_ALL_PROPS = [
    "threshold", "membrane_time_constant", "refractory_period",
    "reset_potential", "initial_voltage", "v_rest",
    "bias", "gain", "noise_std", "dropout_prob",
    "adaptation_rate", "adaptation_decay",
    "activation_fn", "rbf_centre", "rbf_sigma", "input_schedule",
]

def cmd_props(args: argparse.Namespace) -> str:
    path = Path(args.path)
    network = load_network(path)
    ensure_input_current_shape(network)
    target_ids = set(args.ids) if args.ids else None

    lines = []
    for i, neu in enumerate(network["neurons"]):
        nid = neu["id"]
        if target_ids is not None and nid not in target_ids:
            continue
        ic = network["input_current"][i]
        row = [f"Neuron {nid:>3}  input_current={ic:+.4f}"]
        for prop in _ALL_PROPS:
            val = neu.get(prop)
            if val is None:
                continue
            if isinstance(val, float):
                row.append(f"  {prop}={val:.4f}")
            else:
                row.append(f"  {prop}={val}")
        lines.append("\n".join(row))

    return "\n\n".join(lines) or "No neurons found."


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    commands = {
        "init":           cmd_init,
        "generate":       cmd_generate,
        "add-neuron":     cmd_add_neuron,
        "add-synapse":    cmd_add_synapse,
        "set-input":      cmd_set_input,
        "set-config":     cmd_set_config,
        "set-neuron":     cmd_set_neuron,
        "mutate":         cmd_mutate,
        "apply-profile":  cmd_apply_profile,
        "set-schedule":   cmd_set_schedule,
        "props":          cmd_props,
        "summary":        cmd_summary,
        "validate":       lambda parsed_args: (
            validate_network(load_network(Path(parsed_args.path)))
            or f"Validated {parsed_args.path}"
        ),
    }

    message = commands[args.command](args)
    print(message)


if __name__ == "__main__":
    main()
