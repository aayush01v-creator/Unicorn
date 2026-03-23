import argparse
import json
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
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    init_parser.add_argument("--steps", type=int, default=10)
    init_parser.add_argument("--dt", type=float, default=1.0)
    init_parser.add_argument("--refractory-period", type=float, default=0.0)

    neuron_parser = subparsers.add_parser("add-neuron", help="Append a neuron definition.")
    neuron_parser.add_argument("id", type=int)
    neuron_parser.add_argument("--threshold", type=float, default=1.0)
    neuron_parser.add_argument("--membrane-time-constant", type=float, default=10.0)
    neuron_parser.add_argument("--refractory-period", type=float, default=None)
    neuron_parser.add_argument("--reset-potential", type=float, default=None)
    neuron_parser.add_argument("--initial-voltage", type=float, default=None)
    neuron_parser.add_argument("--input-current", type=float, default=0.0)

    synapse_parser = subparsers.add_parser("add-synapse", help="Append a synapse definition.")
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


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "add-neuron": cmd_add_neuron,
        "add-synapse": cmd_add_synapse,
        "set-input": cmd_set_input,
        "set-config": cmd_set_config,
        "summary": cmd_summary,
        "validate": lambda parsed_args: (validate_network(load_network(Path(parsed_args.path))) or f"Validated {parsed_args.path}"),
    }

    message = commands[args.command](args)
    print(message)


if __name__ == "__main__":
    main()
