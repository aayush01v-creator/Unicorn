import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


NetworkDict = dict[str, Any]


def _normalize_native_network(raw: NetworkDict) -> NetworkDict:
    if "neurons" not in raw or "synapses" not in raw:
        raise ValueError("Native network config must include 'neurons' and 'synapses'.")
    return raw


def _parse_sonata_json(raw: NetworkDict) -> NetworkDict:
    """Parse a lightweight SONATA-style JSON structure into Unicorn's native schema.

    Supported SONATA-like fields:
    - nodes: [{"node_id": int, ...}] or {"nodes": [{"node_id": int, ...}]}
    - edges: [{"source_node_id": int, "target_node_id": int, "weight": float}]
             or {"edges": [...]} wrappers
    - simulation: {"dt": float, "steps": int}
    """

    raw_nodes = raw.get("nodes", [])
    raw_edges = raw.get("edges", [])

    if isinstance(raw_nodes, dict):
        raw_nodes = raw_nodes.get("nodes", [])
    if isinstance(raw_edges, dict):
        raw_edges = raw_edges.get("edges", [])

    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError("SONATA config expects list-like 'nodes' and 'edges' sections.")

    sorted_nodes = sorted(raw_nodes, key=lambda node: int(node.get("node_id", node.get("id", 0))))
    id_map: dict[int, int] = {}
    neurons = []
    input_current = []

    for local_idx, node in enumerate(sorted_nodes):
        source_id = int(node.get("node_id", node.get("id", local_idx)))
        id_map[source_id] = local_idx

        neuron = {"id": local_idx}
        for key in (
            "threshold",
            "membrane_time_constant",
            "refractory_period",
            "reset_potential",
            "initial_voltage",
        ):
            if key in node:
                neuron[key] = node[key]

        neurons.append(neuron)
        input_current.append(float(node.get("input_current", 0.0)))

    synapses = []
    for edge in raw_edges:
        source = int(edge.get("source_node_id", edge.get("source")))
        target = int(edge.get("target_node_id", edge.get("target")))
        if source not in id_map or target not in id_map:
            raise ValueError(f"SONATA edge references unknown node id: {source} -> {target}")
        synapses.append(
            {
                "from": id_map[source],
                "to": id_map[target],
                "weight": float(edge.get("weight", edge.get("syn_weight", 1.0))),
            }
        )

    simulation = raw.get("simulation", {})
    config: NetworkDict = {
        "neurons": neurons,
        "synapses": synapses,
        "input_current": input_current,
    }
    if "steps" in simulation:
        config["steps"] = int(simulation["steps"])
    if "dt" in simulation:
        config["dt"] = float(simulation["dt"])

    return config


def _parse_neuroml(path: Path) -> NetworkDict:
    tree = ET.parse(path)
    root = tree.getroot()

    neurons = []
    input_current = []
    population_map: dict[tuple[str, str], int] = {}

    for population in root.findall(".//{*}population"):
        pop_id = population.attrib.get("id", "population")
        instances = population.findall("{*}instance")
        for local_idx, instance in enumerate(instances):
            global_idx = len(neurons)
            population_map[(pop_id, str(local_idx))] = global_idx
            neurons.append({"id": global_idx})
            input_current.append(0.0)

    if not neurons:
        for idx, _cell in enumerate(root.findall(".//{*}cell")):
            neurons.append({"id": idx})
            input_current.append(0.0)

    synapses = []
    for projection in root.findall(".//{*}projection"):
        presyn = projection.attrib.get("presynapticPopulation", "")
        postsyn = projection.attrib.get("postsynapticPopulation", "")

        for connection in projection.findall("{*}connection"):
            pre_idx = connection.attrib.get("preCellId", "../0/0").split("/")[-1]
            post_idx = connection.attrib.get("postCellId", "../0/0").split("/")[-1]
            weight = float(connection.attrib.get("weight", 1.0))

            pre_key = (presyn, pre_idx)
            post_key = (postsyn, post_idx)
            if pre_key not in population_map or post_key not in population_map:
                raise ValueError(
                    f"NeuroML connection references unknown population members: {pre_key} -> {post_key}"
                )

            synapses.append(
                {
                    "from": population_map[pre_key],
                    "to": population_map[post_key],
                    "weight": weight,
                }
            )

    return {
        "neurons": neurons,
        "synapses": synapses,
        "input_current": input_current,
    }


def load_network(path: str) -> NetworkDict:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix in {".nml", ".xml"}:
        return _parse_neuroml(source)

    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "neurons" in data and "synapses" in data:
        return _normalize_native_network(data)

    if "nodes" in data and "edges" in data:
        return _parse_sonata_json(data)

    raise ValueError(
        "Unsupported network format. Provide Unicorn JSON, SONATA-style JSON (nodes/edges), or NeuroML (.nml/.xml)."
    )
