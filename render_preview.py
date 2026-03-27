import json
import argparse
import math
from pathlib import Path

EXCITATORY_COLOR = "#2ecc71"
INHIBITORY_COLOR = "#e74c3c"
ARROW_COLOR = "#f5f5f5"
WEIGHT_COLORSCALE = "RdBu"
NODE_COLORSCALE = "Viridis"
CAMERA_EYE = dict(x=1.55, y=1.55, z=1.15)


def go_module():
    import plotly.graph_objects as go

    return go


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def ensure_positions(network, layout):
    """Return a complete neuron-id -> [x, y, z] map.

    Layout files can be stale (missing recently added neurons). Missing ids get
    deterministic fallback coordinates on a unit circle so rendering never fails.
    """
    if isinstance(layout, dict):
        pos = {
            int(neuron_id): coords
            for neuron_id, coords in layout.items()
            if isinstance(coords, list) and len(coords) == 3
        }
    else:
        pos = {
            item["id"]: item["position"]
            for item in layout
            if "id" in item and isinstance(item.get("position"), list) and len(item["position"]) == 3
        }

    neuron_ids = [neuron["id"] for neuron in network.get("neurons", [])]
    missing_ids = [nid for nid in neuron_ids if nid not in pos]
    if not missing_ids:
        return pos

    start_index = len(pos)
    total = max(len(neuron_ids), 1)
    for offset, nid in enumerate(missing_ids):
        angle = 2 * math.pi * ((start_index + offset) / total)
        pos[nid] = [round(math.cos(angle), 4), round(math.sin(angle), 4), 0.0]

    return pos


def parse_args():
    parser = argparse.ArgumentParser(description="Generate static 3D network preview HTML.")
    parser.add_argument("network", nargs="?", default="samples/network.json", help="Path to network JSON")
    parser.add_argument("--layout", default="samples/layout_output.json", help="Path to layout JSON")
    parser.add_argument("--output", default="output/network_preview.html", help="Output HTML path")
    return parser.parse_args()


def synapse_style(weight: float):
    kind = "positive" if weight >= 0 else "negative"
    color = EXCITATORY_COLOR if weight >= 0 else INHIBITORY_COLOR
    return kind, color


def compute_node_metrics(network):
    metrics = {}
    for neuron in network["neurons"]:
        metrics[neuron["id"]] = {"in_degree": 0, "out_degree": 0, "weight_load": 0.0}

    for synapse in network.get("synapses", []):
        source = metrics[synapse["from"]]
        target = metrics[synapse["to"]]
        weight = synapse.get("weight", 0.0)
        source["out_degree"] += 1
        target["in_degree"] += 1
        source["weight_load"] += abs(weight)
        target["weight_load"] += abs(weight)

    return metrics


def network_summary(network):
    excitatory = sum(1 for syn in network.get("synapses", []) if syn.get("weight", 0.0) >= 0)
    inhibitory = sum(1 for syn in network.get("synapses", []) if syn.get("weight", 0.0) < 0)
    return {
        "neurons": len(network.get("neurons", [])),
        "synapses": len(network.get("synapses", [])),
        "excitatory": excitatory,
        "inhibitory": inhibitory,
    }


def build_edge_geometry(network, pos):
    if not network["synapses"]:
        return {
            "edge_x": [],
            "edge_y": [],
            "edge_z": [],
            "edge_colors": [],
            "mid_x": [],
            "mid_y": [],
            "mid_z": [],
            "weights": [],
            "weight_text": [],
            "arrow_x": [],
            "arrow_y": [],
            "arrow_z": [],
            "arrow_u": [],
            "arrow_v": [],
            "arrow_w": [],
            "arrow_colors": [],
            "line_widths": [],
            "max_abs_weight": 1.0,
        }

    max_abs_weight = max(abs(syn.get("weight", 0.0)) for syn in network["synapses"]) or 1.0

    edge_x, edge_y, edge_z = [], [], []
    mid_x, mid_y, mid_z, weights, weight_text = [], [], [], [], []
    arrow_x, arrow_y, arrow_z = [], [], []
    arrow_u, arrow_v, arrow_w, arrow_colors = [], [], [], []
    line_widths = []

    for syn in network["synapses"]:
        start = pos[syn["from"]]
        end = pos[syn["to"]]
        weight = syn.get("weight", 0.0)
        kind, color = synapse_style(weight)
        normalized_weight = abs(weight) / max_abs_weight
        direction = [end[i] - start[i] for i in range(3)]

        edge_x += [start[0], end[0], None]
        edge_y += [start[1], end[1], None]
        edge_z += [start[2], end[2], None]
        line_widths.append(3 + (3 * normalized_weight))

        midpoint = [(start[i] + end[i]) / 2 for i in range(3)]
        mid_x.append(midpoint[0])
        mid_y.append(midpoint[1])
        mid_z.append(midpoint[2])
        weights.append(weight)
        weight_text.append(
            f"{kind.title()} direct interaction<br>{syn['from']} → {syn['to']}<br>weight={weight:+.2f}"
        )

        arrow_anchor = [start[i] + (direction[i] * 0.78) for i in range(3)]
        arrow_x.append(arrow_anchor[0])
        arrow_y.append(arrow_anchor[1])
        arrow_z.append(arrow_anchor[2])
        arrow_u.append(direction[0] * 0.18)
        arrow_v.append(direction[1] * 0.18)
        arrow_w.append(direction[2] * 0.18)
        arrow_colors.append(color)

    return {
        "edge_x": edge_x,
        "edge_y": edge_y,
        "edge_z": edge_z,
        "mid_x": mid_x,
        "mid_y": mid_y,
        "mid_z": mid_z,
        "weights": weights,
        "weight_text": weight_text,
        "arrow_x": arrow_x,
        "arrow_y": arrow_y,
        "arrow_z": arrow_z,
        "arrow_u": arrow_u,
        "arrow_v": arrow_v,
        "arrow_w": arrow_w,
        "arrow_colors": arrow_colors,
        "line_widths": line_widths,
        "max_abs_weight": max_abs_weight,
    }


def build_figure(network, pos):
    go = go_module()
    xs, ys, zs = [], [], []
    node_sizes, node_colors, node_hover, labels = [], [], [], []
    metrics = compute_node_metrics(network)
    summary = network_summary(network)

    for neuron in network["neurons"]:
        nid = neuron["id"]
        x, y, z = pos[nid]
        node_metric = metrics[nid]
        total_degree = node_metric["in_degree"] + node_metric["out_degree"]

        xs.append(x)
        ys.append(y)
        zs.append(z)
        labels.append(f"Random Variable {nid}")
        node_sizes.append(5 + (2 * total_degree))
        node_colors.append(neuron.get("input_current", network.get("input_current", [0.0] * len(network["neurons"]))[nid] if nid < len(network.get("input_current", [])) else 0.0))
        node_hover.append(
            "<br>".join(
                [
                    f"Random Variable {nid}",
                    f"threshold={neuron.get('threshold', 1.0):.2f}",
                    f"tau={neuron.get('membrane_time_constant', 'default')}",
                    f"refractory={neuron.get('refractory_period', network.get('refractory_period', 'default'))}",
                    f"input_current={node_colors[-1]:.2f}",
                    f"in_degree={node_metric['in_degree']}",
                    f"out_degree={node_metric['out_degree']}",
                    f"weight_load={node_metric['weight_load']:.2f}",
                ]
            )
        )

    geometry = build_edge_geometry(network, pos)
    avg_line_width = sum(geometry["line_widths"]) / len(geometry["line_widths"]) if geometry["line_widths"] else 4

    edge_trace = go.Scatter3d(
        x=geometry["edge_x"],
        y=geometry["edge_y"],
        z=geometry["edge_z"],
        mode="lines",
        line=dict(
            width=avg_line_width,
            color="#b8c2cc",
        ),
        hoverinfo="none",
        name="Direct Interactions",
    )

    weight_trace = go.Scatter3d(
        x=geometry["mid_x"],
        y=geometry["mid_y"],
        z=geometry["mid_z"],
        mode="markers",
        marker=dict(
            size=3,
            color=geometry["weights"],
            colorscale=WEIGHT_COLORSCALE,
            cmin=-geometry["max_abs_weight"],
            cmax=geometry["max_abs_weight"],
            opacity=0.95,
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=geometry["weight_text"],
        name="Weights",
    )

    arrow_trace = go.Cone(
        x=geometry["arrow_x"],
        y=geometry["arrow_y"],
        z=geometry["arrow_z"],
        u=geometry["arrow_u"],
        v=geometry["arrow_v"],
        w=geometry["arrow_w"],
        anchor="tail",
        sizemode="absolute",
        sizeref=0.18,
        colorscale=[[0, ARROW_COLOR], [1, ARROW_COLOR]],
        showscale=False,
        hoverinfo="skip",
        name="Direction",
    )

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale=NODE_COLORSCALE,
            colorbar=dict(title="Input current", x=1.05, len=0.45),
            line=dict(width=1, color="#ffffff"),
            opacity=0.96,
        ),
        hovertemplate="%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        customdata=node_hover,
        name="Random Variables",
    )

    fig = go.Figure(data=[edge_trace, weight_trace, arrow_trace, node_trace])
    fig.update_layout(
        title="Graph Describing Model Structure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=CAMERA_EYE),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=70),
        template="plotly_dark",
        paper_bgcolor="#111111",
        annotations=[
            dict(
                x=0.01,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                text=(
                    f"<span style='font-size: 12px; color: #b0b0b0'>"
                    f"Structured probabilistic models use graphs (in the graph theory sense of 'nodes' or 'vertices' connected by edges)<br>"
                    f"to represent interactions between random variables. Each node represents a random variable.<br>"
                    f"Each edge represents a direct interaction. These direct interactions imply other, indirect interactions,<br>"
                    f"but only the direct interactions need to be explicitly modeled. There is more than one way to describe the interaction."
                    f"</span>"
                ),
            ),
            dict(
                x=0.01,
                y=1.01,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                text=(
                    f"Random Variables: {summary['neurons']} | Interactions: {summary['synapses']} | "
                    f"Positive: {summary['excitatory']} | Negative: {summary['inhibitory']}<br>"
                    f"<span style='font-size: 11px; color: #a0a0a0'>Green: Positive | Red: Negative | Size: Degree | Color: Input Current</span>"
                ),
                font=dict(size=13),
            )
        ],
    )
    return fig


def main():
    args = parse_args()
    network = load_json(args.network)
    layout = load_json(args.layout)
    pos = ensure_positions(network, layout)

    fig = build_figure(network, pos)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
