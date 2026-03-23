import json
from pathlib import Path

import plotly.graph_objects as go


EXCITATORY_COLOR = "#2ecc71"
INHIBITORY_COLOR = "#e74c3c"
ARROW_COLOR = "#f5f5f5"
WEIGHT_COLORSCALE = "RdBu"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def synapse_style(weight: float):
    kind = "excitatory" if weight >= 0 else "inhibitory"
    color = EXCITATORY_COLOR if weight >= 0 else INHIBITORY_COLOR
    return kind, color


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
        }

    max_abs_weight = max(abs(syn.get("weight", 0.0)) for syn in network["synapses"]) or 1.0

    edge_x, edge_y, edge_z, edge_colors = [], [], [], []
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
        edge_colors += [weight, weight, None]
        line_widths.append(3 + (3 * normalized_weight))

        midpoint = [(start[i] + end[i]) / 2 for i in range(3)]
        mid_x.append(midpoint[0])
        mid_y.append(midpoint[1])
        mid_z.append(midpoint[2])
        weights.append(weight)
        weight_text.append(
            f"{kind.title()} synapse<br>{syn['from']} → {syn['to']}<br>weight={weight:+.2f}"
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
        "edge_colors": edge_colors,
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
    xs, ys, zs, labels = [], [], [], []
    for neuron in network["neurons"]:
        nid = neuron["id"]
        x, y, z = pos[nid]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        labels.append(f"Neuron {nid}")

    geometry = build_edge_geometry(network, pos)
    avg_line_width = sum(geometry["line_widths"]) / len(geometry["line_widths"]) if geometry["line_widths"] else 4

    edge_trace = go.Scatter3d(
        x=geometry["edge_x"],
        y=geometry["edge_y"],
        z=geometry["edge_z"],
        mode="lines",
        line=dict(width=avg_line_width, color=geometry["edge_colors"], colorscale=WEIGHT_COLORSCALE, cmin=-geometry["max_abs_weight"], cmax=geometry["max_abs_weight"], colorbar=dict(title="Weight", len=0.7)),
        hoverinfo="none",
        name="Synapses",
    )

    weight_trace = go.Scatter3d(
        x=geometry["mid_x"],
        y=geometry["mid_y"],
        z=geometry["mid_z"],
        mode="markers+text",
        text=[f"{weight:+.2f}" for weight in geometry["weights"]],
        textposition="top center",
        marker=dict(size=4, color=geometry["weights"], colorscale=WEIGHT_COLORSCALE, cmin=-geometry["max_abs_weight"], cmax=geometry["max_abs_weight"], opacity=0.95),
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
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=10, color=list(range(len(xs))), colorscale="Viridis"),
        hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        name="Neurons",
    )

    fig = go.Figure(data=[edge_trace, weight_trace, arrow_trace, node_trace])
    fig.update_layout(
        title="Unicorn 3D Neural Network Preview",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            annotations=[
                dict(
                    showarrow=False,
                    x=0,
                    y=0,
                    z=0,
                    text="Green = excitatory, red = inhibitory, cones show direction",
                    xshift=10,
                    font=dict(size=12),
                )
            ],
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    return fig


def main():
    network = load_json("samples/network.json")
    layout = load_json("samples/layout_output.json")
    pos = {item["id"]: item["position"] for item in layout}

    fig = build_figure(network, pos)

    Path("viewer").mkdir(exist_ok=True)
    fig.write_html("viewer/network_preview.html", include_plotlyjs=True)

    print("Saved: viewer/network_preview.html")


if __name__ == "__main__":
    main()
