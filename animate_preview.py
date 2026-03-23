import json
from pathlib import Path

import plotly.graph_objects as go

from backend.data_loader.json_loader import load_network
from backend.neuron_sim.simple_snn import SimpleSNN
from render_preview import build_edge_geometry


SPIKE_COLOR = "#f1c40f"
IDLE_COLOR = "#4c78a8"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_node_trace(xs, ys, zs, labels, spikes):
    colors = [SPIKE_COLOR if s == 1 else IDLE_COLOR for s in spikes]
    sizes = [15 if s == 1 else 9 for s in spikes]
    hover_text = [
        f"{label}<br>spike={'yes' if spike else 'no'}"
        for label, spike in zip(labels, spikes)
    ]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=sizes, color=colors),
        hovertemplate="%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        customdata=hover_text,
        name="Neurons",
    )


def main():
    network = load_network("samples/network.json")
    layout = load_json("samples/layout_output.json")

    sim = SimpleSNN(network)
    history = sim.run()

    pos = {item["id"]: item["position"] for item in layout}

    xs, ys, zs, labels = [], [], [], []
    for neuron in network["neurons"]:
        nid = neuron["id"]
        x, y, z = pos[nid]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        labels.append(f"Neuron {nid}")

    geometry = build_edge_geometry(network, pos)

    edge_trace = go.Scatter3d(
        x=geometry["edge_x"],
        y=geometry["edge_y"],
        z=geometry["edge_z"],
        mode="lines",
        line=dict(width=5, color=geometry["edge_colors"], colorscale="RdBu", cmin=-geometry["max_abs_weight"], cmax=geometry["max_abs_weight"], colorbar=dict(title="Weight", len=0.7)),
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
        marker=dict(size=4, color=geometry["weights"], colorscale="RdBu", cmin=-geometry["max_abs_weight"], cmax=geometry["max_abs_weight"], opacity=0.95),
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
        colorscale=[[0, "#f5f5f5"], [1, "#f5f5f5"]],
        showscale=False,
        hoverinfo="skip",
        name="Direction",
    )

    node_trace = build_node_trace(xs, ys, zs, labels, history[0]["spikes"])

    frames = []
    for step_data in history:
        frames.append(
            go.Frame(
                data=[edge_trace, weight_trace, arrow_trace, build_node_trace(xs, ys, zs, labels, step_data["spikes"])],
                name=str(step_data["step"]),
            )
        )

    fig = go.Figure(data=[edge_trace, weight_trace, arrow_trace, node_trace], frames=frames)

    fig.update_layout(
        title="Unicorn 3D Spike Animation",
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
                    text="Edges: green excitatory / red inhibitory, labels show weights, cones show direction",
                    xshift=10,
                    font=dict(size=12),
                )
            ],
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "label": str(i),
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for i in range(len(history))
                ]
            }
        ],
    )

    Path("viewer").mkdir(exist_ok=True)
    fig.write_html("viewer/spike_animation.html", include_plotlyjs=True)

    with open("samples/spike_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Saved: viewer/spike_animation.html")
    print("Saved: samples/spike_history.json")


if __name__ == "__main__":
    main()
