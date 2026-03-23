import json
from pathlib import Path
import plotly.graph_objects as go

from backend.data_loader.json_loader import load_network
from backend.neuron_sim.simple_snn import SimpleSNN

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

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

    edge_x, edge_y, edge_z = [], [], []
    for syn in network["synapses"]:
        a = pos[syn["from"]]
        b = pos[syn["to"]]
        edge_x += [a[0], b[0], None]
        edge_y += [a[1], b[1], None]
        edge_z += [a[2], b[2], None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=4, color="gray"),
        hoverinfo="none",
        name="Synapses"
    )

    initial_spikes = history[0]["spikes"]
    initial_colors = ["red" if s == 1 else "blue" for s in initial_spikes]
    initial_sizes = [14 if s == 1 else 8 for s in initial_spikes]

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(
            size=initial_sizes,
            color=initial_colors
        ),
        hovertemplate="%{text}<br>x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
        name="Neurons"
    )

    frames = []
    for step_data in history:
        spikes = step_data["spikes"]
        colors = ["red" if s == 1 else "blue" for s in spikes]
        sizes = [14 if s == 1 else 8 for s in spikes]

        frame = go.Frame(
            data=[
                edge_trace,
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    marker=dict(size=sizes, color=colors),
                    hovertemplate="%{text}<br>x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
                    name="Neurons"
                )
            ],
            name=str(step_data["step"])
        )
        frames.append(frame)

    fig = go.Figure(data=[edge_trace, node_trace], frames=frames)

    fig.update_layout(
        title="Unicorn 3D Spike Animation",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
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
                        "args": [None, {"frame": {"duration": 700, "redraw": True},
                                        "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate"}]
                    }
                ]
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "label": str(i),
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate"}]
                    }
                    for i in range(len(history))
                ]
            }
        ]
    )

    Path("viewer").mkdir(exist_ok=True)
    fig.write_html("viewer/spike_animation.html", include_plotlyjs=True)

    with open("samples/spike_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Saved: viewer/spike_animation.html")
    print("Saved: samples/spike_history.json")

if __name__ == "__main__":
    main()
