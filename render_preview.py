import json
from pathlib import Path
import plotly.graph_objects as go

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    network = load_json("samples/network.json")
    layout = load_json("samples/layout_output.json")

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

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(
            size=8,
            color=list(range(len(xs))),
            colorscale="Viridis"
        ),
        hovertemplate="%{text}<br>x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
        name="Neurons"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Unicorn 3D Neural Network Preview",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    Path("viewer").mkdir(exist_ok=True)
    fig.write_html("viewer/network_preview.html", include_plotlyjs=True)

    print("Saved: viewer/network_preview.html")

if __name__ == "__main__":
    main()
