import argparse
import json
from pathlib import Path

from backend.data_loader.json_loader import load_network
from backend.neuron_sim.framework_runner import run_simulation
from tools.render_preview import ensure_positions


SPIKE_COLOR = "#f1c40f"
IDLE_COLOR = "#4c78a8"
TRAIL_COLOR = "#f39c12"
ACTIVE_PATH_COLOR = "#fff6b7"
TRAIL_WINDOW = 3
SPEEDS = {
    "Slow": 1200,
    "Normal": 700,
    "Fast": 300,
}


def go_module():
    import plotly.graph_objects as go

    return go


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_node_trace(xs, ys, zs, labels, spikes, time_value):
    go = go_module()
    colors = [SPIKE_COLOR if s == 1 else IDLE_COLOR for s in spikes]
    sizes = [10 if s == 1 else 6 for s in spikes]
    hover_text = [
        f"{label}<br>time={time_value:.2f}<br>spike={'yes' if spike else 'no'}"
        for label, spike in zip(labels, spikes)
    ]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="#ffffff")),
        hovertemplate="%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        customdata=hover_text,
        name="Neurons",
    )


def build_trail_trace(xs, ys, zs, labels, intensities, time_value):
    go = go_module()
    sizes = [10 + (18 * intensity) for intensity in intensities]
    opacities = [
        round(0.15 + (0.55 * intensity), 3) if intensity > 0 else 0.0
        for intensity in intensities
    ]
    hover_text = [
        f"{label}<br>time={time_value:.2f}<br>recent-spike-intensity={intensity:.2f}"
        for label, intensity in zip(labels, intensities)
    ]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=sizes, color=TRAIL_COLOR, opacity=opacities, symbol="circle-open"
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
        name="Recent spike trail",
    )


def build_active_path_trace(active_synapses, pos, current_spikes, time_value):
    go = go_module()
    edge_x, edge_y, edge_z = [], [], []
    midpoint_x, midpoint_y, midpoint_z, text = [], [], [], []

    for synapse in active_synapses:
        start = pos[synapse["from"]]
        end = pos[synapse["to"]]
        edge_x += [start[0], end[0], None]
        edge_y += [start[1], end[1], None]
        edge_z += [start[2], end[2], None]

        midpoint_x.append((start[0] + end[0]) / 2)
        midpoint_y.append((start[1] + end[1]) / 2)
        midpoint_z.append((start[2] + end[2]) / 2)
        text.append(
            f"Active path<br>{synapse['from']} → {synapse['to']}<br>weight={synapse.get('weight', 0.0):+.2f}<br>time={time_value:.2f}<br>source_spike={current_spikes[synapse['from']]}"
        )

    line_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=11, color=ACTIVE_PATH_COLOR),
        opacity=0.95,
        hoverinfo="none",
        name="Active path",
    )
    marker_trace = go.Scatter3d(
        x=midpoint_x,
        y=midpoint_y,
        z=midpoint_z,
        mode="markers",
        marker=dict(size=6, color=ACTIVE_PATH_COLOR, opacity=0.95),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=text,
        name="Active path labels",
    )
    return line_trace, marker_trace


def compute_trail_intensities(history, step_index):
    intensities = [0.0] * len(history[step_index]["spikes"])
    for offset in range(TRAIL_WINDOW):
        lookup = step_index - offset
        if lookup < 0:
            break
        weight = (TRAIL_WINDOW - offset) / TRAIL_WINDOW
        for neuron_index, spike in enumerate(history[lookup]["spikes"]):
            if spike:
                intensities[neuron_index] = max(intensities[neuron_index], weight)
    return intensities


def build_frame_data(history, step_index, xs, ys, zs, labels, pos, network):
    step_data = history[step_index]
    active_synapses = [
        synapse
        for synapse in network["synapses"]
        if step_data["spikes"][synapse["from"]]
    ]
    trail_intensities = compute_trail_intensities(history, step_index)
    active_path_trace, active_path_marker_trace = build_active_path_trace(
        active_synapses, pos, step_data["spikes"], step_data["time"]
    )
    node_trace = build_node_trace(
        xs, ys, zs, labels, step_data["spikes"], step_data["time"]
    )
    trail_trace = build_trail_trace(
        xs, ys, zs, labels, trail_intensities, step_data["time"]
    )
    return step_data, [
        active_path_trace,
        active_path_marker_trace,
        trail_trace,
        node_trace,
    ]


def animation_args(duration):
    return [
        None,
        {"frame": {"duration": duration, "redraw": True}, "fromcurrent": True},
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate spike animation from network/layout files."
    )
    parser.add_argument(
        "network",
        nargs="?",
        default="samples/network.json",
        help="Path to Unicorn JSON, SONATA-style JSON, or NeuroML file",
    )
    parser.add_argument(
        "--layout",
        default="samples/layout_output.json",
        help="Path to layout JSON file",
    )
    parser.add_argument(
        "--output",
        default="output/spike_animation.html",
        help="Animation HTML output path",
    )
    parser.add_argument(
        "--spikes",
        dest="spikes",
        default=None,
        help="Optional path to an existing spike history JSON (alias for skipping simulation rerun)",
    )
    parser.add_argument(
        "--history-output",
        default="samples/spike_history.json",
        help="Spike history JSON output path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    go = go_module()
    from render_preview import build_edge_geometry

    network = load_network(args.network)
    layout = load_json(args.layout)

    if args.spikes:
        history = load_json(args.spikes)
    else:
        history = run_simulation(network)

    pos = ensure_positions(network, layout)

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
        line=dict(
            width=5,
            color=geometry["edge_colors"],
            colorscale="RdBu",
            cmin=-geometry["max_abs_weight"],
            cmax=geometry["max_abs_weight"],
            colorbar=dict(title="Weight", len=0.7),
        ),
        hoverinfo="none",
        name="Synapses",
    )

    weight_trace = go.Scatter3d(
        x=geometry["mid_x"],
        y=geometry["mid_y"],
        z=geometry["mid_z"],
        mode="markers",
        marker=dict(
            size=3,
            color=geometry["weights"],
            colorscale="RdBu",
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
        colorscale=[[0, "#f5f5f5"], [1, "#f5f5f5"]],
        showscale=False,
        hoverinfo="skip",
        name="Direction",
    )

    initial_step, initial_dynamic_traces = build_frame_data(
        history, 0, xs, ys, zs, labels, pos, network
    )

    frames = []
    for step_index, _step_data in enumerate(history):
        step_data, dynamic_traces = build_frame_data(
            history, step_index, xs, ys, zs, labels, pos, network
        )
        frames.append(
            go.Frame(
                data=[edge_trace, weight_trace, arrow_trace, *dynamic_traces],
                name=str(step_data["step"]),
                layout=go.Layout(
                    title=f"Unicorn 3D Spike Animation — step {step_data['step']} ({step_data['time']:.2f}s)"
                ),
            )
        )

    fig = go.Figure(
        data=[edge_trace, weight_trace, arrow_trace, *initial_dynamic_traces],
        frames=frames,
    )

    fig.update_layout(
        title=f"Unicorn 3D Spike Animation — step {initial_step['step']} ({initial_step['time']:.2f}s)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        annotations=[
            dict(
                x=0.01,
                y=-0.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                text="<span style='font-size: 11px; color: #a0a0a0'>Edges: green excitatory / red inhibitory | Bright paths: active synapses | Rings: recent trails</span>",
                font=dict(size=13),
            )
        ],
        margin=dict(l=0, r=0, b=0, t=60),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": True,
                "x": 0.0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": label,
                        "method": "animate",
                        "args": animation_args(duration),
                    }
                    for label, duration in SPEEDS.items()
                ]
                + [
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    }
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "method": "animate",
                        "label": f"{step_data['step']} ({step_data['time']:.2f}s)",
                        "args": [
                            [str(step_data["step"])],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for step_data in history
                ],
            }
        ],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
    )

    history_path = Path(args.history_output)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved: {output_path}")
    print(f"Saved: {history_path}")


if __name__ == "__main__":
    main()
