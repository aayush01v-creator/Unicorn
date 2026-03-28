import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.render_preview import (
    build_edge_geometry,
    build_figure,
    compute_node_metrics,
    ensure_positions,
)


class RenderPreviewTests(unittest.TestCase):
    def test_build_edge_geometry_handles_empty_synapse_list(self):
        network = {"neurons": [{"id": 0}], "synapses": []}
        geometry = build_edge_geometry(network, {0: [0.0, 0.0, 0.0]})

        self.assertEqual(geometry["max_abs_weight"], 1.0)
        self.assertEqual(geometry["edge_x"], [])
        self.assertEqual(geometry["line_widths"], [])

    def test_compute_node_metrics_tracks_in_and_out_degree(self):
        network = {
            "neurons": [{"id": 0}, {"id": 1}, {"id": 2}],
            "synapses": [
                {"from": 0, "to": 1, "weight": 0.5},
                {"from": 2, "to": 1, "weight": -0.2},
            ],
        }

        metrics = compute_node_metrics(network)

        self.assertEqual(metrics[1]["in_degree"], 2)
        self.assertEqual(metrics[0]["out_degree"], 1)
        self.assertAlmostEqual(metrics[1]["weight_load"], 0.7)

    def test_build_figure_exposes_summary_and_node_sizes(self):
        class FakeFigure:
            def __init__(self, data):
                self.data = data
                self.layout = SimpleNamespace(annotations=[])

            def update_layout(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self.layout, key, value)

        class FakeGo:
            @staticmethod
            def Scatter3d(**kwargs):
                return SimpleNamespace(**kwargs)

            @staticmethod
            def Cone(**kwargs):
                return SimpleNamespace(**kwargs)

            @staticmethod
            def Figure(data):
                return FakeFigure(data)

        network = {
            "neurons": [
                {"id": 0, "threshold": 1.0},
                {"id": 1, "threshold": 1.1},
            ],
            "synapses": [{"from": 0, "to": 1, "weight": 0.5}],
            "input_current": [0.9, 0.1],
        }
        pos = {0: [0.0, 0.0, 0.0], 1: [1.0, 0.0, 0.0]}

        with patch("tools.render_preview.go_module", return_value=FakeGo):
            fig = build_figure(network, pos)

        self.assertIn("Random Variables: 2 | Interactions: 1", fig.layout.annotations[1]["text"])
        self.assertEqual(list(fig.data[-1].marker["size"]), [7, 7])
        self.assertEqual(list(fig.data[-1].marker["color"]), [0.9, 0.1])

    def test_ensure_positions_fills_missing_ids(self):
        network = {"neurons": [{"id": 0}, {"id": 1}, {"id": 2}]}
        stale_layout = [
            {"id": 0, "position": [0.0, 0.0, 0.0]},
            {"id": 1, "position": [1.0, 0.0, 0.0]},
        ]

        pos = ensure_positions(network, stale_layout)

        self.assertEqual(pos[0], [0.0, 0.0, 0.0])
        self.assertEqual(pos[1], [1.0, 0.0, 0.0])
        self.assertIn(2, pos)
        self.assertEqual(len(pos[2]), 3)


if __name__ == "__main__":
    unittest.main()
