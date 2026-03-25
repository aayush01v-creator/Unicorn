import unittest

from physics_engine.force_layout.simple_layout import ForceDirectedLayout


class ForceDirectedLayoutTests(unittest.TestCase):
    def test_layout_returns_3d_positions_for_each_neuron(self):
        config = {
            "neurons": [{"id": 0}, {"id": 1}, {"id": 2}],
            "synapses": [
                {"from": 0, "to": 1, "weight": 1.0},
                {"from": 1, "to": 2, "weight": -0.4},
            ],
        }

        positions = ForceDirectedLayout(config, seed=7).run(iterations=50)

        self.assertEqual(len(positions), 3)
        self.assertTrue(all(len(item["position"]) == 3 for item in positions))

    def test_negative_weight_still_acts_as_spring_strength(self):
        config = {
            "neurons": [{"id": 0}, {"id": 1}],
            "synapses": [{"from": 0, "to": 1, "weight": -1.0}],
            "layout": {
                "k_repulsion": 0.0,
                "k_spring": 0.8,
                "rest_length": 0.1,
                "dt": 0.1,
                "damping": 1.0,
                "max_step": 1.0,
            },
        }

        layout = ForceDirectedLayout(config, seed=1)
        before_delta = layout.positions[1] - layout.positions[0]
        before = float((before_delta @ before_delta) ** 0.5)

        for _ in range(20):
            layout.step()

        after_delta = layout.positions[1] - layout.positions[0]
        after = float((after_delta @ after_delta) ** 0.5)

        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
