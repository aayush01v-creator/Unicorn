import unittest

from webgpu_preview import build_webgpu_payload


class WebGpuPreviewPayloadTests(unittest.TestCase):
    def test_payload_contains_expected_counts_and_buffers(self):
        network = {
            "neurons": [{"id": 0}, {"id": 1}],
            "synapses": [{"from": 0, "to": 1, "weight": 0.4}],
            "dt": 0.5,
        }
        layout = [
            {"id": 0, "position": [0.0, 0.0, 0.0]},
            {"id": 1, "position": [1.0, 0.0, 0.0]},
        ]
        history = [
            {"step": 0, "time": 0.0, "spikes": [1, 0]},
            {"step": 1, "time": 0.5, "spikes": [0, 1]},
        ]

        payload = build_webgpu_payload(network, layout, history)

        self.assertEqual(payload["neuronCount"], 2)
        self.assertEqual(payload["synapseCount"], 1)
        self.assertEqual(payload["stepCount"], 2)
        self.assertAlmostEqual(payload["dt"], 0.5)
        self.assertEqual(payload["spikes"], [1, 0, 0, 1])
        self.assertGreater(len(payload["neuronVertices"]), 0)
        self.assertEqual(len(payload["edgeVertices"]), 8)

    def test_payload_maps_spikes_by_neuron_order_for_sparse_ids(self):
        network = {
            "neurons": [{"id": 10}, {"id": 42}, {"id": 99}],
            "synapses": [{"from": 10, "to": 42, "weight": 0.4}],
            "dt": 0.1,
        }
        layout = [
            {"id": 10, "position": [0.0, 0.0, 0.0]},
            {"id": 42, "position": [1.0, 0.0, 0.0]},
            {"id": 99, "position": [2.0, 0.0, 0.0]},
        ]
        history = [
            {"step": 0, "time": 0.0, "spikes": [1, 0, 1]},
            {"step": 1, "time": 0.1, "spikes": [0, 1, 0]},
        ]

        payload = build_webgpu_payload(network, layout, history)

        self.assertEqual(payload["spikes"], [1, 0, 1, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
