import unittest

from backend.neuron_sim.simple_snn import SimpleSNN


class SimpleSNNTests(unittest.TestCase):
    def test_membrane_decay_and_dt_shape_voltage_growth(self):
        config = {
            "neurons": [
                {"id": 0, "threshold": 10.0, "membrane_time_constant": 2.0},
            ],
            "synapses": [],
            "input_current": [1.0],
            "steps": 2,
            "dt": 0.5,
        }

        history = SimpleSNN(config).run()

        self.assertAlmostEqual(history[0]["voltages"][0], 0.5, places=6)
        self.assertAlmostEqual(history[1]["voltages"][0], 0.8894003915, places=6)
        self.assertEqual(history[1]["time"], 0.5)

    def test_refractory_period_blocks_integration_until_countdown_ends(self):
        config = {
            "neurons": [
                {"id": 0, "threshold": 1.0, "refractory_period": 2.0},
            ],
            "synapses": [],
            "input_current": [1.2],
            "steps": 4,
            "dt": 1.0,
        }

        history = SimpleSNN(config).run()

        self.assertEqual(history[0]["spikes"], [1])
        self.assertEqual(history[1]["spikes"], [0])
        self.assertEqual(history[2]["spikes"], [0])
        self.assertEqual(history[3]["spikes"], [1])
        self.assertEqual(history[1]["voltages"], [0.0])
        self.assertEqual(history[2]["voltages"], [0.0])

    def test_invalid_dt_raises(self):
        config = {
            "neurons": [{"id": 0}],
            "synapses": [],
            "dt": 0,
        }

        with self.assertRaises(ValueError):
            SimpleSNN(config)

    def test_input_current_is_normalized_to_neuron_count(self):
        config = {
            "neurons": [{"id": 0}, {"id": 1}],
            "synapses": [],
            "input_current": [1.0],
            "steps": 1,
            "dt": 1.0,
        }

        history = SimpleSNN(config).run()

        self.assertEqual(len(history[0]["voltages"]), 2)
        self.assertEqual(history[0]["voltages"][1], 0.0)

    def test_invalid_synapse_reference_raises(self):
        config = {
            "neurons": [{"id": 0}],
            "synapses": [{"from": 0, "to": 1, "weight": 0.5}],
        }

        with self.assertRaises(ValueError):
            SimpleSNN(config)


if __name__ == "__main__":
    unittest.main()
