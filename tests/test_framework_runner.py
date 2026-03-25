import unittest
from unittest.mock import patch

from backend.neuron_sim.framework_runner import run_simulation, select_simulator


class FrameworkRunnerTests(unittest.TestCase):
    def test_auto_prefers_simple_when_frameworks_are_missing(self):
        with patch("backend.neuron_sim.framework_runner._module_available", return_value=False):
            self.assertEqual(select_simulator({"simulator": "auto"}), "simple")

    def test_invalid_simulator_raises(self):
        with self.assertRaises(ValueError):
            select_simulator({"simulator": "invalid"})

    def test_simple_simulator_runs(self):
        config = {
            "simulator": "simple",
            "neurons": [{"id": 0}],
            "synapses": [],
            "steps": 1,
            "dt": 1.0,
            "input_current": [0.0],
        }
        history = run_simulation(config)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["spikes"], [0])


if __name__ == "__main__":
    unittest.main()
