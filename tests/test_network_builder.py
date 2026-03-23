import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from animate_preview import compute_trail_intensities


REPO_ROOT = Path(__file__).resolve().parents[1]


class NetworkBuilderCliTests(unittest.TestCase):
    def run_cli(self, *args):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "network_builder.py"), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def test_cli_can_build_network_incrementally(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            network_path = Path(tmpdir) / "network.json"

            self.run_cli(str(network_path), "init", "--steps", "12", "--dt", "0.25")
            self.run_cli(
                str(network_path),
                "add-neuron",
                "0",
                "--threshold",
                "1.2",
                "--membrane-time-constant",
                "3.0",
                "--input-current",
                "0.8",
            )
            self.run_cli(str(network_path), "add-neuron", "1", "--input-current", "0.1")
            self.run_cli(str(network_path), "add-synapse", "0", "1", "0.5")
            self.run_cli(str(network_path), "set-input", "1", "0.4")
            summary = self.run_cli(str(network_path), "summary")
            validation = self.run_cli(str(network_path), "validate")

            with network_path.open("r") as f:
                network = json.load(f)

            self.assertEqual(network["steps"], 12)
            self.assertEqual(network["dt"], 0.25)
            self.assertEqual([neuron["id"] for neuron in network["neurons"]], [0, 1])
            self.assertEqual(network["input_current"], [0.8, 0.4])
            self.assertEqual(network["synapses"], [{"from": 0, "to": 1, "weight": 0.5}])
            self.assertIn("Synapses: 1", summary)
            self.assertIn("Validated", validation)


class AnimationHelperTests(unittest.TestCase):
    def test_trail_intensities_decay_over_recent_steps(self):
        history = [
            {"spikes": [1, 0]},
            {"spikes": [0, 1]},
            {"spikes": [0, 0]},
        ]

        intensities = compute_trail_intensities(history, 2)

        self.assertAlmostEqual(intensities[0], 1 / 3)
        self.assertAlmostEqual(intensities[1], 2 / 3)


if __name__ == "__main__":
    unittest.main()
