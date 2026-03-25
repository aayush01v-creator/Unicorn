import json
import tempfile
import unittest
from pathlib import Path

from backend.data_loader.json_loader import load_network


class NetworkLoaderStandardizationTests(unittest.TestCase):
    def test_loads_native_json(self):
        config = {
            "neurons": [{"id": 0}],
            "synapses": [],
            "steps": 5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "network.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            loaded = load_network(str(path))

        self.assertEqual(loaded["neurons"][0]["id"], 0)
        self.assertEqual(loaded["steps"], 5)

    def test_loads_sonata_style_json(self):
        sonata = {
            "nodes": [
                {"node_id": 10, "input_current": 0.3, "threshold": 0.8},
                {"node_id": 11, "input_current": 0.1},
            ],
            "edges": [
                {"source_node_id": 10, "target_node_id": 11, "weight": 0.6},
            ],
            "simulation": {"steps": 20, "dt": 0.25},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "network.sonata.json"
            path.write_text(json.dumps(sonata), encoding="utf-8")

            loaded = load_network(str(path))

        self.assertEqual(len(loaded["neurons"]), 2)
        self.assertEqual(loaded["synapses"], [{"from": 0, "to": 1, "weight": 0.6}])
        self.assertEqual(loaded["input_current"], [0.3, 0.1])
        self.assertEqual(loaded["steps"], 20)
        self.assertEqual(loaded["dt"], 0.25)

    def test_loads_neuroml_projection(self):
        neuroml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"toy\">
  <network id=\"net\">
    <population id=\"popA\" component=\"iafCell\" size=\"2\">
      <instance id=\"0\"/>
      <instance id=\"1\"/>
    </population>
    <projection id=\"proj\" presynapticPopulation=\"popA\" postsynapticPopulation=\"popA\">
      <connection id=\"0\" preCellId=\"../popA/0\" postCellId=\"../popA/1\" weight=\"0.9\"/>
    </projection>
  </network>
</neuroml>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "network.nml"
            path.write_text(neuroml, encoding="utf-8")

            loaded = load_network(str(path))

        self.assertEqual(len(loaded["neurons"]), 2)
        self.assertEqual(loaded["synapses"], [{"from": 0, "to": 1, "weight": 0.9}])


if __name__ == "__main__":
    unittest.main()
