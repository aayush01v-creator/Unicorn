import unittest
from unittest.mock import patch

from tools.animate_preview import parse_args


class AnimatePreviewCliTests(unittest.TestCase):
    def test_parse_args_accepts_spikes_alias(self):
        argv = [
            "animate_preview.py",
            "samples/network.json",
            "--layout",
            "samples/layout_output.json",
            "--spikes",
            "samples/spike_history.json",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.spikes, "samples/spike_history.json")


if __name__ == "__main__":
    unittest.main()
