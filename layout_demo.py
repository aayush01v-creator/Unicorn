import argparse
import json

from backend.data_loader.json_loader import load_network
from physics_engine.force_layout.simple_layout import ForceDirectedLayout


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D layout from a network file.")
    parser.add_argument("network", nargs="?", default="samples/network.json", help="Path to Unicorn JSON, SONATA-style JSON, or NeuroML file")
    parser.add_argument("--output", default="samples/layout_output.json", help="Layout JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_network(args.network)
    layout = ForceDirectedLayout(config)
    positions = layout.run(iterations=200)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2)

    print("=== 3D Layout Output ===")
    for item in positions:
        print(f"Neuron {item['id']}: {item['position']}")


if __name__ == "__main__":
    main()
