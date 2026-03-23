import json
from backend.data_loader.json_loader import load_network
from physics_engine.force_layout.simple_layout import ForceDirectedLayout

def main():
    config = load_network("samples/network.json")
    layout = ForceDirectedLayout(config)
    positions = layout.run(iterations=200)

    with open("samples/layout_output.json", "w") as f:
        json.dump(positions, f, indent=2)

    print("=== 3D Layout Output ===")
    for item in positions:
        print(f"Neuron {item['id']}: {item['position']}")

if __name__ == "__main__":
    main()
