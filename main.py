import argparse

from backend.data_loader.json_loader import load_network
from backend.neuron_sim.framework_runner import run_simulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Unicorn simulation from a network file."
    )
    parser.add_argument(
        "network",
        nargs="?",
        default="samples/network.json",
        help="Path to Unicorn JSON, SONATA-style JSON, or NeuroML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_network(args.network)
    history = run_simulation(config)

    print("=== Simulation Output ===")
    for step_data in history:
        print(
            f"Step {step_data['step']}: spikes={step_data['spikes']} voltages={step_data['voltages']}"
        )


if __name__ == "__main__":
    main()
