from backend.data_loader.json_loader import load_network
from backend.neuron_sim.framework_runner import run_simulation

def main():
    config = load_network("samples/network.json")
    history = run_simulation(config)

    print("=== Simulation Output ===")
    for step_data in history:
        print(f"Step {step_data['step']}: spikes={step_data['spikes']} voltages={step_data['voltages']}")

if __name__ == "__main__":
    main()
