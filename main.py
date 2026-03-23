from backend.data_loader.json_loader import load_network
from backend.neuron_sim.simple_snn import SimpleSNN

def main():
    config = load_network("samples/network.json")
    sim = SimpleSNN(config)
    history = sim.run()

    print("=== Simulation Output ===")
    for step_data in history:
        print(f"Step {step_data['step']}: spikes={step_data['spikes']} voltages={step_data['voltages']}")

if __name__ == "__main__":
    main()
