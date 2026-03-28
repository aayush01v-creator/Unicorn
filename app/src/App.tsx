// app/src/App.tsx
import { useState, useEffect } from 'react';
import { SimpleSNN } from './lib/snn';
import type { NetworkConfig } from './lib/snn';
import { ForceDirectedLayout } from './lib/layout';
import { WebGPUViewer } from './components/WebGPUViewer';
import type { ViewerPayload } from './components/WebGPUViewer';
import { BrainCircuit, Play, Hexagon } from 'lucide-react';
import './index.css';

export default function App() {
  const [network, setNetwork] = useState<NetworkConfig | null>(null);
  const [viewerPayload, setViewerPayload] = useState<ViewerPayload | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  // Generate a random network on load
  useEffect(() => {
    const N = 500;
    const config: NetworkConfig = { neurons: [], synapses: [], steps: 200, dt: 1.0 };
    for(let i=0; i<N; i++) {
        config.neurons.push({
            id: i,
            activation_fn: Math.random() > 0.8 ? 'relu' : 'lif',
            threshold: 1.0 + Math.random() * 0.5,
            noise_std: Math.random() > 0.9 ? 0.3 : 0.0,
            dropout_prob: Math.random() > 0.95 ? 0.2 : 0.0,
            gain: 0.8 + Math.random() * 0.4,
            input_current: Math.random() > 0.8 ? 1.2 : 0.0
        });
    }
    // Random sparse connections
    for(let i=0; i<N; i++) {
        const k = 5 + Math.floor(Math.random()*15);
        for(let j=0; j<k; j++) {
            const target = Math.floor(Math.random()*N);
            if(target !== i) config.synapses.push({ from: i, to: target, weight: (Math.random()*0.8) - 0.2 });
        }
    }
    setNetwork(config);
  }, []);

  const runSimulation = async () => {
    if (!network) return;
    setIsSimulating(true);

    setTimeout(() => {
        const snn = new SimpleSNN(network);
        const history = snn.run();
        
        const layout = new ForceDirectedLayout(network, { k_repulsion: 0.1, k_spring: 0.05, max_step: 0.2 });
        const positions = layout.run(250);

        const props: [number, number, number, number, number][] = network.neurons.map(n => {
            const actId = {lif:0, relu:1, softplus:2, tanh:3, sigmoid:4, rbf:5}[n.activation_fn || 'lif'] ?? 0;
            return [actId, n.noise_std || 0, n.dropout_prob || 0, n.adaptation_rate || 0, n.gain || 1.0];
        });

        const numSteps = history.length;
        const totalSpikes = new Float32Array(numSteps * network.neurons.length);
        
        for (let t=0; t<numSteps; t++) {
            for(let i=0; i<network.neurons.length; i++) {
                totalSpikes[t * network.neurons.length + i] = history[t].spikes[i];
            }
        }

        const payload: ViewerPayload = {
            neuronCount: network.neurons.length,
            synapseCount: network.synapses.length,
            stepCount: numSteps,
            dt: network.dt || 1.0,
            decay: Math.exp(-(network.dt || 1.0) / 10.0),
            boost: 1.0,
            neuronProps: props,
            sceneGraph: {
                neurons: positions,
                synapses: network.synapses
            },
            cameraRadius: 20,
            spikes: Array.from(totalSpikes)
        };

        setViewerPayload(payload);
        setIsSimulating(false);
    }, 50);
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="header">
          <Hexagon className="logo-icon" size={28} color="#74b6ff" />
          <h1>Unicorn</h1>
        </div>
        <p className="subtitle">Cross-Platform SNN Framework</p>

        <div className="control-group">
          <div className="control-header">
             <BrainCircuit size={16} /> Networks
          </div>
          <div className="stat">Neurons: <span>{network?.neurons.length || 0}</span></div>
          <div className="stat">Synapses: <span>{network?.synapses.length || 0}</span></div>
        </div>

        <button className="primary-btn" onClick={runSimulation} disabled={isSimulating || !network}>
          {isSimulating ? 'Computing Physics & SNN...' : <><Play size={16} /> Run Visualization</>}
        </button>

        <div className="instructions">
          <h3>Simulation Ready</h3>
          <p>The network is configured to use fully vectorised client-side TypedArrays logic.</p>
        </div>
      </div>
      <div className="main-view">
        {viewerPayload ? (
            <WebGPUViewer payload={viewerPayload} />
        ) : (
            <div className="empty-state">
                <Hexagon size={64} className="empty-icon" />
                <h2>Ready to Simulate</h2>
                <p>Click "Run Visualization" to compute the force-directed layout and Spiking Neural Network dynamics.</p>
            </div>
        )}
      </div>
    </div>
  );
}
