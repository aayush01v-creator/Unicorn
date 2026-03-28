// app/src/App.tsx
import { useState, useCallback } from 'react';
import { SimpleSNN } from './lib/snn';
import type { NetworkConfig, ActivationFn } from './lib/snn';
import { ForceDirectedLayout } from './lib/layout';
import { WebGPUViewer } from './components/WebGPUViewer';
import type { ViewerPayload } from './components/WebGPUViewer';
import './index.css';

// ── Icons (inline SVG for zero deps) ─────────────────────────
const IconBrain = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-1.14Z"/>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-1.14Z"/>
  </svg>
);
const IconLink = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
  </svg>
);
const IconSliders = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/>
  </svg>
);
const IconPlay = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="6 3 20 12 6 21 6 3"/>
  </svg>
);
const IconRefresh = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>
  </svg>
);
const IconLogo = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M4.22 4.22l2.12 2.12m11.32 11.32 2.12 2.12M2 12h3m14 0h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/>
  </svg>
);

// ── Configuration State ────────────────────────────────────────
interface NetworkParams {
  neuronCount: number;
  connectionDensity: number; // avg connections per neuron
  steps: number;
  dt: number;
  activationMix: Record<ActivationFn, number>; // % of each type
  globalThreshold: number;
  globalNoise: number;
  globalDropout: number;
  globalGain: number;
  adaptationRate: number;
  inputFraction: number; // fraction of neurons with input current
  inputStrength: number;
  plasticity: boolean;
}

const DEFAULT_PARAMS: NetworkParams = {
  neuronCount: 200,
  connectionDensity: 12,
  steps: 100,
  dt: 1.0,
  activationMix: { lif: 70, relu: 15, tanh: 10, sigmoid: 3, softplus: 2, rbf: 0 },
  globalThreshold: 1.0,
  globalNoise: 0.05,
  globalDropout: 0.02,
  globalGain: 1.0,
  adaptationRate: 0.0,
  inputFraction: 0.2,
  inputStrength: 1.2,
  plasticity: false,
};

const ACTIVATION_TYPES: ActivationFn[] = ['lif', 'relu', 'tanh', 'sigmoid', 'softplus', 'rbf'];

function buildNetwork(p: NetworkParams): NetworkConfig {
  const N = p.neuronCount;
  const config: NetworkConfig = { neurons: [], synapses: [], steps: p.steps, dt: p.dt };

  // Determine activation for each neuron from mix percentages
  const types: ActivationFn[] = [];
  const totalPct = ACTIVATION_TYPES.reduce((s, k) => s + (p.activationMix[k] || 0), 0);
  for (let i = 0; i < N; i++) {
    let cumul = 0;
    const r = Math.random() * Math.max(totalPct, 1);
    let chosen: ActivationFn = 'lif';
    for (const fn of ACTIVATION_TYPES) {
      cumul += (p.activationMix[fn] || 0);
      if (r <= cumul) { chosen = fn; break; }
    }
    types.push(chosen);
  }

  for (let i = 0; i < N; i++) {
    config.neurons.push({
      id: i,
      activation_fn: types[i],
      threshold: p.globalThreshold + (Math.random() - 0.5) * 0.3,
      noise_std: p.globalNoise > 0 ? p.globalNoise * (0.5 + Math.random()) : 0,
      dropout_prob: Math.random() < p.globalDropout * 5 ? p.globalDropout : 0,
      gain: p.globalGain * (0.85 + Math.random() * 0.3),
      adaptation_rate: p.adaptationRate > 0 ? p.adaptationRate * Math.random() : 0,
      input_current: Math.random() < p.inputFraction ? p.inputStrength * (0.8 + Math.random() * 0.4) : 0,
    });
  }

  // Sparse random connections based on density
  for (let i = 0; i < N; i++) {
    const k = Math.max(1, Math.round(p.connectionDensity * (0.6 + Math.random() * 0.8)));
    const targets = new Set<number>();
    let tries = 0;
    while (targets.size < k && tries < k * 4) {
      const t = Math.floor(Math.random() * N);
      if (t !== i) targets.add(t);
      tries++;
    }
    for (const t of targets) {
      const w = (Math.random() * 0.8) - 0.15;
      config.synapses.push({
        from: i, to: t, weight: w,
        plasticity_rule: p.plasticity && Math.random() < 0.1 ? 'hebbian' : null,
        plasticity_lr: 0.01,
      });
    }
  }
  return config;
}

// ── Slider Component ──────────────────────────────────────────
function Slider({ label, value, min, max, step = 1, format, onChange }: {
  label: string; value: number; min: number; max: number; step?: number;
  format?: (v: number) => string; onChange: (v: number) => void;
}) {
  const fmt = format ?? ((v: number) => String(v));
  return (
    <div className="param-row">
      <div className="param-label">
        <span style={{ color: 'var(--text-2)', fontWeight: 500 }}>{label}</span>
        <span>{fmt(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────
export default function App() {
  const [params, setParams] = useState<NetworkParams>(DEFAULT_PARAMS);
  const [viewerPayload, setViewerPayload] = useState<ViewerPayload | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  const set = useCallback((k: keyof NetworkParams, v: NetworkParams[keyof NetworkParams]) => {
    setParams(p => ({ ...p, [k]: v }));
  }, []);

  const setMix = useCallback((fn: ActivationFn, pct: number) => {
    setParams(p => ({ ...p, activationMix: { ...p.activationMix, [fn]: pct } }));
  }, []);

  const synapseEstimate = Math.round(params.neuronCount * params.connectionDensity);

  const runSimulation = () => {
    setIsSimulating(true);
    setTimeout(() => {
      const network = buildNetwork(params);
      const snn = new SimpleSNN(network);
      const history = snn.run();

      const layout = new ForceDirectedLayout(network, { k_repulsion: 0.12, k_spring: 0.06, max_step: 0.25 });
      const positions = layout.run(200);

      const props: [number, number, number, number, number][] = network.neurons.map(n => {
        const actId = ({ lif: 0, relu: 1, softplus: 2, tanh: 3, sigmoid: 4, rbf: 5 } as Record<string, number>)[n.activation_fn || 'lif'] ?? 0;
        return [actId, n.noise_std || 0, n.dropout_prob || 0, n.adaptation_rate || 0, n.gain ?? 1.0];
      });

      const numSteps = history.length;
      const totalSpikes = new Float32Array(numSteps * network.neurons.length);
      for (let t = 0; t < numSteps; t++) {
        for (let i = 0; i < network.neurons.length; i++) {
          totalSpikes[t * network.neurons.length + i] = history[t].spikes[i];
        }
      }

      const payload: ViewerPayload = {
        neuronCount: network.neurons.length,
        synapseCount: network.synapses.length,
        stepCount: numSteps,
        dt: network.dt || 1.0,
        decay: Math.exp(-(network.dt || 1.0) / 10.0),
        boost: 1.2,
        neuronProps: props,
        sceneGraph: { neurons: positions, synapses: network.synapses },
        cameraRadius: network.neurons.length > 500 ? 30 : 20,
        spikes: Array.from(totalSpikes),
      };
      setViewerPayload(payload);
      setIsSimulating(false);
    }, 60);
  };

  const reset = () => { setViewerPayload(null); setParams(DEFAULT_PARAMS); };

  return (
    <div className="app">
      {/* ── Top Bar ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="logo"><IconLogo /></div>
          <h1>Unicorn</h1>
          <span className="badge">SNN</span>
        </div>
        <div className="topbar-stats">
          <div className="topbar-stat">
            <strong>{params.neuronCount.toLocaleString()}</strong>
            neurons
          </div>
          <div className="topbar-stat">
            <strong>{synapseEstimate.toLocaleString()}</strong>
            synapses ~
          </div>
        </div>
      </header>

      {/* ── Body ── */}
      <div className="main">
        {/* ── Config Panel ── */}
        <aside className="config-panel">
          <div className="config-scroll">

            {/* Network Size */}
            <section className="config-section">
              <div className="config-section-header">
                <IconBrain /> Network Architecture
              </div>
              <div className="config-section-body">
                <Slider label="Neurons" value={params.neuronCount} min={20} max={2000} step={10}
                  onChange={v => set('neuronCount', v)} />
                <Slider label="Connections / Neuron" value={params.connectionDensity} min={1} max={50}
                  onChange={v => set('connectionDensity', v)} />
                <Slider label="Sim Steps" value={params.steps} min={10} max={500} step={10}
                  onChange={v => set('steps', v)} />
                <Slider label="dt (ms)" value={params.dt} min={0.1} max={5.0} step={0.1}
                  format={v => v.toFixed(1)}
                  onChange={v => set('dt', v)} />
              </div>
            </section>

            {/* Activation Mix */}
            <section className="config-section">
              <div className="config-section-header">
                <IconSliders /> Activation Mix (%)
              </div>
              <div className="config-section-body">
                {ACTIVATION_TYPES.map(fn => (
                  <Slider key={fn} label={fn.toUpperCase()} value={params.activationMix[fn]} min={0} max={100}
                    format={v => `${v}%`}
                    onChange={v => setMix(fn, v)} />
                ))}
              </div>
            </section>

            {/* Neuron Properties */}
            <section className="config-section">
              <div className="config-section-header">
                <IconSliders /> Neuron Properties
              </div>
              <div className="config-section-body">
                <Slider label="Threshold" value={params.globalThreshold} min={0.1} max={5.0} step={0.05}
                  format={v => v.toFixed(2)}
                  onChange={v => set('globalThreshold', v)} />
                <Slider label="Synaptic Gain" value={params.globalGain} min={0.1} max={4.0} step={0.05}
                  format={v => v.toFixed(2)}
                  onChange={v => set('globalGain', v)} />
                <Slider label="Noise σ" value={params.globalNoise} min={0} max={1.0} step={0.01}
                  format={v => v.toFixed(2)}
                  onChange={v => set('globalNoise', v)} />
                <Slider label="Dropout p" value={params.globalDropout} min={0} max={0.5} step={0.01}
                  format={v => v.toFixed(2)}
                  onChange={v => set('globalDropout', v)} />
                <Slider label="Adaptation" value={params.adaptationRate} min={0} max={2.0} step={0.05}
                  format={v => v.toFixed(2)}
                  onChange={v => set('adaptationRate', v)} />
              </div>
            </section>

            {/* Input Config */}
            <section className="config-section">
              <div className="config-section-header">
                <IconLink /> Input Drive
              </div>
              <div className="config-section-body">
                <Slider label="Input fraction" value={params.inputFraction} min={0} max={1} step={0.01}
                  format={v => `${(v * 100).toFixed(0)}%`}
                  onChange={v => set('inputFraction', v)} />
                <Slider label="Current strength" value={params.inputStrength} min={0.1} max={5.0} step={0.05}
                  format={v => v.toFixed(2)}
                  onChange={v => set('inputStrength', v)} />
                <div className="toggle-row">
                  <span>Hebbian Plasticity</span>
                  <label className="toggle">
                    <input type="checkbox" checked={params.plasticity}
                      onChange={e => set('plasticity', e.target.checked)} />
                    <span className="toggle-slider" />
                  </label>
                </div>
              </div>
            </section>

          </div>

          {/* Footer Buttons */}
          <div className="config-footer">
            <button className="run-btn" onClick={runSimulation} disabled={isSimulating}
              id="run-simulation-btn">
              {isSimulating ? <><div className="spinner" /> Computing…</> : <><IconPlay /> Run Simulation</>}
            </button>
            <button className="run-btn danger" onClick={reset} disabled={isSimulating}
              id="reset-btn" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-2)', boxShadow: 'none', border: '1px solid var(--border)' }}>
              <IconRefresh /> Reset
            </button>
          </div>
        </aside>

        {/* ── Viewer ── */}
        <div className="viewer">
          {isSimulating && (
            <div className="progress-overlay">
              <div className="progress-card">
                <div className="progress-title">Building Neural Network</div>
                <div className="progress-sub">Running {params.steps} steps on {params.neuronCount} neurons…</div>
                <div className="progress-bar-track">
                  <div className="progress-bar-fill" />
                </div>
              </div>
            </div>
          )}

          {viewerPayload ? (
            <>
              <WebGPUViewer payload={viewerPayload} />
              <div className="stats-bar">
                <div className="stat-card">
                  <div className="label">Neurons</div>
                  <div className="value">{viewerPayload.neuronCount.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                  <div className="label">Synapses</div>
                  <div className="value">{viewerPayload.synapseCount.toLocaleString()}</div>
                </div>
                <div className="stat-card active">
                  <div className="label">Steps</div>
                  <div className="value">{viewerPayload.stepCount}</div>
                </div>
              </div>
            </>
          ) : !isSimulating ? (
            <div className="empty-state">
              <div className="empty-orb">
                <IconLogo />
              </div>
              <h2>Configure & Simulate</h2>
              <p>Use the control panel to shape your Spiking Neural Network, then hit Run to watch it fire in 3D.</p>
              <div className="empty-steps">
                <div className="empty-step"><div className="empty-step-num">1</div>Set neuron count &amp; connectivity</div>
                <div className="empty-step"><div className="empty-step-num">2</div>Mix activation types &amp; tune properties</div>
                <div className="empty-step"><div className="empty-step-num">3</div>Hit Run Simulation to visualize</div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
