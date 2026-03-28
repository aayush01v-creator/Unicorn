// app/src/lib/snn.ts
export type ActivationFn = 'lif' | 'relu' | 'softplus' | 'tanh' | 'sigmoid' | 'rbf';

export interface Schedule {
  mode: 'constant' | 'ramp' | 'pulse' | 'sine';
  amplitude?: number;
  period?: number;
  offset?: number;
  duration?: number;
}

export interface NeuronConfig {
  id: string | number;
  initial_voltage?: number;
  threshold?: number;
  reset_potential?: number;
  v_rest?: number;
  membrane_time_constant?: number;
  refractory_period?: number;
  activation_fn?: ActivationFn;
  bias?: number;
  gain?: number;
  noise_std?: number;
  dropout_prob?: number;
  adaptation_rate?: number;
  adaptation_decay?: number;
  rbf_centre?: number;
  rbf_sigma?: number;
  input_schedule?: Schedule;
  input_current?: number;
}

export interface SynapseConfig {
  from: number;
  to: number;
  weight: number;
  plasticity_rule?: 'hebbian' | 'oja' | null;
  plasticity_lr?: number;
}

export interface NetworkConfig {
  neurons: NeuronConfig[];
  synapses: SynapseConfig[];
  steps?: number;
  dt?: number;
  membrane_time_constant?: number;
  refractory_period?: number;
  input_current?: number[];
}

export interface StepHistory {
  step: number;
  time: number;
  voltages: number[];
  spikes: number[];
  refractory_remaining: number[];
  adaptation: number[];
}

const ACTIVATION_IDS: Record<string, number> = {
  lif: 0, relu: 1, softplus: 2, tanh: 3, sigmoid: 4, rbf: 5
};

export class SimpleSNN {
  n: number;
  steps: number;
  dt: number;

  v: Float64Array;
  thresholds: Float64Array;
  reset_potentials: Float64Array;
  v_rest: Float64Array;
  decay: Float64Array;
  
  refractory_steps: Int32Array;
  refractory_countdown: Int32Array;
  
  input_current: Float64Array;
  bias: Float64Array;
  gain: Float64Array;
  noise_std: Float64Array;
  dropout_prob: Float64Array;
  adaptation_rate: Float64Array;
  adaptation_decay: Float64Array;
  adaptation: Float64Array;
  effective_thresh: Float64Array;
  
  fn_ids: Int32Array;
  rbf_centres: Float64Array;
  rbf_sigmas: Float64Array;

  schedules: { index: number, sched: Schedule }[];

  W: Float32Array;
  sparse: boolean;
  pre: Int32Array;
  post: Int32Array;
  wvals: Float64Array;

  plasticity: { pre: number, post: number, rule: string, lr: number }[];

  constructor(config: NetworkConfig) {
    this.n = config.neurons.length;
    this.steps = config.steps || 10;
    this.dt = config.dt || 1.0;
    if (this.dt <= 0) throw new Error("dt must be > 0");

    const n = this.n;
    this.v = new Float64Array(n);
    this.thresholds = new Float64Array(n);
    this.reset_potentials = new Float64Array(n);
    this.v_rest = new Float64Array(n);
    this.decay = new Float64Array(n);
    this.refractory_steps = new Int32Array(n);
    this.refractory_countdown = new Int32Array(n);
    
    this.input_current = new Float64Array(n);
    this.bias = new Float64Array(n);
    this.gain = new Float64Array(n);
    this.noise_std = new Float64Array(n);
    this.dropout_prob = new Float64Array(n);
    this.adaptation_rate = new Float64Array(n);
    this.adaptation_decay = new Float64Array(n);
    this.adaptation = new Float64Array(n);
    this.effective_thresh = new Float64Array(n);
    
    this.fn_ids = new Int32Array(n);
    this.rbf_centres = new Float64Array(n);
    this.rbf_sigmas = new Float64Array(n);

    this.schedules = [];

    const defTau = config.membrane_time_constant || 10.0;
    const defRef = config.refractory_period || 0.0;
    const globalIC = config.input_current || [];

    for (let i = 0; i < n; i++) {
       const neu = config.neurons[i];
       this.v[i] = neu.initial_voltage || 0.0;
       this.thresholds[i] = neu.threshold ?? 1.0;
       this.reset_potentials[i] = neu.reset_potential || 0.0;
       this.v_rest[i] = neu.v_rest || 0.0;
       
       const tau = neu.membrane_time_constant || defTau;
       this.decay[i] = Math.exp(-this.dt / tau);
       
       const refPeriod = neu.refractory_period ?? defRef;
       this.refractory_steps[i] = Math.ceil(refPeriod / this.dt);
       
       let ic = neu.input_current ?? (globalIC[i] || 0.0);
       this.input_current[i] = ic;
       
       this.bias[i] = neu.bias || 0.0;
       this.gain[i] = neu.gain ?? 1.0;
       this.noise_std[i] = neu.noise_std || 0.0;
       this.dropout_prob[i] = neu.dropout_prob || 0.0;
       this.adaptation_rate[i] = neu.adaptation_rate || 0.0;
       this.adaptation_decay[i] = neu.adaptation_decay ?? 1.0;
       this.effective_thresh[i] = this.thresholds[i];
       
       this.fn_ids[i] = ACTIVATION_IDS[neu.activation_fn || 'lif'] || 0;
       this.rbf_centres[i] = neu.rbf_centre ?? 0.5;
       this.rbf_sigmas[i] = neu.rbf_sigma ?? 0.3;

       if (neu.input_schedule) {
         this.schedules.push({ index: i, sched: neu.input_schedule });
       }
    }

    if (n <= 8000) {
       this.sparse = false;
       this.W = new Float32Array(n * n);
       this.pre = new Int32Array(0); this.post = new Int32Array(0); this.wvals = new Float64Array(0);
       for (const syn of config.synapses) {
         this.W[syn.from * n + syn.to] = syn.weight;
       }
    } else {
       this.sparse = true;
       this.W = new Float32Array(0);
       const sLen = config.synapses.length;
       this.pre = new Int32Array(sLen);
       this.post = new Int32Array(sLen);
       this.wvals = new Float64Array(sLen);
       for (let i = 0; i < sLen; i++) {
         const syn = config.synapses[i];
         this.pre[i] = syn.from;
         this.post[i] = syn.to;
         this.wvals[i] = syn.weight;
       }
    }

    this.plasticity = [];
    if (!this.sparse) {
       for (const syn of config.synapses) {
         if (syn.plasticity_rule === 'hebbian' || syn.plasticity_rule === 'oja') {
           this.plasticity.push({
             pre: syn.from, post: syn.to,
             rule: syn.plasticity_rule, lr: syn.plasticity_lr ?? 0.01
           });
         }
       }
    }
  }

  scheduleCurrent(sched: Schedule, step: number, dt: number): number {
    const t = step * dt;
    const mode = sched.mode || 'constant';
    const amp = sched.amplitude || 0.0;
    const per = Math.max(sched.period || 10.0, 1e-9);
    const off = sched.offset || 0.0;
    const dur = sched.duration;
    
    if (mode === 'constant') return amp + off;
    if (mode === 'ramp') {
      if (dur !== undefined && t > dur * dt) return amp + off;
      const tDur = (dur !== undefined ? dur : per) * dt;
      const frac = Math.min(t / Math.max(tDur, 1e-9), 1.0);
      return amp * frac + off;
    }
    if (mode === 'pulse') {
      return (step % per) < Math.max(Math.floor(per / 2), 1) ? amp + off : off;
    }
    if (mode === 'sine') {
      return amp * Math.sin(2 * Math.PI * t / (per * dt)) + off;
    }
    return off;
  }

  gaussianRandom(): number {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  run(): StepHistory[] {
    const history: StepHistory[] = [];
    let spikes = new Float64Array(this.n);
    const ic = new Float64Array(this.n);
    const syn = new Float64Array(this.n);

    for (let t = 0; t < this.steps; t++) {
      for (let i = 0; i < this.n; i++) ic[i] = this.input_current[i];
      for (const s of this.schedules) ic[s.index] = this.scheduleCurrent(s.sched, t, this.dt);

      syn.fill(0);
      if (!this.sparse) {
        for (let j = 0; j < this.n; j++) {
           let sum = 0;
           for (let i = 0; i < this.n; i++) {
             if (spikes[i]) sum += spikes[i] * this.W[i * this.n + j];
           }
           syn[j] = sum;
        }
      } else {
        for(let i=0; i<this.pre.length; i++) {
          if (spikes[this.pre[i]]) {
            syn[this.post[i]] += spikes[this.pre[i]] * this.wvals[i];
          }
        }
      }

      for(let i=0; i<this.n; i++) {
        syn[i] *= this.gain[i];
        if (this.noise_std[i] > 0) {
          syn[i] += this.gaussianRandom() * this.noise_std[i];
        }
      }

      for(let i=0; i<this.n; i++) {
        let alive = true;
        if (this.dropout_prob[i] > 0 && Math.random() < this.dropout_prob[i]) {
          alive = false;
        }
        
        const active = alive && (this.refractory_countdown[i] === 0);
        const drive = (ic[i] + syn[i] + this.bias[i]) * this.dt;

        if (active) {
           this.v[i] = this.v[i] * this.decay[i] + this.v_rest[i] * (1.0 - this.decay[i]) + drive;
        } else {
           this.v[i] = this.reset_potentials[i];
        }

        let act_out = this.v[i];
        switch(this.fn_ids[i]) {
           case 1: act_out = Math.max(0, this.v[i]); break;
           case 2: act_out = Math.log1p(Math.exp(Math.max(-80, Math.min(80, this.v[i])))); break;
           case 3: act_out = Math.tanh(this.v[i]); break;
           case 4: act_out = 1.0 / (1.0 + Math.exp(-Math.max(-80, Math.min(80, this.v[i])))); break;
           case 5: act_out = Math.exp(-0.5 * Math.pow((this.v[i] - this.rbf_centres[i]) / Math.max(this.rbf_sigmas[i], 1e-8), 2)); break;
        }

        if (this.fn_ids[i] === 0) {
           const didSpike = active && (this.v[i] >= this.effective_thresh[i]);
           spikes[i] = didSpike ? 1.0 : 0.0;
           if (didSpike) {
              this.v[i] = this.reset_potentials[i];
              this.refractory_countdown[i] = this.refractory_steps[i];
           } else {
              if (this.refractory_countdown[i] > 0) this.refractory_countdown[i] -= 1;
           }
        } else {
           spikes[i] = active ? Math.max(0, Math.min(1.0, act_out)) : 0.0;
           if (this.refractory_countdown[i] > 0) this.refractory_countdown[i] -= 1;
        }

        this.adaptation[i] *= this.adaptation_decay[i];
        if (spikes[i] > 0) {
           this.adaptation[i] += this.adaptation_rate[i];
        }
        this.effective_thresh[i] = this.thresholds[i] + this.adaptation[i];
      }

      if (this.plasticity.length > 0) {
         for (const p of this.plasticity) {
             const ps = spikes[p.pre];
             const qs = spikes[p.post];
             const wIdx = p.pre * this.n + p.post;
             if (p.rule === 'hebbian') {
                this.W[wIdx] += p.lr * ps * qs;
             } else if (p.rule === 'oja') {
                const w = this.W[wIdx];
                this.W[wIdx] += p.lr * (ps * qs - qs * qs * w);
             }
         }
      }

      history.push({
         step: t,
         time: parseFloat((t * this.dt).toFixed(10)),
         voltages: Array.from(this.v),
         spikes: Array.from(spikes).map(x => Math.floor(x)),
         refractory_remaining: Array.from(this.refractory_countdown).map(x => x * this.dt),
         adaptation: Array.from(this.adaptation)
      });
    }

    return history;
  }
}
