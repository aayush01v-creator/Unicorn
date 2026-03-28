// app/src/lib/layout.ts
import type { NetworkConfig } from './snn';

export interface LayoutOptions {
  k_repulsion?: number;
  k_spring?: number;
  rest_length?: number;
  dt?: number;
  damping?: number;
  max_step?: number;
}

export class ForceDirectedLayout {
  n: number;
  dim: number;
  dt: number;
  k_repulsion: number;
  k_spring: number;
  rest_length: number;
  damping: number;
  max_step: number;

  positions: Float64Array;
  velocities: Float64Array;

  spring_i: Int32Array;
  spring_j: Int32Array;
  spring_k: Float64Array;

  constructor(config: NetworkConfig, options: LayoutOptions = {}, dim: number = 3) {
    this.n = config.neurons.length;
    this.dim = dim;
    this.k_repulsion = options.k_repulsion ?? 0.05;
    this.k_spring = options.k_spring ?? 0.08;
    this.rest_length = options.rest_length ?? 1.0;
    this.dt = options.dt ?? 0.05;
    this.damping = options.damping ?? 0.85;
    this.max_step = options.max_step ?? 0.15;

    this.positions = new Float64Array(this.n * this.dim);
    this.velocities = new Float64Array(this.n * this.dim);
    
    // Initialize random positions [-1, 1]
    for(let i=0; i < this.positions.length; i++) {
       this.positions[i] = (Math.random() * 2) - 1.0;
    }

    const syns = config.synapses || [];
    this.spring_i = new Int32Array(syns.length);
    this.spring_j = new Int32Array(syns.length);
    this.spring_k = new Float64Array(syns.length);

    for (let i=0; i < syns.length; i++) {
        this.spring_i[i] = syns[i].from;
        this.spring_j[i] = syns[i].to;
        this.spring_k[i] = this.k_spring * Math.abs(syns[i].weight || 1.0);
    }
  }

  step() {
    if (this.n === 0) return;

    const n = this.n;
    const dim = this.dim;
    const forces = new Float64Array(n * dim);
    const pos = this.positions;

    // Repulsion (O(N^2) loop, which is fine for N < ~10k in V8)
    for (let i=0; i < n; i++) {
        for (let j=i+1; j < n; j++) {
            let dist2 = 0;
            for (let d=0; d<dim; d++) {
                const delta = pos[i*dim + d] - pos[j*dim + d];
                dist2 += delta * delta;
            }
            if (dist2 === 0) dist2 = 1e-12;
            const dist = Math.sqrt(dist2);
            const mag = this.k_repulsion / dist2;
            
            for (let d=0; d<dim; d++) {
                const unit = (pos[i*dim + d] - pos[j*dim + d]) / dist;
                const force = mag * unit;
                forces[i*dim + d] += force;
                forces[j*dim + d] -= force;
            }
        }
    }

    // Spring forces
    for (let s=0; s < this.spring_i.length; s++) {
        const i = this.spring_i[s];
        const j = this.spring_j[s];
        let dist2 = 0;
        for (let d=0; d<dim; d++) {
            const delta = pos[j*dim + d] - pos[i*dim + d];
            dist2 += delta * delta;
        }
        if (dist2 === 0) dist2 = 1e-8;
        const dist = Math.sqrt(dist2);
        const extension = dist - this.rest_length;
        const mag = this.spring_k[s] * extension;

        for (let d=0; d<dim; d++) {
            const unit = (pos[j*dim + d] - pos[i*dim + d]) / dist;
            const force = unit * mag;
            forces[i*dim + d] += force;
            forces[j*dim + d] -= force;
        }
    }

    // Velocity update
    for (let i=0; i < n; i++) {
        let speed2 = 0;
        for (let d=0; d < dim; d++) {
            this.velocities[i*dim + d] = (this.velocities[i*dim + d] + this.dt * forces[i*dim + d]) * this.damping;
            speed2 += this.velocities[i*dim + d] ** 2;
        }
        const speed = Math.max(Math.sqrt(speed2), 1e-8);
        if (speed > this.max_step) {
            const scale = this.max_step / speed;
            for (let d=0; d < dim; d++) {
                this.velocities[i*dim + d] *= scale;
            }
        }

        for (let d=0; d < dim; d++) {
            this.positions[i*dim + d] += this.dt * this.velocities[i*dim + d];
        }
    }

    // Recenter
    for (let d=0; d < dim; d++) {
        let sum = 0;
        for (let i=0; i < n; i++) sum += this.positions[i*dim + d];
        const mean = sum / n;
        for (let i=0; i < n; i++) this.positions[i*dim + d] -= mean;
    }
  }

  run(iterations: number = 200) {
      for(let i=0; i < iterations; i++) {
          this.step();
      }
      return Array.from({ length: this.n }, (_, i) => ({
          id: i,
          position: [
              this.positions[i*this.dim + 0],
              this.positions[i*this.dim + 1],
              this.positions[i*this.dim + 2]
          ]
      }));
  }
}
