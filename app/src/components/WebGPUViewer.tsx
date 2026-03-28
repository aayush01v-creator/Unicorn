// app/src/components/WebGPUViewer.tsx
import { useEffect, useRef, useState } from 'react';

export interface ViewerPayload {
  neuronCount: number;
  synapseCount: number;
  stepCount: number;
  dt: number;
  decay: number;
  boost: number;
  neuronProps: [number, number, number, number, number][];
  sceneGraph: {
    neurons: { id: number | string; position: number[] }[];
    synapses: { from: number | string; to: number | string; weight?: number }[];
  };
  cameraRadius: number;
  spikes: number[]; // Flat array, size = stepCount * neuronCount
}

export function WebGPUViewer({ payload }: { payload: ViewerPayload | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [stats, setStats] = useState({ step: 0, dt: 0.1 });

  const pausedRef = useRef(paused);
  const speedRef = useRef(speed);
  pausedRef.current = paused;
  speedRef.current = speed;

  useEffect(() => {
    if (!payload || !canvasRef.current) return;
    
    let isCleanedUp = false;
    let animationFrameId: number;

    async function initGPU() {
      if (!navigator.gpu) {
        setError("WebGPU unavailable. Try Chrome 113+.");
        return;
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        setError("No compatible GPU adapter found.");
        return;
      }
      const device = await adapter.requestDevice();
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("webgpu") as unknown as GPUCanvasContext;
      if (!ctx) return;
      
      const format = navigator.gpu.getPreferredCanvasFormat();

      const resize = () => {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(canvas.clientWidth * dpr);
        canvas.height = Math.floor(canvas.clientHeight * dpr);
        ctx.configure({ device, format, alphaMode: "opaque" });
      };
      window.addEventListener("resize", resize);
      resize();

      let yaw = 0.3, pitch = 0.25;
      let radius = payload!.cameraRadius || 10;
      canvas.tabIndex = 0;
      canvas.style.touchAction = 'none';

      let isDragging = false, lastMX = 0, lastMY = 0;

      const onPointerDown = (e: PointerEvent) => { 
        isDragging = true; lastMX = e.clientX; lastMY = e.clientY; 
        try { canvas.setPointerCapture(e.pointerId); } catch(err) {}
      };
      const onPointerUp = (e: PointerEvent) => { 
        isDragging = false; 
        try { canvas.releasePointerCapture(e.pointerId); } catch(err) {}
      };
      const onPointerMove = (e: PointerEvent) => {
        if (!isDragging) return;
        yaw += (e.clientX - lastMX) * 0.005;
        pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - lastMY) * 0.005));
        lastMX = e.clientX; lastMY = e.clientY;
      };
      const onWheel = (e: WheelEvent) => {
        radius = Math.max(payload!.cameraRadius * 0.3, Math.min(payload!.cameraRadius * 3, radius + e.deltaY * 0.04));
      };
      const onKeyDown = (e: KeyboardEvent) => {
        const step = 0.1, zoomStep = payload!.cameraRadius * 0.1;
        switch(e.key) {
          case 'ArrowLeft': yaw -= step; e.preventDefault(); break;
          case 'ArrowRight': yaw += step; e.preventDefault(); break;
          case 'ArrowUp': pitch = Math.min(1.5, pitch + step); e.preventDefault(); break;
          case 'ArrowDown': pitch = Math.max(-1.5, pitch - step); e.preventDefault(); break;
          case '+': case '=': radius = Math.max(payload!.cameraRadius * 0.3, radius - zoomStep); e.preventDefault(); break;
          case '-': case '_': radius = Math.min(payload!.cameraRadius * 3, radius + zoomStep); e.preventDefault(); break;
        }
      };

      canvas.addEventListener("pointerdown", onPointerDown as EventListener);
      canvas.addEventListener("pointerup", onPointerUp as EventListener);
      canvas.addEventListener("pointercancel", onPointerUp as EventListener);
      canvas.addEventListener("pointermove", onPointerMove as EventListener);
      canvas.addEventListener("wheel", onWheel, { passive: true });
      canvas.addEventListener("keydown", onKeyDown);
      
      canvas.focus({ preventScroll: true });

      // Matrix helpers
      const matMul = (a: Float32Array, b: Float32Array) => {
        const o = new Float32Array(16);
        for (let r=0;r<4;r++) for (let c=0;c<4;c++) {
          let s=0; for (let k=0;k<4;k++) s+=a[r*4+k] * b[k*4+c]; o[r*4+c]=s;
        } return o;
      };
      const transpose = (m: Float32Array) => new Float32Array([
        m[0],m[4],m[8],m[12], m[1],m[5],m[9],m[13], m[2],m[6],m[10],m[14], m[3],m[7],m[11],m[15]
      ]);
      const perspective = (fovy: number, aspect: number, near: number, far: number) => {
        const f = 1.0/Math.tan(fovy/2), nf = 1.0/(near-far);
        return new Float32Array([ f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,2*far*near*nf, 0,0,-1,0 ]);
      };
      const lookAt = (eye: number[], target: number[], up: number[]) => {
        let fx=target[0]-eye[0], fy=target[1]-eye[1], fz=target[2]-eye[2];
        const fL = Math.hypot(fx,fy,fz)||1; fx/=fL; fy/=fL; fz/=fL;
        let rx=fy*up[2]-fz*up[1], ry=fz*up[0]-fx*up[2], rz=fx*up[1]-fy*up[0];
        const rL = Math.hypot(rx,ry,rz)||1; rx/=rL; ry/=rL; rz/=rL;
        const ux=ry*fz-rz*fy, uy=rz*fx-rx*fz, uz=rx*fy-ry*fx;
        return new Float32Array([
          rx, ry, rz, -(rx*eye[0]+ry*eye[1]+rz*eye[2]),
          ux, uy, uz, -(ux*eye[0]+uy*eye[1]+uz*eye[2]),
          -fx,-fy,-fz,  (fx*eye[0]+fy*eye[1]+fz*eye[2]),
          0,0,0,1
        ]);
      };

      const neurons = [...payload!.sceneGraph.neurons];
      const byId = new Map(neurons.map(n => [n.id, n.position]));
      const idToIndex = new Map(neurons.map((n,i) => [n.id, i]));

      const span = payload!.cameraRadius / 2.5;
      const baseArm = span * 0.015;

      const neuronVerts: number[] = [];
      const edgeVerts: number[] = [];

      for (const node of neurons) {
        const [x,y,z] = node.position;
        const owner = idToIndex.get(node.id) || 0;
        const props = payload!.neuronProps[owner] || [0,0,0,0,1.0];
        const gain = props[4] || 1.0;
        const arm = baseArm * Math.max(0.5, Math.min(3.0, gain));
        neuronVerts.push(x-arm,y,z,owner, x+arm,y,z,owner);
        neuronVerts.push(x,y-arm,z,owner, x,y+arm,z,owner);
        neuronVerts.push(x,y,z-arm,owner, x,y,z+arm,owner);
      }

      for (const syn of payload!.sceneGraph.synapses) {
        if (!byId.has(syn.from) || !byId.has(syn.to)) continue;
        const src = byId.get(syn.from)!, dst = byId.get(syn.to)!;
        const owner = idToIndex.get(syn.from) || 0;
        edgeVerts.push(src[0],src[1],src[2],owner, dst[0],dst[1],dst[2],owner);
      }

      const createBuf = (d: Float32Array | Uint32Array, usage: number) => {
        const b = device.createBuffer({ size: Math.max(d.byteLength, 4), usage, mappedAtCreation: true });
        if (d instanceof Float32Array) new Float32Array(b.getMappedRange()).set(d);
        else new Uint32Array(b.getMappedRange()).set(d);
        b.unmap(); return b;
      };

      const neuronBuf = createBuf(new Float32Array(neuronVerts), GPUBufferUsage.VERTEX);
      const edgeBuf = createBuf(new Float32Array(edgeVerts), GPUBufferUsage.VERTEX);
      const spikeBuf = createBuf(new Uint32Array(payload!.spikes), GPUBufferUsage.STORAGE);
      
      const nC = Math.max(payload!.neuronCount, 1);
      const intensityBuf = createBuf(new Float32Array(nC), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      
      const propsData = new Float32Array(nC * 4);
      for(let i=0; i<payload!.neuronCount; i++) {
        const p = payload!.neuronProps[i] || [0,0,0,0,1];
        propsData[i*4+0]=p[0]; propsData[i*4+1]=p[1]; propsData[i*4+2]=p[2]; propsData[i*4+3]=p[3];
      }
      const propsBuf = createBuf(propsData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

      const simUniBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const edgeUniBuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const neuronUniBuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

      const computeMod = device.createShaderModule({ code: `
        struct Sim { step: u32, neuronCount: u32, decay: f32, boost: f32 }
        @group(0) @binding(0) var<uniform> sim: Sim;
        @group(0) @binding(1) var<storage,read> spikes: array<u32>;
        @group(0) @binding(2) var<storage,read_write> intensities: array<f32>;
        @group(0) @binding(3) var<storage,read> neuronProps: array<f32>;
        
        fn hash(seed:u32)->f32{
          var s=seed; s=(s^61u)^(s>>16u); s*=9u; s^=s>>4u; s*=668265261u; s^=s>>15u;
          return f32(s&0xFFFFFFu)/16777215.0;
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
           let i = gid.x;
           if (i >= sim.neuronCount) { return; }
           let dropProb = neuronProps[i*4u + 2u];
           let rng = hash(sim.step * sim.neuronCount + i);
           if (dropProb > 0.0 && rng < dropProb) {
             intensities[i] *= sim.decay;
             return;
           }
           let idx = sim.step * sim.neuronCount + i;
           let spike = f32(spikes[idx]);
           intensities[i] = max(intensities[i]*sim.decay, spike*sim.boost);
        }
      `});

      const drawMod = device.createShaderModule({code: `
        struct Draw { mvp:mat4x4<f32>, tint:vec4<f32>, time:f32 }
        @group(0) @binding(0) var<uniform>      draw: Draw;
        @group(0) @binding(1) var<storage,read> intensities: array<f32>;
        @group(0) @binding(2) var<storage,read> neuronProps: array<f32>;

        struct In  { @location(0) pos:vec3<f32>, @location(1) owner:f32 }
        struct Out { @builtin(position) pos:vec4<f32>, @location(0) color:vec4<f32> }

        fn typeColor(tid: f32) -> vec3<f32> {
          let t=i32(tid);
          if(t==1){return vec3<f32>(0.30,0.81,0.43);}  
          if(t==2){return vec3<f32>(0.19,0.84,0.78);}  
          if(t==3){return vec3<f32>(0.63,0.40,0.87);}  
          if(t==4){return vec3<f32>(0.96,0.61,0.17);}  
          if(t==5){return vec3<f32>(0.96,0.90,0.26);}  
          return vec3<f32>(0.29,0.56,0.89);            
        }

        @vertex
        fn vs_main(input:In)->Out{
          var out:Out;
          let idx = u32(input.owner);
          let pulse = clamp(intensities[idx],0.0,1.0);
          let typeId = neuronProps[idx*4u];
          let noiseStd = neuronProps[idx*4u+1u];
          let dropProb = neuronProps[idx*4u+2u];
          let adaptRate = neuronProps[idx*4u+3u];
          
          var col = mix(typeColor(typeId), vec3<f32>(1.0, 0.76, 0.28), pulse * 0.8);
          col += vec3<f32>(1.0, 0.76, 0.28) * (pulse*pulse) * 0.9;
          col += vec3<f32>(noiseStd * 0.4 * sin(draw.time * 8.0 + f32(idx) * 1.9));
          
          let adaptDim = 1.0 - clamp(adaptRate*0.4, 0.0, 0.5);
          col *= adaptDim;
          let grey = (col.r+col.g+col.b)/3.0;
          col = mix(col, vec3<f32>(grey), dropProb*0.5);
          
          let alpha = mix(draw.tint.w, 1.0, pulse * 0.6) * adaptDim;
          out.pos = draw.mvp * vec4<f32>(input.pos, 1.0);
          out.color = vec4<f32>(col, alpha);
          return out;
        }

        @fragment fn fs_main(input:Out)->@location(0) vec4<f32> { return input.color; }
      `});

      const computeBGL = device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
        {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
        {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
        {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      ]});

      const drawBGL = device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},
        {binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},
        {binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},
      ]});

      const computeBG = device.createBindGroup({layout:computeBGL, entries:[
        {binding:0,resource:{buffer:simUniBuf}},
        {binding:1,resource:{buffer:spikeBuf}},
        {binding:2,resource:{buffer:intensityBuf}},
        {binding:3,resource:{buffer:propsBuf}},
      ]});

      const edgeDrawBG = device.createBindGroup({layout:drawBGL, entries:[
        {binding:0,resource:{buffer:edgeUniBuf}},
        {binding:1,resource:{buffer:intensityBuf}},
        {binding:2,resource:{buffer:propsBuf}},
      ]});

      const neuronDrawBG = device.createBindGroup({layout:drawBGL, entries:[
        {binding:0,resource:{buffer:neuronUniBuf}},
        {binding:1,resource:{buffer:intensityBuf}},
        {binding:2,resource:{buffer:propsBuf}},
      ]});

      const computePipe = device.createComputePipeline({
        layout:device.createPipelineLayout({bindGroupLayouts:[computeBGL]}),
        compute:{module:computeMod, entryPoint:"main"}
      });

      const renderPipe = device.createRenderPipeline({
        layout:device.createPipelineLayout({bindGroupLayouts:[drawBGL]}),
        vertex:{module:drawMod, entryPoint:"vs_main", buffers:[{arrayStride:16,attributes:[
          {shaderLocation:0,offset:0,format:"float32x3"}, {shaderLocation:1,offset:12,format:"float32"}
        ]}]},
        fragment:{module:drawMod,entryPoint:"fs_main",targets:[{
          format, blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},
                         alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}
        }]},
        primitive:{topology:"line-list"}
      });

      let step = 0, accMs = 0, lastTs = 0;
      const stepDurationMs = Math.max(16, payload!.dt * 1000);

      const frame = (ts: number) => {
        if (isCleanedUp) return;
        if (!lastTs) lastTs = ts;
        const deltaMs = ts - lastTs; lastTs = ts;
        const timeSec = ts * 0.001;

        const ex = Math.cos(yaw)*Math.cos(pitch)*radius;
        const ey = Math.sin(pitch)*radius;
        const ez = Math.sin(yaw)*Math.cos(pitch)*radius;
        const proj = perspective(Math.PI/3, canvas.width/Math.max(1,canvas.height), 0.1, 2000);
        const view = lookAt([ex,ey,ez],[0,0,0],[0,1,0]);
        const mvp = transpose(matMul(proj,view));

        if (!pausedRef.current) {
          accMs += deltaMs * speedRef.current;
          const adv = Math.floor(accMs / stepDurationMs);
          if (adv > 0) {
             step = (step + adv) % payload!.stepCount; 
             accMs -= adv * stepDurationMs; 
             setStats({ step, dt: payload!.dt });
          }
        }

        const bSim = new ArrayBuffer(16), vSim = new DataView(bSim);
        vSim.setUint32(0, step, true); vSim.setUint32(4, payload!.neuronCount, true);
        vSim.setFloat32(8, payload!.decay, true); vSim.setFloat32(12, payload!.boost, true);
        device.queue.writeBuffer(simUniBuf, 0, bSim);

        const encoder = device.createCommandEncoder();
        const cpass = encoder.beginComputePass();
        cpass.setPipeline(computePipe);
        cpass.setBindGroup(0, computeBG);
        cpass.dispatchWorkgroups(Math.ceil(payload!.neuronCount/64));
        cpass.end();

        const pass = encoder.beginRenderPass({
          colorAttachments:[{view:ctx.getCurrentTexture().createView(), clearValue:{r:0.027,g:0.035,b:0.059,a:1}, loadOp:"clear", storeOp:"store"}]
        });
        pass.setPipeline(renderPipe);

        const writeDraw = (buf: GPUBuffer, r:number, g:number, b:number, a:number) => {
          const d = new Float32Array(20); d.set(mvp,0); d[16]=r;d[17]=g;d[18]=b;d[19]=a;
          device.queue.writeBuffer(buf,0,d);
          device.queue.writeBuffer(buf,80,new Float32Array([timeSec,0,0,0]));
        };

        writeDraw(edgeUniBuf, 0.10, 0.42, 0.80, 0.45);
        pass.setBindGroup(0, edgeDrawBG); pass.setVertexBuffer(0, edgeBuf); pass.draw(edgeVerts.length/4);

        writeDraw(neuronUniBuf, 0.75, 0.92, 1.00, 0.90);
        pass.setBindGroup(0, neuronDrawBG); pass.setVertexBuffer(0, neuronBuf); pass.draw(neuronVerts.length/4);

        pass.end();
        device.queue.submit([encoder.finish()]);

        animationFrameId = requestAnimationFrame(frame);
      };
      
      animationFrameId = requestAnimationFrame(frame);

      return () => {
        isCleanedUp = true;
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener("resize", resize);
        canvas.removeEventListener("pointerdown", onPointerDown as EventListener);
        canvas.removeEventListener("pointerup", onPointerUp as EventListener);
        canvas.removeEventListener("pointercancel", onPointerUp as EventListener);
        canvas.removeEventListener("pointermove", onPointerMove as EventListener);
        canvas.removeEventListener("wheel", onWheel);
        canvas.removeEventListener("keydown", onKeyDown);
      };
    }
    
    const cleanupPromise = initGPU();
    return () => { isCleanedUp = true; cleanupPromise.then(fn => fn && fn()); };
  }, [payload]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0a0d16', overflow: 'hidden' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block', outline: 'none' }} />
      {error && (
        <div style={{ position: 'absolute', top: 20, left: 20, color: '#ff8080', background: 'rgba(20,10,10,0.8)', padding: 12, borderRadius: 8 }}>
          {error}
        </div>
      )}
      {payload && (
        <div style={{
          position: 'absolute', top: 20, left: 20, background: 'rgba(10, 15, 30, 0.85)', 
          backdropFilter: 'blur(10px)', border: '1px solid rgba(80, 110, 200, 0.3)', 
          borderRadius: 16, padding: 16, color: '#e0e6f0', width: 280,
          fontFamily: 'Inter, system-ui, sans-serif'
        }}>
          <h2 style={{ margin: '0 0 10px 0', fontSize: 14, fontWeight: 600, color: '#c8d8ff' }}>Neurogen Unicorn View</h2>
          <div style={{ fontSize: 12, color: '#8ea4c8', marginBottom: 16 }}>
            <div><span style={{color: '#d0e4ff'}}>{payload.neuronCount}</span> neurons</div>
            <div><span style={{color: '#d0e4ff'}}>{payload.synapseCount}</span> synapses</div>
            <div>Step: <span style={{color: '#d0e4ff'}}>{stats.step + 1} / {payload.stepCount}</span></div>
          </div>
          
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button 
              onClick={() => setPaused(!paused)}
              style={{
                background: paused ? 'rgba(80, 110, 200, 0.4)' : 'rgba(20, 30, 60, 0.8)',
                color: '#fff', border: '1px solid rgba(80, 110, 200, 0.4)', borderRadius: 8,
                padding: '6px 14px', cursor: 'pointer', fontSize: 12, fontWeight: 500
              }}>
              {paused ? 'Play' : 'Pause'}
            </button>
            <select 
              value={speed} onChange={e => setSpeed(Number(e.target.value))}
              style={{
                background: 'rgba(20, 30, 60, 0.8)', color: '#fff', 
                border: '1px solid rgba(80, 110, 200, 0.4)', borderRadius: 8,
                padding: '6px 10px', fontSize: 12, outline: 'none'
              }}>
              <option value="0.5">0.5x</option>
              <option value="1">1x</option>
              <option value="2">2x</option>
              <option value="4">4x</option>
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
