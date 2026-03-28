"""WebGPU neural spike preview generator — enhanced visualizer.

New features vs previous version:
  - Per-neuron activation-type colour coding (lif/relu/softplus/tanh/sigmoid/rbf)
  - Adaptation state: effective threshold rise dims the neuron's base colour
  - Dropout/noise indicator: noisy neurons pulse with a shimmer ring
  - Property-aware HUD: shows neuron-type breakdown and property legend
  - Improved camera: mouse-drag orbit + scroll zoom
  - Neuron size encodes its gain property (bigger = higher gain)
"""

import argparse
import json
from pathlib import Path

from backend.data_loader.json_loader import load_network
from backend.neuron_sim.framework_runner import run_simulation

# ── Activation type → integer id mapping (must match WGSL) ───────────────────
_ACTIVATION_IDS = {
    "lif": 0, "relu": 1, "softplus": 2,
    "tanh": 3, "sigmoid": 4, "rbf": 5,
}

# ── Colour legend for HUD (RGBA components 0-255) ────────────────────────────
_TYPE_COLOURS_CSS = {
    "lif":      "#4a90e2",  # cool blue
    "relu":     "#4ecf6e",  # green
    "softplus": "#30d5c8",  # cyan
    "tanh":     "#a066dd",  # purple
    "sigmoid":  "#f59c2a",  # orange
    "rbf":      "#f5e642",  # yellow
}

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Unicorn WebGPU Preview</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif;
           background: #07090f; color: #e0e6f0; overflow: hidden; }

    /* ── HUD ──────────────────────────────────────────────────────────── */
    #hud {
      position: fixed; left: 14px; top: 14px; z-index: 10;
      background: rgba(8,12,28,.90);
      padding: 12px 14px; border-radius: 12px;
      border: 1px solid rgba(80,110,200,.35);
      backdrop-filter: blur(6px);
      max-width: 260px;
      box-shadow: 0 4px 24px rgba(0,0,0,.5);
    }
    #hud h1 { margin: 0 0 6px 0; font-size: 13px; font-weight: 600;
              letter-spacing: .04em; color: #c8d8ff; }
    #hud p  { margin: 2px 0; font-size: 11px; color: #8ea4c8; line-height: 1.5; }
    #hud .val { color: #d0e4ff; font-variant-numeric: tabular-nums; }

    /* ── Type legend ──────────────────────────────────────────────────── */
    #legend { margin-top: 8px; }
    .legend-row { display: flex; align-items: center; gap: 6px;
                  font-size: 10px; color: #7a90b8; padding: 1px 0; }
    .legend-dot { width: 8px; height: 8px; border-radius: 50%;
                  flex-shrink: 0; box-shadow: 0 0 4px currentColor; }

    /* ── Controls ─────────────────────────────────────────────────────── */
    #controls { margin-top: 10px; display: flex; align-items: center;
                gap: 6px; flex-wrap: wrap; }
    #controls button, #controls select {
      background: rgba(20,30,60,.8);
      color: #c8d8ff;
      border: 1px solid rgba(80,110,200,.4);
      border-radius: 7px;
      padding: 3px 8px;
      font-size: 11px;
      cursor: pointer;
      transition: background .15s;
    }
    #controls button:hover { background: rgba(40,60,120,.9); }

    /* ── Status messages ──────────────────────────────────────────────── */
    #gpu-error    { color: #ff8080; display: none; font-size: 11px; margin-top:4px; }
    #collab-error { color: #ff8080; display: none; font-size: 11px; }

    /* ── Canvas ───────────────────────────────────────────────────────── */
    canvas { width: 100vw; height: 100vh; display: block; cursor: grab; }
    canvas:active { cursor: grabbing; }
  </style>
</head>
<body>
  <div id="hud">
    <h1>⬡ Unicorn WebGPU Preview</h1>
    <p id="stats"></p>
    <p id="time"></p>
    <p id="gpu-error"></p>
    <div id="legend"></div>
    <div id="controls">
      <button id="play-toggle">Pause</button>
      <label for="speed" style="font-size:11px;color:#8ea4c8">Speed</label>
      <select id="speed">
        <option value="0.5">½×</option>
        <option value="1" selected>1×</option>
        <option value="2">2×</option>
        <option value="4">4×</option>
      </select>
    </div>
  </div>
  <canvas id="gpu-canvas"></canvas>
  <script type="module">
  const payload = __PAYLOAD__;

  // ── Type colour lookup (RGBA floats) for WGSL uniform ─────────────────────
  // Index = activation_fn id: 0=lif 1=relu 2=softplus 3=tanh 4=sigmoid 5=rbf
  const TYPE_COLOURS = [
    [0.29, 0.56, 0.89, 1.0],   // lif      – cool blue
    [0.30, 0.81, 0.43, 1.0],   // relu     – green
    [0.19, 0.84, 0.78, 1.0],   // softplus – cyan
    [0.63, 0.40, 0.87, 1.0],   // tanh     – purple
    [0.96, 0.61, 0.17, 1.0],   // sigmoid  – orange
    [0.96, 0.90, 0.26, 1.0],   // rbf      – yellow
  ];

  const TYPE_NAMES    = ["LIF","ReLU","Softplus","Tanh","Sigmoid","RBF"];
  const TYPE_CSS      = ["#4a90e2","#4ecf6e","#30d5c8","#a066dd","#f59c2a","#f5e642"];

  // ── Matrix helpers ─────────────────────────────────────────────────────────
  function matMul(a, b) {
    const o = new Float32Array(16);
    for (let r=0;r<4;r++)
      for (let c=0;c<4;c++) {
        let s=0;
        for (let k=0;k<4;k++) s+=a[r*4+k]*b[k*4+c];
        o[r*4+c]=s;
      }
    return o;
  }
  function transpose(m) {
    return new Float32Array([
      m[0],m[4],m[8], m[12],
      m[1],m[5],m[9], m[13],
      m[2],m[6],m[10],m[14],
      m[3],m[7],m[11],m[15],
    ]);
  }
  function perspective(fovy,aspect,near,far) {
    const f=1.0/Math.tan(fovy/2), nf=1.0/(near-far);
    return new Float32Array([
      f/aspect,0,0,0,  0,f,0,0,
      0,0,(far+near)*nf,2*far*near*nf,
      0,0,-1,0,
    ]);
  }
  function lookAt(eye,target,up) {
    let fx=target[0]-eye[0],fy=target[1]-eye[1],fz=target[2]-eye[2];
    const fL=Math.hypot(fx,fy,fz)||1; fx/=fL;fy/=fL;fz/=fL;
    let rx=fy*up[2]-fz*up[1],ry=fz*up[0]-fx*up[2],rz=fx*up[1]-fy*up[0];
    const rL=Math.hypot(rx,ry,rz)||1; rx/=rL;ry/=rL;rz/=rL;
    const ux=ry*fz-rz*fy,uy=rz*fx-rx*fz,uz=rx*fy-ry*fx;
    return new Float32Array([
       rx, ry, rz,-(rx*eye[0]+ry*eye[1]+rz*eye[2]),
       ux, uy, uz,-(ux*eye[0]+uy*eye[1]+uz*eye[2]),
      -fx,-fy,-fz, (fx*eye[0]+fy*eye[1]+fz*eye[2]),
       0,  0,  0,  1,
    ]);
  }
  function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v));}

  // ── Build HUD legend from neuronProps ─────────────────────────────────────
  function buildLegend(neuronProps) {
    const legend = document.getElementById("legend");
    legend.innerHTML = "";
    const counts = new Array(6).fill(0);
    for (const p of neuronProps) counts[p[0]]++;
    for (let i=0;i<6;i++) {
      if (!counts[i]) continue;
      const row = document.createElement("div");
      row.className = "legend-row";
      row.style.color = TYPE_CSS[i];
      const dot = document.createElement("div");
      dot.className = "legend-dot";
      dot.style.background = TYPE_CSS[i];
      dot.style.color = TYPE_CSS[i];
      row.appendChild(dot);
      const spec = [];
      if (neuronProps.find((p,idx)=>p[0]===i&&p[2]>0)) spec.push("dropout");
      if (neuronProps.find((p,idx)=>p[0]===i&&p[1]>0)) spec.push("noise");
      if (neuronProps.find((p,idx)=>p[0]===i&&p[3]>0)) spec.push("adapt");
      const specStr = spec.length ? ` <span style="opacity:.6">(${spec.join(",")})</span>` : "";
      row.innerHTML += `${TYPE_NAMES[i]} ×${counts[i]}${specStr}`;
      legend.appendChild(row);
    }
  }

  // ── Geometry builder ──────────────────────────────────────────────────────
  function buildGeometryFromScene(sceneGraph) {
    const neurons  = [...(sceneGraph.neurons||[])].sort((a,b)=>a.id-b.id);
    const synapses = sceneGraph.synapses||[];
    const byId     = new Map(neurons.map(n=>[n.id,n.position]));
    const idToIndex= new Map(neurons.map((n,i)=>[n.id,i]));

    const xs=neurons.map(n=>n.position[0]);
    const ys=neurons.map(n=>n.position[1]);
    const zs=neurons.map(n=>n.position[2]);
    const span=Math.max(
      (Math.max(...xs)-Math.min(...xs))||0,
      (Math.max(...ys)-Math.min(...ys))||0,
      (Math.max(...zs)-Math.min(...zs))||0, 1);
    const baseArm = span*0.015;

    const neuronVerts=[];
    const neuronProps = payload.neuronProps||[];   // [typeId, noisStd, dropProb, adaptRate]

    for (const node of neurons) {
      const [x,y,z]=node.position;
      const owner=idToIndex.get(node.id);
      // Arm size scaled by gain (prop[4] if present, else 1.0)
      const props = neuronProps[owner] || [0,0,0,0,1.0];
      const gain  = props[4] || 1.0;
      const arm   = baseArm * Math.min(Math.max(gain,0.5), 3.0);
      neuronVerts.push(x-arm,y,z,owner, x+arm,y,z,owner);
      neuronVerts.push(x,y-arm,z,owner, x,y+arm,z,owner);
      neuronVerts.push(x,y,z-arm,owner, x,y,z+arm,owner);
    }

    const edgeVerts=[];
    for (const syn of synapses) {
      if (!byId.has(syn.from)||!byId.has(syn.to)) continue;
      const src=byId.get(syn.from), dst=byId.get(syn.to);
      const owner=idToIndex.get(syn.from)??0;
      edgeVerts.push(src[0],src[1],src[2],owner, dst[0],dst[1],dst[2],owner);
    }

    return {
      neuronCount:neurons.length, synapseCount:synapses.length,
      neuronVertices:new Float32Array(neuronVerts),
      edgeVertices:new Float32Array(edgeVerts),
      idToIndex,
    };
  }

  // ── Snapshot interpolator ─────────────────────────────────────────────────
  class SnapshotInterpolator {
    constructor(n){this.n=n;this.prev=null;this.next=null;}
    ingest(s){this.prev=this.next;this.next=s;}
    sample(now){
      if(!this.next)return null;
      if(!this.prev)return this.next.spikes;
      const dt=Math.max(1,this.next.time-this.prev.time);
      const a=clamp((now-this.prev.time)/dt,0,1);
      const b=new Float32Array(this.n);
      for(let i=0;i<this.n;i++) b[i]=(this.prev.spikes[i]||0)*(1-a)+(this.next.spikes[i]||0)*a;
      return b;
    }
  }

  // ── Collaboration (unchanged) ─────────────────────────────────────────────
  class SpikeChannel {
    constructor(doc,awareness,sessionId,role){
      this.doc=doc;this.awareness=awareness;this.sessionId=sessionId;this.role=role;
      this.localId=String(awareness.clientID);this.peers=new Map();this.onSnapshot=null;
      this.inbox=doc.getMap("spikeInbox");
      this.inbox.observeDeep(()=>this._consumeInbox());this._ensureInbox();
      awareness.on("change",()=>this._syncPeers());this._syncPeers();
    }
    _ensureInbox(){if(!this.inbox.get(this.localId))this.inbox.set(this.localId,[]);}
    _appendInbox(t,m){const k=String(t),q=this.inbox.get(k)||[];q.push(m);this.inbox.set(k,q.slice(-200));}
    _sendSignal(t,p){this._appendInbox(t,{kind:"signal",from:this.localId,payload:p,time:Date.now()});}
    _consumeInbox(){
      const q=this.inbox.get(this.localId)||[];
      const unseen=q.filter(i=>!i._seenBy||!i._seenBy.includes(this.localId));
      if(!unseen.length)return;
      const marked=q.map(i=>{const s=new Set(i._seenBy||[]);s.add(this.localId);return{...i,_seenBy:[...s]};});
      this.inbox.set(this.localId,marked.slice(-200));
      for(const m of unseen){if(m.kind!=="signal"||!m.from)continue;this._onSignal(m.from,m.payload);}
    }
    _peerIds(){return[...this.awareness.getStates().keys()].map(id=>String(id)).filter(id=>id!==this.localId);}
    _syncPeers(){
      const peerIds=this._peerIds();
      for(const p of peerIds){if(!this.peers.has(p)&&this.localId<p)this._createPeer(p,true);}
      for(const k of[...this.peers.keys()]){if(!peerIds.includes(k)){this.peers.get(k)?.pc?.close();this.peers.delete(k);}}
    }
    _createPeer(remoteId,initiator){
      const pc=new RTCPeerConnection({iceServers:[{urls:["stun:stun.l.google.com:19302"]}]});
      const peer={pc,channel:null,ready:false,initiator};this.peers.set(remoteId,peer);
      pc.onicecandidate=e=>{if(e.candidate)this._sendSignal(remoteId,{type:"candidate",candidate:e.candidate});};
      if(initiator){
        const ch=pc.createDataChannel("spike-snapshots",{ordered:false,maxRetransmits:0});
        this._bindChannel(remoteId,ch);
        pc.createOffer().then(o=>pc.setLocalDescription(o)).then(()=>this._sendSignal(remoteId,{type:"offer",sdp:pc.localDescription})).catch(console.error);
      } else {pc.ondatachannel=e=>this._bindChannel(remoteId,e.channel);}
    }
    _bindChannel(remoteId,ch){
      const peer=this.peers.get(remoteId);if(!peer)return;peer.channel=ch;
      ch.onopen=()=>{peer.ready=true;};ch.onclose=()=>{peer.ready=false;};
      ch.onmessage=e=>{if(!this.onSnapshot)return;try{const m=JSON.parse(e.data);if(m.kind==="snapshot")this.onSnapshot(m.payload);}catch(_){}};
    }
    async _onSignal(remoteId,signal){
      let peer=this.peers.get(remoteId);if(!peer){this._createPeer(remoteId,false);peer=this.peers.get(remoteId);}
      const pc=peer.pc;
      if(signal.type==="offer"){await pc.setRemoteDescription(new RTCSessionDescription(signal.sdp));const a=await pc.createAnswer();await pc.setLocalDescription(a);this._sendSignal(remoteId,{type:"answer",sdp:pc.localDescription});}
      else if(signal.type==="answer"){await pc.setRemoteDescription(new RTCSessionDescription(signal.sdp));}
      else if(signal.type==="candidate"){try{await pc.addIceCandidate(new RTCIceCandidate(signal.candidate));}catch(_){}}
    }
    broadcast(snapshot){const m=JSON.stringify({kind:"snapshot",payload:snapshot});for(const peer of this.peers.values())if(peer.ready&&peer.channel)peer.channel.send(m);}
  }

  async function initCollaboration(sceneGraph,onSceneGraph){
    const params=new URLSearchParams(window.location.search);
    const session=params.get("session");if(!session)return null;
    const role=params.get("role")||"host";
    const collabError=document.getElementById("collab-error");
    try{
      const Y=await import("https://cdn.jsdelivr.net/npm/yjs@13.6.18/+esm");
      const{WebrtcProvider}=await import("https://cdn.jsdelivr.net/npm/y-webrtc@10.3.0/+esm");
      const doc=new Y.Doc(),provider=new WebrtcProvider(`unicorn-structure-${session}`,doc);
      const sharedScene=doc.getMap("sceneGraph");
      if(role==="host")sharedScene.set("graph",sceneGraph);
      sharedScene.observe(()=>{const g=sharedScene.get("graph");if(g)onSceneGraph(g);});
      const spikeChannel=new SpikeChannel(doc,provider.awareness,session,role);
      return{role,sharedScene,spikeChannel};
    }catch(err){collabError.style.display="block";collabError.textContent=`Collaboration disabled: ${err.message}`;return null;}
  }

  // ── Main ──────────────────────────────────────────────────────────────────
  async function main() {
    const stats      = document.getElementById("stats");
    const timeLabel  = document.getElementById("time");
    const gpuError   = document.getElementById("gpu-error");
    const playToggle = document.getElementById("play-toggle");
    const speedSelect= document.getElementById("speed");

    stats.innerHTML =
      `<span class="val">${payload.neuronCount}</span> neurons &nbsp;·&nbsp; `+
      `<span class="val">${payload.synapseCount}</span> synapses &nbsp;·&nbsp; `+
      `<span class="val">${payload.stepCount}</span> steps`;

    // ── Legend ───────────────────────────────────────────────────────────────
    const neuronProps = payload.neuronProps || [];
    buildLegend(neuronProps);

    if (!navigator.gpu) {
      gpuError.style.display="block";
      gpuError.textContent="WebGPU unavailable. Try Chrome 113+ or Edge 113+.";
      return;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      gpuError.style.display="block";
      gpuError.textContent="No compatible GPU adapter found.";
      return;
    }
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("gpu-canvas");
    const ctx    = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();

    function resize() {
      const dpr = Math.max(1, window.devicePixelRatio||1);
      canvas.width  = Math.floor(canvas.clientWidth  * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      ctx.configure({device, format, alphaMode:"opaque"});
    }
    window.addEventListener("resize", resize);
    resize();

    // ── Camera state (mouse-drag orbit + scroll zoom) ─────────────────────
    let yaw   = 0.3, pitch = 0.25;
    let radius= payload.cameraRadius;
    let isDragging=false, lastMX=0, lastMY=0;
    canvas.addEventListener("mousedown", e=>{isDragging=true;lastMX=e.clientX;lastMY=e.clientY;});
    window.addEventListener("mouseup",   ()=>{isDragging=false;});
    window.addEventListener("mousemove", e=>{
      if(!isDragging)return;
      yaw   += (e.clientX-lastMX)*0.005;
      pitch  = clamp(pitch+(e.clientY-lastMY)*0.005, -1.4, 1.4);
      lastMX=e.clientX;lastMY=e.clientY;
    });
    canvas.addEventListener("wheel", e=>{
      radius = clamp(radius+e.deltaY*0.04, radius*0.3, radius*3);
      e.preventDefault();
    },{passive:false});

    // ── GPU buffers ───────────────────────────────────────────────────────
    let sceneGraph    = payload.sceneGraph;
    let geometry      = buildGeometryFromScene(sceneGraph);
    let neuronVerts   = geometry.neuronVertices;
    let edgeVerts     = geometry.edgeVertices;
    let neuronCount   = geometry.neuronCount;
    const stepCount   = payload.stepCount;
    const localSpikes = new Uint32Array(payload.spikes);
    let liveInterpolated = new Float32Array(neuronCount);
    let usingLiveSnapshots = false;

    // Per-neuron properties storage buffer: [typeId, noiseStd, dropProb, adaptRate, gain]
    // Packed as float32 × 5 per neuron for WGSL alignment.
    function buildNeuronPropsBuffer(props, n) {
      const data = new Float32Array(n * 4); // 4 floats: typeId, noiseStd, dropProb, adaptRate
      for (let i=0; i<n; i++) {
        const p = props[i] || [0,0,0,0,1];
        data[i*4+0] = p[0]; // activation type id
        data[i*4+1] = p[1]; // noise_std
        data[i*4+2] = p[2]; // dropout_prob
        data[i*4+3] = p[3]; // adaptation_rate
      }
      return data;
    }

    function createVertexBuffer(data) {
      const buf = device.createBuffer({size:Math.max(data.byteLength,4),
        usage:GPUBufferUsage.VERTEX, mappedAtCreation:true});
      if(data.byteLength>0) new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap(); return buf;
    }
    function createStorageBuffer(data) {
      const buf = device.createBuffer({size:Math.max(data.byteLength,4),
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST, mappedAtCreation:true});
      new Float32Array(buf.getMappedRange()).set(data); buf.unmap(); return buf;
    }

    let neuronBuffer    = createVertexBuffer(neuronVerts);
    let edgeBuffer      = createVertexBuffer(edgeVerts);
    let spikeBuffer     = device.createBuffer({size:Math.max(localSpikes.byteLength,4),
                           usage:GPUBufferUsage.STORAGE, mappedAtCreation:true});
    new Uint32Array(spikeBuffer.getMappedRange()).set(localSpikes); spikeBuffer.unmap();

    let intensityBuffer  = device.createBuffer({
      size:Math.max(neuronCount,1)*4,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(intensityBuffer,0,new Float32Array(Math.max(neuronCount,1)));

    const propsData = buildNeuronPropsBuffer(neuronProps, Math.max(neuronCount,1));
    let   neuronPropsBuffer = createStorageBuffer(propsData);

    // Per-pass uniforms: 16-byte sim + 80-byte draw
    const simUniformBuffer    = device.createBuffer({size:16, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const edgeUniformBuffer   = device.createBuffer({size:96, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const neuronUniformBuffer = device.createBuffer({size:96, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});

    function writeSimUniform(stepVal, nCount, decay, boost) {
      const b=new ArrayBuffer(16), v=new DataView(b);
      v.setUint32(0,stepVal,true); v.setUint32(4,nCount,true);
      v.setFloat32(8,decay,true);  v.setFloat32(12,boost,true);
      device.queue.writeBuffer(simUniformBuffer,0,b);
    }

    // ── WGSL shaders ─────────────────────────────────────────────────────
    const computeModule = device.createShaderModule({code:`
      struct Sim { step:u32, neuronCount:u32, decay:f32, boost:f32 }
      @group(0) @binding(0) var<uniform>  sim:        Sim;
      @group(0) @binding(1) var<storage,read>       spikes:     array<u32>;
      @group(0) @binding(2) var<storage,read_write> intensities:array<f32>;
      @group(0) @binding(3) var<storage,read>       neuronProps:array<f32>; // 4 floats per neuron

      // Minimal hash for per-neuron per-step pseudo-random number (0..1)
      fn hash(seed:u32)->f32{
        var s=seed;
        s=(s^61u)^(s>>16u); s*=9u; s^=s>>4u; s*=668265261u; s^=s>>15u;
        return f32(s&0xFFFFFFu)/16777215.0;
      }

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid:vec3<u32>){
        let i=gid.x;
        if(i>=sim.neuronCount){return;}

        let propBase = i*4u;
        let dropProb = neuronProps[propBase+2u];  // dropout_prob

        // Apply dropout: if hash < dropout_prob, silence this neuron this step
        let rng = hash(sim.step*sim.neuronCount+i);
        if(dropProb>0.0 && rng<dropProb){
          intensities[i]=intensities[i]*sim.decay;
          return;
        }

        let idx  = sim.step*sim.neuronCount+i;
        let spike= f32(spikes[idx]);
        let decayed=intensities[i]*sim.decay;
        intensities[i]=max(decayed, spike*sim.boost);
      }`});

    // Draw shader: uses per-neuron type colour + glow + noise shimmer
    const drawModule = device.createShaderModule({code:`
      struct Draw { mvp:mat4x4<f32>, tint:vec4<f32>, time:f32, _pad:f32, _pad2:f32, _pad3:f32 }
      @group(0) @binding(0) var<uniform>      draw:       Draw;
      @group(0) @binding(1) var<storage,read> intensities:array<f32>;
      @group(0) @binding(2) var<storage,read> neuronProps:array<f32>;

      struct In  { @location(0) pos:vec3<f32>, @location(1) owner:f32 }
      struct Out { @builtin(position) pos:vec4<f32>, @location(0) color:vec4<f32> }

      // Activation-type colour palette (matches JavaScript TYPE_COLOURS)
      fn typeColour(tid:f32)->vec3<f32>{
        let t=i32(tid);
        if(t==1){return vec3<f32>(0.30,0.81,0.43);}  // relu  – green
        if(t==2){return vec3<f32>(0.19,0.84,0.78);}  // softplus – cyan
        if(t==3){return vec3<f32>(0.63,0.40,0.87);}  // tanh  – purple
        if(t==4){return vec3<f32>(0.96,0.61,0.17);}  // sigmoid – orange
        if(t==5){return vec3<f32>(0.96,0.90,0.26);}  // rbf   – yellow
        return vec3<f32>(0.29,0.56,0.89);             // lif   – cool blue (default)
      }

      @vertex
      fn vs_main(input:In)->Out{
        var out:Out;
        let idx      = u32(input.owner);
        let pulse    = clamp(intensities[idx],0.0,1.0);
        let glow     = pulse*pulse;
        let propBase = idx*4u;
        let typeId   = neuronProps[propBase+0u];
        let noiseStd = neuronProps[propBase+1u];
        let dropProb = neuronProps[propBase+2u];
        let adaptRate= neuronProps[propBase+3u];

        // Base colour from neuron type, brightened by pulse
        let cool = typeColour(typeId);
        let warm = vec3<f32>(1.0, 0.76, 0.28);
        var col  = mix(cool, warm, pulse*0.8);

        // Glow halo on spike
        col += warm*glow*0.9;

        // Noise shimmer: oscillate brightness for noisy neurons
        let shimmer = noiseStd*0.4*sin(draw.time*8.0+f32(idx)*1.9);
        col += vec3<f32>(shimmer);

        // Adaptation dimming: adapt suppresses brightness
        let adaptDim = 1.0 - clamp(adaptRate*0.4, 0.0, 0.5);
        col *= adaptDim;

        // Dropout desaturation: dropout neurons become grey-tinted
        let grey = (col.r+col.g+col.b)/3.0;
        col = mix(col, vec3<f32>(grey), dropProb*0.5);

        let alpha = mix(draw.tint.w, 1.0, pulse*0.6)*adaptDim;
        out.pos   = draw.mvp*vec4<f32>(input.pos,1.0);
        out.color = vec4<f32>(col, alpha);
        return out;
      }
      @fragment
      fn fs_main(input:Out)->@location(0) vec4<f32>{ return input.color; }`});

    // Fallback (no storage buffers)
    const drawFallbackModule = device.createShaderModule({code:`
      struct Draw { mvp:mat4x4<f32>, tint:vec4<f32>, time:f32, _pad:f32, _pad2:f32, _pad3:f32 }
      @group(0) @binding(0) var<uniform> draw:Draw;
      struct In  { @location(0) pos:vec3<f32>, @location(1) owner:f32 }
      struct Out { @builtin(position) pos:vec4<f32>, @location(0) color:vec4<f32> }
      @vertex fn vs_main(input:In)->Out{
        var out:Out;
        out.pos=draw.mvp*vec4<f32>(input.pos,1.0);
        out.color=draw.tint; return out;
      }
      @fragment fn fs_main(input:Out)->@location(0) vec4<f32>{ return input.color; }`});

    // ── Bind group layouts ────────────────────────────────────────────────
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
    const drawFallbackBGL = device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},
    ]});

    function makeComputeBG() {
      return device.createBindGroup({layout:computeBGL, entries:[
        {binding:0,resource:{buffer:simUniformBuffer}},
        {binding:1,resource:{buffer:spikeBuffer}},
        {binding:2,resource:{buffer:intensityBuffer}},
        {binding:3,resource:{buffer:neuronPropsBuffer}},
      ]});
    }
    function makeDrawBG(uniformBuf) {
      return device.createBindGroup({layout:drawBGL, entries:[
        {binding:0,resource:{buffer:uniformBuf}},
        {binding:1,resource:{buffer:intensityBuffer}},
        {binding:2,resource:{buffer:neuronPropsBuffer}},
      ]});
    }
    function makeFallbackBG(uniformBuf) {
      return device.createBindGroup({layout:drawFallbackBGL, entries:[
        {binding:0,resource:{buffer:uniformBuf}},
      ]});
    }

    let computeBG  = makeComputeBG();
    let edgeDrawBG = null, neuronDrawBG = null;
    let activeBGL  = drawBGL;
    let usePulse   = true;

    try {
      edgeDrawBG   = makeDrawBG(edgeUniformBuffer);
      neuronDrawBG = makeDrawBG(neuronUniformBuffer);
    } catch (_err) {
      usePulse    = false;
      activeBGL   = drawFallbackBGL;
      edgeDrawBG  = makeFallbackBG(edgeUniformBuffer);
      neuronDrawBG= makeFallbackBG(neuronUniformBuffer);
      gpuError.style.display = "block";
      gpuError.textContent   = "Limited mode: storage buffers unavailable; type colours disabled.";
    }

    const computePipeline = device.createComputePipeline({
      layout:device.createPipelineLayout({bindGroupLayouts:[computeBGL]}),
      compute:{module:computeModule, entryPoint:"main"},
    });
    const renderPipeline = device.createRenderPipeline({
      layout:device.createPipelineLayout({bindGroupLayouts:[activeBGL]}),
      vertex:{
        module:usePulse?drawModule:drawFallbackModule, entryPoint:"vs_main",
        buffers:[{arrayStride:16, attributes:[
          {shaderLocation:0,offset:0,format:"float32x3"},
          {shaderLocation:1,offset:12,format:"float32"},
        ]}],
      },
      fragment:{module:usePulse?drawModule:drawFallbackModule,entryPoint:"fs_main",targets:[{
        format,
        blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},
               alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}},
      }]},
      primitive:{topology:"line-list"},
    });

    // ── Rebuid everything when collab changes scene ───────────────────────
    function rebuildSceneBuffers(nextGraph) {
      sceneGraph = nextGraph;
      geometry   = buildGeometryFromScene(sceneGraph);
      neuronVerts= geometry.neuronVertices; edgeVerts = geometry.edgeVertices;
      neuronCount= geometry.neuronCount;   liveInterpolated=new Float32Array(neuronCount);
      neuronBuffer.destroy(); edgeBuffer.destroy(); intensityBuffer.destroy(); neuronPropsBuffer.destroy();
      neuronBuffer      = createVertexBuffer(neuronVerts);
      edgeBuffer        = createVertexBuffer(edgeVerts);
      intensityBuffer   = device.createBuffer({size:Math.max(neuronCount,1)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(intensityBuffer,0,new Float32Array(Math.max(neuronCount,1)));
      const pd = buildNeuronPropsBuffer(neuronProps,Math.max(neuronCount,1));
      neuronPropsBuffer = createStorageBuffer(pd);
      computeBG   = makeComputeBG();
      edgeDrawBG  = usePulse ? makeDrawBG(edgeUniformBuffer)   : makeFallbackBG(edgeUniformBuffer);
      neuronDrawBG= usePulse ? makeDrawBG(neuronUniformBuffer) : makeFallbackBG(neuronUniformBuffer);
      stats.innerHTML = `<span class="val">${neuronCount}</span> neurons &nbsp;·&nbsp; `+
        `<span class="val">${geometry.synapseCount}</span> synapses &nbsp;·&nbsp; `+
        `<span class="val">${stepCount}</span> steps`;
    }

    const collab      = await initCollaboration(sceneGraph, rebuildSceneBuffers);
    const interpolator= new SnapshotInterpolator(neuronCount);
    if (collab) {
      collab.spikeChannel.onSnapshot=(packet)=>{
        usingLiveSnapshots=true;
        if(!packet||!Array.isArray(packet.spikes))return;
        const sp=new Float32Array(packet.spikes.slice(0,neuronCount));
        interpolator.ingest({time:packet.time||performance.now(),spikes:sp});
      };
    }

    let step=0, paused=false, speed=1, accMs=0, lastTs=0;
    const stepDurationMs=Math.max(16,payload.dt*1000);
    playToggle.addEventListener("click",()=>{paused=!paused;playToggle.textContent=paused?"Play":"Pause";});
    speedSelect.addEventListener("change",()=>{speed=Number(speedSelect.value||1);});

    // ── Render loop ───────────────────────────────────────────────────────
    function frame(ts) {
      if(!lastTs) lastTs=ts;
      const deltaMs = ts-lastTs; lastTs=ts;
      const timeSec = ts*0.001;

      // Camera
      const ex = Math.cos(yaw)*Math.cos(pitch)*radius;
      const ey = Math.sin(pitch)*radius;
      const ez = Math.sin(yaw)*Math.cos(pitch)*radius;
      const proj= perspective(Math.PI/3, canvas.width/Math.max(1,canvas.height), 0.1, 2000);
      const view= lookAt([ex,ey,ez],[0,0,0],[0,1,0]);
      const mvp = transpose(matMul(proj,view));

      // Draw uniform: mvp (16f) + tint (4f) + time (1f) + pad (3f) = 80 bytes
      const drawData = new Float32Array(20);
      drawData.set(mvp,0);
      // tint at [16..19], time at a separate slot → we pack into drawData[16..19]
      // (time sent as uniform below via write; tint overwritten per pass)

      if(!paused){
        accMs += deltaMs*speed;
        const adv=Math.floor(accMs/stepDurationMs);
        if(adv>0){step=(step+adv)%stepCount; accMs-=adv*stepDurationMs;}
      }
      writeSimUniform(step,neuronCount,payload.decay,payload.boost);

      const encoder = device.createCommandEncoder();

      if(usingLiveSnapshots){
        const sp=interpolator.sample(ts);
        if(sp){
          for(let i=0;i<liveInterpolated.length;i++)
            liveInterpolated[i]=Math.max((liveInterpolated[i]||0)*payload.decay,sp[i]||0);
          device.queue.writeBuffer(intensityBuffer,0,liveInterpolated);
        }
      } else {
        const cpass=encoder.beginComputePass();
        cpass.setPipeline(computePipeline);
        cpass.setBindGroup(0,computeBG);
        cpass.dispatchWorkgroups(Math.ceil(Math.max(neuronCount,1)/64));
        cpass.end();
      }

      const textureView=ctx.getCurrentTexture().createView();
      const pass=encoder.beginRenderPass({
        colorAttachments:[{view:textureView,clearValue:{r:0.027,g:0.035,b:0.059,a:1},loadOp:"clear",storeOp:"store"}],
      });
      pass.setPipeline(renderPipeline);

      // Helper to write draw uniform with tint + time
      function writeDraw(buf, r,g,b,a) {
        const d=new Float32Array(20);
        d.set(mvp,0);
        d[16]=r; d[17]=g; d[18]=b; d[19]=a;
        device.queue.writeBuffer(buf,0,d);
        // time packed into bytes 80..84 (after the 80-byte struct)
        // — simpler: add time as part of the remaining 16 bytes (96-byte buffer)
        const t=new Float32Array([timeSec,0,0,0]);
        device.queue.writeBuffer(buf,80,t);
      }

      // Edges: dim blue, slightly transparent
      writeDraw(edgeUniformBuffer, 0.10, 0.42, 0.80, 0.45);
      pass.setBindGroup(0,edgeDrawBG);
      pass.setVertexBuffer(0,edgeBuffer);
      pass.draw(edgeVerts.length/4);

      // Neurons: type colours applied in shader; tint.w sets base alpha
      writeDraw(neuronUniformBuffer, 0.75, 0.92, 1.00, 0.90);
      pass.setBindGroup(0,neuronDrawBG);
      pass.setVertexBuffer(0,neuronBuffer);
      pass.draw(neuronVerts.length/4);

      pass.end();
      device.queue.submit([encoder.finish()]);

      // Broadcast spikes for collab
      if(collab?.role==="host"){
        const sf=[];
        for(let i=0;i<neuronCount;i++) sf.push(localSpikes[step*neuronCount+i]||0);
        collab.spikeChannel.broadcast({time:ts,step,spikes:sf});
      }

      timeLabel.innerHTML =
        `step <span class="val">${step+1}/${stepCount}</span> &nbsp;dt=<span class="val">${payload.dt.toFixed(3)}s</span>`;
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  main();
  </script>
</body>
</html>
"""


# ── Activation type id ────────────────────────────────────────────────────────
_ACTIVATION_IDS = {
    "lif": 0, "relu": 1, "softplus": 2,
    "tanh": 3, "sigmoid": 4, "rbf": 5,
}


def _build_vertex_buffers(network, layout):
    by_id = {item["id"]: item["position"] for item in layout}
    neuron_ids = [n["id"] for n in network["neurons"]]
    id_to_index = {nid: idx for idx, nid in enumerate(neuron_ids)}

    xs = [by_id[nid][0] for nid in neuron_ids]
    ys = [by_id[nid][1] for nid in neuron_ids]
    zs = [by_id[nid][2] for nid in neuron_ids]
    span          = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 1.0)
    base_arm      = span * 0.015
    camera_radius = span * 2.5

    # Build per-neuron property array for HUD and shader
    # [activation_fn_id, noise_std, dropout_prob, adaptation_rate, gain]
    neuron_props = []
    for neu in network["neurons"]:
        fn_id   = _ACTIVATION_IDS.get(neu.get("activation_fn", "lif"), 0)
        noise   = float(neu.get("noise_std", 0.0))
        dropout = float(neu.get("dropout_prob", 0.0))
        adapt   = float(neu.get("adaptation_rate", 0.0))
        gain    = float(neu.get("gain", 1.0))
        neuron_props.append([fn_id, noise, dropout, adapt, gain])

    neuron_vertices = []
    for nid in neuron_ids:
        x, y, z = by_id[nid]
        owner = float(id_to_index[nid])
        props = neuron_props[id_to_index[nid]]
        arm   = base_arm * min(max(props[4], 0.5), 3.0)
        neuron_vertices.extend([x-arm, y, z, owner, x+arm, y, z, owner])
        neuron_vertices.extend([x, y-arm, z, owner, x, y+arm, z, owner])
        neuron_vertices.extend([x, y, z-arm, owner, x, y, z+arm, owner])

    edge_vertices = []
    for syn in network.get("synapses", []):
        src_id = syn["from"]
        src_idx = float(id_to_index[src_id])
        x0, y0, z0 = by_id[src_id]
        x1, y1, z1 = by_id[syn["to"]]
        edge_vertices.extend([x0, y0, z0, src_idx, x1, y1, z1, src_idx])

    return neuron_ids, id_to_index, neuron_vertices, edge_vertices, camera_radius, neuron_props


def build_webgpu_payload(network, layout, history):
    neuron_ids, id_to_index, neuron_vertices, edge_vertices, camera_radius, neuron_props = (
        _build_vertex_buffers(network, layout)
    )

    spikes = []
    for step in history:
        frame = [0] * len(neuron_ids)
        spike_values = step.get("spikes", [])
        if isinstance(spike_values, dict):
            for nid, fired in spike_values.items():
                nid_int = int(nid)
                if fired and nid_int in id_to_index:
                    frame[id_to_index[nid_int]] = 1
        else:
            spike_list = list(spike_values)
            is_dense = len(spike_list) == len(neuron_ids) and all(
                v in (0, 1, 0.0, 1.0, False, True) for v in spike_list
            )
            if is_dense:
                for idx, fired in enumerate(spike_list):
                    if fired:
                        frame[id_to_index[neuron_ids[idx]]] = 1
            else:
                for nid in spike_list:
                    nid_int = int(nid)
                    if nid_int in id_to_index:
                        frame[id_to_index[nid_int]] = 1
        spikes.extend(frame)

    dt = (
        float(history[1]["time"] - history[0]["time"])
        if len(history) > 1 and "time" in history[0] and "time" in history[1]
        else float(network.get("dt", 1.0))
    )

    layout_by_id = {item["id"]: item["position"] for item in layout}
    scene_graph = {
        "neurons":  [{"id": nid, "position": layout_by_id[nid]}
                     for nid in neuron_ids if nid in layout_by_id],
        "synapses": [{"from": s["from"], "to": s["to"], "weight": s.get("weight", 0.0)}
                     for s in network.get("synapses", [])],
    }

    return {
        "neuronCount":   len(neuron_ids),
        "synapseCount":  len(network.get("synapses", [])),
        "stepCount":     len(history),
        "dt":            dt,
        "decay":         0.92,
        "boost":         1.0,
        "cameraRadius":  camera_radius,
        "neuronVertices": neuron_vertices,
        "edgeVertices":  edge_vertices,
        "spikes":        spikes,
        "neuronProps":   neuron_props,   # NEW: per-neuron property data
        "sceneGraph":    scene_graph,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an enhanced WebGPU neural spike preview HTML."
    )
    parser.add_argument("network", nargs="?", default="samples/network.json",
                        help="Path to Unicorn network JSON.")
    parser.add_argument("--layout", default="samples/layout_output.json",
                        help="Layout JSON path.")
    parser.add_argument("--history", default=None,
                        help="Optional spike history JSON. If omitted, simulation runs automatically.")
    parser.add_argument("--output", default="output/webgpu_preview.html",
                        help="Output HTML path.")
    return parser.parse_args()


def main():
    args = parse_args()
    network = load_network(args.network)
    with open(args.layout) as f:
        layout = json.load(f)

    if args.history:
        with open(args.history) as f:
            history = json.load(f)
    else:
        history = run_simulation(network)

    payload = build_webgpu_payload(network, layout, history)
    html    = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Saved WebGPU preview to {out}")


if __name__ == "__main__":
    main()
