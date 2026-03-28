import argparse
import json
from pathlib import Path

from backend.data_loader.json_loader import load_network
from backend.neuron_sim.framework_runner import run_simulation


HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Unicorn WebGPU Preview</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #0b1020; color: #e5e7eb; }
    #hud { position: fixed; left: 12px; top: 12px; z-index: 10; background: rgba(12,18,40,.85); padding: 10px 12px; border-radius: 10px; border: 1px solid #27304d; }
    #hud h1 { margin: 0 0 6px 0; font-size: 14px; }
    #hud p { margin: 2px 0; font-size: 12px; color: #aeb8d6; }
    #controls { margin-top: 8px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    #controls button, #controls select {
      background: #18213d;
      color: #e5e7eb;
      border: 1px solid #30406e;
      border-radius: 8px;
      padding: 4px 8px;
      font-size: 12px;
    }
    #gpu-error { color: #fca5a5; display: none; }
    #collab-error { color: #fca5a5; display: none; }
    canvas { width: 100vw; height: 100vh; display: block; }
  </style>
</head>
<body>
  <div id=\"hud\">
    <h1>Unicorn WebGPU Spike Preview</h1>
    <p id="stats"></p>
    <p id="time"></p>
    <p id="gpu-error"></p>
    <div id="controls">
      <button id="play-toggle">Pause</button>
      <label for="speed">Speed</label>
      <select id="speed">
        <option value="0.5">0.5×</option>
        <option value="1" selected>1×</option>
        <option value="2">2×</option>
        <option value="3">3×</option>
      </select>
    </div>
  </div>
  <canvas id=\"gpu-canvas\"></canvas>
  <script type=\"module\">
  const payload = __PAYLOAD__;

  function matMul(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) sum += a[k * 4 + r] * b[c * 4 + k];
        out[c * 4 + r] = sum;
      }
    }
    return out;
  }
  function perspective(fov, aspect, near, far) {
    const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
    return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,(2*far*near)*nf,0]);
  }
  function lookAt(eye, target, up) {
    const zx = eye[0]-target[0], zy = eye[1]-target[1], zz = eye[2]-target[2];
    const zLen = Math.hypot(zx,zy,zz) || 1;
    const zxN = zx/zLen, zyN = zy/zLen, zzN = zz/zLen;
    const xx = up[1]*zzN - up[2]*zyN, xy = up[2]*zxN - up[0]*zzN, xz = up[0]*zyN - up[1]*zxN;
    const xLen = Math.hypot(xx,xy,xz) || 1;
    const xxN = xx/xLen, xyN = xy/xLen, xzN = xz/xLen;
    const yx = zyN*xzN - zzN*xyN, yy = zzN*xxN - zxN*xzN, yz = zxN*xyN - zyN*xxN;
    return new Float32Array([
      xxN,yx,zxN,0, xyN,yy,zyN,0, xzN,yz,zzN,0,
      -(xxN*eye[0]+xyN*eye[1]+xzN*eye[2]),
      -(yx*eye[0]+yy*eye[1]+yz*eye[2]),
      -(zxN*eye[0]+zyN*eye[1]+zzN*eye[2]),1
    ]);
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function buildGeometryFromScene(sceneGraph) {
    const neurons = [...(sceneGraph.neurons || [])].sort((a, b) => a.id - b.id);
    const synapses = sceneGraph.synapses || [];
    const byId = new Map(neurons.map((node) => [node.id, node.position]));
    const idToIndex = new Map(neurons.map((node, index) => [node.id, index]));

    const xs = neurons.map((node) => node.position[0]);
    const ys = neurons.map((node) => node.position[1]);
    const zs = neurons.map((node) => node.position[2]);
    const span = Math.max((Math.max(...xs) - Math.min(...xs)) || 0, (Math.max(...ys) - Math.min(...ys)) || 0, (Math.max(...zs) - Math.min(...zs)) || 0, 1);
    const arm = span * 0.015;

    const neuronVertices = [];
    for (const node of neurons) {
      const [x, y, z] = node.position;
      const owner = idToIndex.get(node.id);
      neuronVertices.push(x - arm, y, z, owner, x + arm, y, z, owner);
      neuronVertices.push(x, y - arm, z, owner, x, y + arm, z, owner);
      neuronVertices.push(x, y, z - arm, owner, x, y, z + arm, owner);
    }

    const edgeVertices = [];
    for (const synapse of synapses) {
      if (!byId.has(synapse.from) || !byId.has(synapse.to)) continue;
      const src = byId.get(synapse.from);
      const dst = byId.get(synapse.to);
      const owner = idToIndex.get(synapse.from) ?? 0;
      edgeVertices.push(src[0], src[1], src[2], owner, dst[0], dst[1], dst[2], owner);
    }

    return {
      neuronCount: neurons.length,
      synapseCount: synapses.length,
      neuronVertices: new Float32Array(neuronVertices),
      edgeVertices: new Float32Array(edgeVertices),
      idToIndex,
    };
  }

  class SnapshotInterpolator {
    constructor(neuronCount) {
      this.neuronCount = neuronCount;
      this.previous = null;
      this.next = null;
    }

    ingest(snapshot) {
      this.previous = this.next;
      this.next = snapshot;
    }

    sample(now) {
      if (!this.next) return null;
      if (!this.previous) return this.next.spikes;
      const dt = Math.max(1, this.next.time - this.previous.time);
      const alpha = clamp((now - this.previous.time) / dt, 0, 1);
      const blended = new Float32Array(this.neuronCount);
      for (let i = 0; i < this.neuronCount; i++) {
        const a = this.previous.spikes[i] || 0;
        const b = this.next.spikes[i] || 0;
        blended[i] = a * (1 - alpha) + b * alpha;
      }
      return blended;
    }
  }

  class SpikeChannel {
    constructor(doc, awareness, sessionId, role) {
      this.doc = doc;
      this.awareness = awareness;
      this.sessionId = sessionId;
      this.role = role;
      this.localId = String(awareness.clientID);
      this.peers = new Map();
      this.onSnapshot = null;

      this.inbox = doc.getMap("spikeInbox");
      this.inbox.observeDeep(() => this._consumeInbox());
      this._ensureInbox();
      awareness.on("change", () => this._syncPeers());
      this._syncPeers();
    }

    _ensureInbox() {
      if (!this.inbox.get(this.localId)) {
        this.inbox.set(this.localId, []);
      }
    }

    _appendInbox(target, message) {
      const key = String(target);
      const queue = this.inbox.get(key) || [];
      queue.push(message);
      this.inbox.set(key, queue.slice(-200));
    }

    _sendSignal(target, payloadMessage) {
      this._appendInbox(target, {
        kind: "signal",
        from: this.localId,
        payload: payloadMessage,
        time: Date.now(),
      });
    }

    _consumeInbox() {
      const queue = this.inbox.get(this.localId) || [];
      const unseen = queue.filter((item) => !item._seenBy || !item._seenBy.includes(this.localId));
      if (!unseen.length) return;

      const marked = queue.map((item) => {
        const seenBy = new Set(item._seenBy || []);
        seenBy.add(this.localId);
        return { ...item, _seenBy: [...seenBy] };
      });
      this.inbox.set(this.localId, marked.slice(-200));

      for (const message of unseen) {
        if (message.kind !== "signal" || !message.from) continue;
        this._onSignal(message.from, message.payload);
      }
    }

    _peerIds() {
      return [...this.awareness.getStates().keys()].map((id) => String(id)).filter((id) => id !== this.localId);
    }

    _syncPeers() {
      const peerIds = this._peerIds();
      for (const peerId of peerIds) {
        if (!this.peers.has(peerId) && this.localId < peerId) {
          this._createPeer(peerId, true);
        }
      }
      for (const knownId of [...this.peers.keys()]) {
        if (!peerIds.includes(knownId)) {
          const peer = this.peers.get(knownId);
          peer?.pc?.close();
          this.peers.delete(knownId);
        }
      }
    }

    _createPeer(remoteId, initiator) {
      const pc = new RTCPeerConnection({ iceServers: [{ urls: ["stun:stun.l.google.com:19302"] }] });
      const peer = { pc, channel: null, ready: false, initiator };
      this.peers.set(remoteId, peer);

      pc.onicecandidate = (event) => {
        if (event.candidate) {
          this._sendSignal(remoteId, { type: "candidate", candidate: event.candidate });
        }
      };

      if (initiator) {
        const channel = pc.createDataChannel("spike-snapshots", { ordered: false, maxRetransmits: 0 });
        this._bindChannel(remoteId, channel);
        pc.createOffer()
          .then((offer) => pc.setLocalDescription(offer))
          .then(() => this._sendSignal(remoteId, { type: "offer", sdp: pc.localDescription }))
          .catch(console.error);
      } else {
        pc.ondatachannel = (event) => this._bindChannel(remoteId, event.channel);
      }
    }

    _bindChannel(remoteId, channel) {
      const peer = this.peers.get(remoteId);
      if (!peer) return;
      peer.channel = channel;
      channel.onopen = () => {
        peer.ready = true;
      };
      channel.onclose = () => {
        peer.ready = false;
      };
      channel.onmessage = (event) => {
        if (!this.onSnapshot) return;
        try {
          const message = JSON.parse(event.data);
          if (message.kind === "snapshot") {
            this.onSnapshot(message.payload);
          }
        } catch (_error) {
          // Ignore malformed packet.
        }
      };
    }

    async _onSignal(remoteId, signal) {
      let peer = this.peers.get(remoteId);
      if (!peer) {
        this._createPeer(remoteId, false);
        peer = this.peers.get(remoteId);
      }
      const pc = peer.pc;

      if (signal.type === "offer") {
        await pc.setRemoteDescription(new RTCSessionDescription(signal.sdp));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        this._sendSignal(remoteId, { type: "answer", sdp: pc.localDescription });
      } else if (signal.type === "answer") {
        await pc.setRemoteDescription(new RTCSessionDescription(signal.sdp));
      } else if (signal.type === "candidate") {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(signal.candidate));
        } catch (_error) {
          // Ignore late candidates.
        }
      }
    }

    broadcast(snapshot) {
      const payloadMessage = JSON.stringify({ kind: "snapshot", payload: snapshot });
      for (const peer of this.peers.values()) {
        if (peer.ready && peer.channel) {
          peer.channel.send(payloadMessage);
        }
      }
    }
  }

  async function initCollaboration(sceneGraph, onSceneGraph) {
    const params = new URLSearchParams(window.location.search);
    const session = params.get("session");
    if (!session) {
      return null;
    }

    const role = params.get("role") || "host";
    const collabLabel = document.getElementById("collab");
    const collabError = document.getElementById("collab-error");

    try {
      const Y = await import("https://cdn.jsdelivr.net/npm/yjs@13.6.18/+esm");
      const { WebrtcProvider } = await import("https://cdn.jsdelivr.net/npm/y-webrtc@10.3.0/+esm");

      const doc = new Y.Doc();
      const provider = new WebrtcProvider(`unicorn-structure-${session}`, doc);
      const sharedScene = doc.getMap("sceneGraph");

      if (role === "host") {
        sharedScene.set("graph", sceneGraph);
      }

      sharedScene.observe(() => {
        const remoteGraph = sharedScene.get("graph");
        if (remoteGraph) {
          onSceneGraph(remoteGraph);
        }
      });

      const spikeChannel = new SpikeChannel(doc, provider.awareness, session, role);
      collabLabel.textContent = `collab session=${session} · role=${role}`;
      return { role, sharedScene, spikeChannel };
    } catch (error) {
      collabError.style.display = "block";
      collabError.textContent = `Collaboration disabled: ${error.message}`;
      return null;
    }
  }

  async function main() {
    const stats = document.getElementById("stats");
    const timeLabel = document.getElementById("time");
    const gpuError = document.getElementById("gpu-error");
    const playToggle = document.getElementById("play-toggle");
    const speedSelect = document.getElementById("speed");
    stats.textContent = `${payload.neuronCount} neurons · ${payload.synapseCount} synapses · ${payload.stepCount} steps`;

    if (!navigator.gpu) {
      gpuError.style.display = "block";
      gpuError.textContent = "WebGPU unavailable in this browser.";
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      gpuError.style.display = "block";
      gpuError.textContent = "No compatible GPU adapter found.";
      return;
    }
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("gpu-canvas");
    const ctx = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();

    function resize() {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      ctx.configure({ device, format, alphaMode: "opaque" });
    }
    window.addEventListener("resize", resize);
    resize();

    let sceneGraph = payload.sceneGraph;
    let geometry = buildGeometryFromScene(sceneGraph);
    let neuronVerts = geometry.neuronVertices;
    let edgeVerts = geometry.edgeVertices;
    let neuronCount = geometry.neuronCount;
    const stepCount = payload.stepCount;
    const localSpikes = new Uint32Array(payload.spikes);
    let liveInterpolated = new Float32Array(neuronCount);
    let usingLiveSnapshots = false;

    function createVertexBuffer(data) {
      const buffer = device.createBuffer({ size: data.byteLength || 4, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
      if (data.byteLength > 0) {
        new Float32Array(buffer.getMappedRange()).set(data);
      }
      buffer.unmap();
      return buffer;
    }

    let neuronBuffer = createVertexBuffer(neuronVerts);
    let edgeBuffer = createVertexBuffer(edgeVerts);
    let spikeBuffer = device.createBuffer({ size: localSpikes.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Uint32Array(spikeBuffer.getMappedRange()).set(localSpikes); spikeBuffer.unmap();
    let intensityBuffer = device.createBuffer({ size: Math.max(neuronCount, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(intensityBuffer, 0, new Float32Array(Math.max(neuronCount, 1)));

    const simUniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const edgeUniformBuffer = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const neuronUniformBuffer = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    function writeSimUniform(stepValue, neuronCountValue, decayValue, boostValue) {
      const simBytes = new ArrayBuffer(16);
      const simView = new DataView(simBytes);
      simView.setUint32(0, stepValue, true);
      simView.setUint32(4, neuronCountValue, true);
      simView.setFloat32(8, decayValue, true);
      simView.setFloat32(12, boostValue, true);
      device.queue.writeBuffer(simUniformBuffer, 0, simBytes);
    }

    function rebuildSceneBuffers(nextGraph) {
      sceneGraph = nextGraph;
      geometry = buildGeometryFromScene(sceneGraph);
      neuronVerts = geometry.neuronVertices;
      edgeVerts = geometry.edgeVertices;
      neuronCount = geometry.neuronCount;
      liveInterpolated = new Float32Array(neuronCount);

      neuronBuffer.destroy();
      edgeBuffer.destroy();
      intensityBuffer.destroy();

      neuronBuffer = createVertexBuffer(neuronVerts);
      edgeBuffer = createVertexBuffer(edgeVerts);
      intensityBuffer = device.createBuffer({ size: Math.max(neuronCount, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(intensityBuffer, 0, new Float32Array(Math.max(neuronCount, 1)));

      const nextComputeBG = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [{binding:0,resource:{buffer:simUniformBuffer}}, {binding:1,resource:{buffer:spikeBuffer}}, {binding:2,resource:{buffer:intensityBuffer}}],
      });
      const nextEdgeDrawBG = usePulseShader
        ? device.createBindGroup({
            layout: drawBindGroupLayout,
            entries: [{binding:0,resource:{buffer:edgeUniformBuffer}}, {binding:1,resource:{buffer:intensityBuffer}}],
          })
        : device.createBindGroup({
            layout: drawFallbackBindGroupLayout,
            entries: [{binding:0,resource:{buffer:edgeUniformBuffer}}],
          });
      const nextNeuronDrawBG = usePulseShader
        ? device.createBindGroup({
            layout: drawBindGroupLayout,
            entries: [{binding:0,resource:{buffer:neuronUniformBuffer}}, {binding:1,resource:{buffer:intensityBuffer}}],
          })
        : device.createBindGroup({
            layout: drawFallbackBindGroupLayout,
            entries: [{binding:0,resource:{buffer:neuronUniformBuffer}}],
          });
      computeBG = nextComputeBG;
      edgeDrawBG = nextEdgeDrawBG;
      neuronDrawBG = nextNeuronDrawBG;
      stats.textContent = `${neuronCount} neurons · ${geometry.synapseCount} synapses · ${stepCount} steps`;
    }

    const computeModule = device.createShaderModule({code: `
      struct Sim { step:u32, neuronCount:u32, decay:f32, boost:f32 }
      @group(0) @binding(0) var<uniform> sim: Sim;
      @group(0) @binding(1) var<storage, read> spikes: array<u32>;
      @group(0) @binding(2) var<storage, read_write> intensities: array<f32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= sim.neuronCount) { return; }
        let idx = sim.step * sim.neuronCount + i;
        let spike = f32(spikes[idx]);
        let decayed = intensities[i] * sim.decay;
        intensities[i] = max(decayed, spike * sim.boost);
      }`
    });
    const drawModule = device.createShaderModule({code: `
      struct Draw { mvp: mat4x4<f32>, tint: vec4<f32> }
      @group(0) @binding(0) var<uniform> draw: Draw;
      @group(0) @binding(1) var<storage, read> intensities: array<f32>;
      struct In { @location(0) pos: vec3<f32>, @location(1) owner: f32 }
      struct Out { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> }
      @vertex
      fn vs_main(input: In) -> Out {
        var out: Out;
        let idx = u32(input.owner);
        let pulse = clamp(intensities[idx], 0.0, 1.0);
        let glow = pulse * pulse;
        let cool = draw.tint.xyz;
        let warm = vec3<f32>(1.0, 0.76, 0.28);
        let colorBase = mix(cool, warm, pulse);
        let glowColor = warm * glow * 0.9;
        out.pos = draw.mvp * vec4<f32>(input.pos, 1.0);
        out.color = vec4<f32>(colorBase + glowColor, min(1.0, draw.tint.w + pulse * 0.35));
        return out;
      }
      @fragment
      fn fs_main(input: Out) -> @location(0) vec4<f32> { return input.color; }`
    });
    const drawFallbackModule = device.createShaderModule({code: `
      struct Draw { mvp: mat4x4<f32>, tint: vec4<f32> }
      @group(0) @binding(0) var<uniform> draw: Draw;
      struct In { @location(0) pos: vec3<f32>, @location(1) owner: f32 }
      struct Out { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> }
      @vertex
      fn vs_main(input: In) -> Out {
        var out: Out;
        out.pos = draw.mvp * vec4<f32>(input.pos, 1.0);
        out.color = draw.tint;
        return out;
      }
      @fragment
      fn fs_main(input: Out) -> @location(0) vec4<f32> { return input.color; }`
    });

    const computeBindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]});
    const drawBindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ]});
    const drawFallbackBindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
    ]});

    let computeBG = device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [{binding:0,resource:{buffer:simUniformBuffer}}, {binding:1,resource:{buffer:spikeBuffer}}, {binding:2,resource:{buffer:intensityBuffer}}],
    });
    let edgeDrawBG = null;
    let neuronDrawBG = null;
    let activeDrawBindGroupLayout = drawBindGroupLayout;
    let usePulseShader = true;
    try {
      edgeDrawBG = device.createBindGroup({
        layout: drawBindGroupLayout,
        entries: [{binding:0,resource:{buffer:edgeUniformBuffer}}, {binding:1,resource:{buffer:intensityBuffer}}],
      });
      neuronDrawBG = device.createBindGroup({
        layout: drawBindGroupLayout,
        entries: [{binding:0,resource:{buffer:neuronUniformBuffer}}, {binding:1,resource:{buffer:intensityBuffer}}],
      });
    } catch (_error) {
      usePulseShader = false;
      activeDrawBindGroupLayout = drawFallbackBindGroupLayout;
      edgeDrawBG = device.createBindGroup({
        layout: drawFallbackBindGroupLayout,
        entries: [{binding:0,resource:{buffer:edgeUniformBuffer}}],
      });
      neuronDrawBG = device.createBindGroup({
        layout: drawFallbackBindGroupLayout,
        entries: [{binding:0,resource:{buffer:neuronUniformBuffer}}],
      });
      gpuError.style.display = "block";
      gpuError.textContent = "Limited WebGPU mode: vertex storage buffers unsupported; spike pulse glow disabled.";
    }

    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
      compute: { module: computeModule, entryPoint: "main" },
    });
    const renderPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [activeDrawBindGroupLayout] }),
      vertex: {
        module: usePulseShader ? drawModule : drawFallbackModule, entryPoint: "vs_main",
        buffers: [{ arrayStride: 16, attributes: [{shaderLocation:0, offset:0, format:"float32x3"}, {shaderLocation:1, offset:12, format:"float32"}] }]
      },
      fragment: { module: usePulseShader ? drawModule : drawFallbackModule, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology: "line-list" },
      depthStencil: undefined,
    });

    const collab = await initCollaboration(sceneGraph, rebuildSceneBuffers);
    const interpolator = new SnapshotInterpolator(neuronCount);

    if (collab) {
      collab.spikeChannel.onSnapshot = (packet) => {
        usingLiveSnapshots = true;
        if (!packet || !Array.isArray(packet.spikes)) return;
        const spikes = new Float32Array(packet.spikes.slice(0, neuronCount));
        interpolator.ingest({ time: packet.time || performance.now(), spikes });
      };
    }

    let step = 0;
    let paused = false;
    let speed = 1;
    let yaw = 0.3;
    let accumulatorMs = 0;
    let lastTs = 0;
    const stepDurationMs = Math.max(16, payload.dt * 1000);

    playToggle.addEventListener("click", () => {
      paused = !paused;
      playToggle.textContent = paused ? "Play" : "Pause";
    });
    speedSelect.addEventListener("change", () => {
      speed = Number(speedSelect.value || 1);
    });

    function frame(ts) {
      if (!lastTs) {
        lastTs = ts;
      }
      const now = ts;
      const deltaMs = ts - lastTs;
      lastTs = ts;
      yaw += 0.003;
      const radius = payload.cameraRadius;
      const eye = [Math.cos(yaw) * radius, radius * 0.6, Math.sin(yaw) * radius];
      const proj = perspective(Math.PI / 3, canvas.width / Math.max(1, canvas.height), 0.1, 1000);
      const view = lookAt(eye, [0,0,0], [0,1,0]);
      const mvp = matMul(proj, view);
      const drawData = new Float32Array(20);
      drawData.set(mvp, 0);

      if (!paused) {
        accumulatorMs += deltaMs * speed;
        const frameAdvance = Math.floor(accumulatorMs / stepDurationMs);
        if (frameAdvance > 0) {
          step = (step + frameAdvance) % stepCount;
          accumulatorMs -= frameAdvance * stepDurationMs;
        }
      }

      writeSimUniform(step, neuronCount, payload.decay, payload.boost);

      const encoder = device.createCommandEncoder();

      if (usingLiveSnapshots) {
        const interpolated = interpolator.sample(now);
        if (interpolated) {
          for (let i = 0; i < liveInterpolated.length; i++) {
            liveInterpolated[i] = Math.max((liveInterpolated[i] || 0) * payload.decay, interpolated[i] || 0);
          }
          device.queue.writeBuffer(intensityBuffer, 0, liveInterpolated);
        }
      } else {
        const cpass = encoder.beginComputePass();
        cpass.setPipeline(computePipeline);
        cpass.setBindGroup(0, computeBG);
        cpass.dispatchWorkgroups(Math.ceil(Math.max(neuronCount, 1) / 64));
        cpass.end();
      }

      const textureView = ctx.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: textureView, clearValue: {r:0.02,g:0.03,b:0.07,a:1}, loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(renderPipeline);
      drawData[16] = 0.1; drawData[17] = 0.45; drawData[18] = 0.9; drawData[19] = 0.8;
      device.queue.writeBuffer(edgeUniformBuffer, 0, drawData);
      pass.setBindGroup(0, edgeDrawBG);
      pass.setVertexBuffer(0, edgeBuffer);
      pass.draw(edgeVerts.length / 4);

      drawData[16] = 0.75; drawData[17] = 0.95; drawData[18] = 1.0; drawData[19] = 1.0;
      device.queue.writeBuffer(neuronUniformBuffer, 0, drawData);
      pass.setBindGroup(0, neuronDrawBG);
      pass.setVertexBuffer(0, neuronBuffer);
      pass.draw(neuronVerts.length / 4);
      pass.end();

      device.queue.submit([encoder.finish()]);

      const spikesForBroadcast = [];
      for (let i = 0; i < neuronCount; i++) {
        spikesForBroadcast.push(localSpikes[step * neuronCount + i] || 0);
      }
      if (collab?.role === "host") {
        collab.spikeChannel.broadcast({ time: now, step, spikes: spikesForBroadcast });
      }

      timeLabel.textContent = `step ${step + 1}/${stepCount} (dt=${payload.dt.toFixed(3)}s)`;
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  main();
  </script>
</body>
</html>
"""


def _build_vertex_buffers(network, layout):
    by_id = {item["id"]: item["position"] for item in layout}
    neuron_ids = [n["id"] for n in network["neurons"]]
    id_to_index = {nid: idx for idx, nid in enumerate(neuron_ids)}

    xs = [by_id[nid][0] for nid in neuron_ids]
    ys = [by_id[nid][1] for nid in neuron_ids]
    zs = [by_id[nid][2] for nid in neuron_ids]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    arm = span * 0.015
    camera_radius = span * 2.5

    neuron_vertices = []
    for nid in neuron_ids:
        x, y, z = by_id[nid]
        owner = float(id_to_index[nid])
        neuron_vertices.extend([x - arm, y, z, owner, x + arm, y, z, owner])
        neuron_vertices.extend([x, y - arm, z, owner, x, y + arm, z, owner])
        neuron_vertices.extend([x, y, z - arm, owner, x, y, z + arm, owner])

    edge_vertices = []
    for syn in network.get("synapses", []):
        src = syn["from"]
        src_index = float(id_to_index[src])
        x0, y0, z0 = by_id[syn["from"]]
        x1, y1, z1 = by_id[syn["to"]]
        edge_vertices.extend([x0, y0, z0, src_index, x1, y1, z1, src_index])

    return neuron_ids, id_to_index, neuron_vertices, edge_vertices, camera_radius


def build_webgpu_payload(network, layout, history):
    neuron_ids, id_to_index, neuron_vertices, edge_vertices, camera_radius = (
        _build_vertex_buffers(network, layout)
    )
    spikes = []
    for step in history:
        frame = [0] * len(neuron_ids)
        spike_values = step.get("spikes", [])
        if isinstance(spike_values, dict):
            for neuron_id, fired in spike_values.items():
                normalized_id = int(neuron_id)
                if fired and normalized_id in id_to_index:
                    frame[id_to_index[normalized_id]] = 1
        else:
            spike_list = list(spike_values)
            is_dense_binary = len(spike_list) == len(neuron_ids) and all(
                value in (0, 1, 0.0, 1.0, False, True) for value in spike_list
            )
            if is_dense_binary:
                for idx, fired in enumerate(spike_list):
                    if fired:
                        neuron_id = neuron_ids[idx]
                        frame[id_to_index[neuron_id]] = 1
            else:
                for neuron_id in spike_list:
                    normalized_id = int(neuron_id)
                    if normalized_id in id_to_index:
                        frame[id_to_index[normalized_id]] = 1
        spikes.extend(frame)
    if len(history) > 1 and "time" in history[0] and "time" in history[1]:
        dt = float(history[1]["time"] - history[0]["time"])
    else:
        dt = float(network.get("dt", 1.0))

    layout_by_id = {item["id"]: item["position"] for item in layout}
    scene_graph = {
        "neurons": [
            {"id": neuron_id, "position": layout_by_id[neuron_id]}
            for neuron_id in neuron_ids
            if neuron_id in layout_by_id
        ],
        "synapses": [
            {
                "from": synapse["from"],
                "to": synapse["to"],
                "weight": synapse.get("weight", 0.0),
            }
            for synapse in network.get("synapses", [])
        ],
    }

    return {
        "neuronCount": len(neuron_ids),
        "synapseCount": len(network.get("synapses", [])),
        "stepCount": len(history),
        "dt": dt,
        "decay": 0.92,
        "boost": 1.0,
        "cameraRadius": camera_radius,
        "neuronVertices": neuron_vertices,
        "edgeVertices": edge_vertices,
        "spikes": spikes,
        "sceneGraph": scene_graph,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a WebGPU neural spike preview HTML."
    )
    parser.add_argument(
        "network",
        nargs="?",
        default="samples/network.json",
        help="Path to Unicorn JSON, SONATA-style JSON, or NeuroML file",
    )
    parser.add_argument(
        "--layout", default="samples/layout_output.json", help="Layout JSON path"
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Optional spike history JSON path. If omitted, simulation is run automatically.",
    )
    parser.add_argument(
        "--output",
        default="output/webgpu_preview.html",
        help="Output HTML path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    network = load_network(args.network)
    with open(args.layout, "r") as f:
        layout = json.load(f)

    if args.history:
        with open(args.history, "r") as f:
            history = json.load(f)
    else:
        history = run_simulation(network)

    payload = build_webgpu_payload(network, layout, history)
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved WebGPU preview to {output_path}")


if __name__ == "__main__":
    main()
