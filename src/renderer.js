// renderer.js — PIXEL-ART WebGPU renderer for the pelican diorama.
//
// Three passes, one command encoder:
//
//   1. shadow  — 2048x2048 depth-only directional shadow map (unchanged).
//   2. art     — the whole scene at a low resolution with the canvas's aspect
//                (about 384x216 worth of pixels), rgba8unorm colour +
//                depth32float depth, sampleCount 1. NO MSAA: pixel art wants
//                hard edges. Sky first (no depth write), then lit geometry.
//   3. blit    — fullscreen triangle, nearest sampler, integer pixel size,
//                covers the whole canvas (no letterbox, no stretch). Adds the
//                1px dark outline from the art depth.
//
// Per-object uniforms stay in 256-byte dynamic-offset slots; meshes are still
// deduplicated by object identity so 20 spokes share one vertex/index range.
//
// Imports only from './math.js' and './shaders.js'.

import { mat4, vec3 } from './math.js';
import { skyWGSL, shadowWGSL, litWGSL, blitWGSL } from './shaders.js';

// The art target is no longer a fixed 16:9 texture: it is sized to the canvas
// aspect so nothing is ever stretched. ART_PIXELS is the pixel budget (384x216
// worth); the integer pixel size is picked so the art lands near that budget,
// and the art covers the whole canvas (the last partial pixel is cropped).
const ART_PIXELS = 384 * 216;
const SHADOW_SIZE = 2048;
const SLOT = 256; // per-object uniform slot stride (minUniformBufferOffsetAlignment)
const DEPTH_FORMAT = 'depth32float';
const ART_COLOR_FORMAT = 'rgba8unorm';
const VERTEX_STRIDE = 24; // position float32x3 @0, normal float32x3 @12

const VERTEX_LAYOUT = {
  arrayStride: VERTEX_STRIDE,
  stepMode: 'vertex',
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'float32x3' },
  ],
};

// Orthographic projection with WebGPU's 0..1 clip depth (right-handed view space,
// camera looking down -Z, so view z runs -near .. -far).
function ortho(l, r, b, t, n, f) {
  const m = new Float32Array(16);
  m[0] = 2 / (r - l);
  m[5] = 2 / (t - b);
  m[10] = 1 / (n - f);
  m[12] = -(r + l) / (r - l);
  m[13] = -(t + b) / (t - b);
  m[14] = n / (n - f);
  m[15] = 1;
  return m;
}

export class Renderer {
  constructor(opts = {}) {
    this.lightDir = vec3.normalize(opts.lightDir || [-0.55, 0.82, 0.42]);
    // Elevation (rad) of the visible sun disc, for the sky halo; 0 = the light's own.
    this.sunDiscElev = opts.sunDiscElev || 0;
    this.ambient = opts.ambient || [0.34, 0.38, 0.44];
    // Small biases: the shadow pass culls front faces, so the stored depth is
    // the far side of each solid and no acne correction is needed.
    this.shadowBias = opts.shadowBias !== undefined ? opts.shadowBias : 0.0004;
    this.shadowSlopeBias = opts.shadowSlopeBias !== undefined ? opts.shadowSlopeBias : 0.0016;
    this.shadowExtent = opts.shadowExtent !== undefined ? opts.shadowExtent : 3.2;
    this.shadowDistance = opts.shadowDistance !== undefined ? opts.shadowDistance : 9.0;
    this.shadowCenter = opts.shadowCenter || [0, 0.55, 0];

    this.device = null;
    this.context = null;
    this.format = null;
    this.lost = false;
    this.lostReason = null;
    this.onDeviceLost = null;

    this.parts = [];
    this.meshRanges = [];
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.objectBuffer = null;
    this.objectData = null;

    // Canvas size and the integer blit rect derived from it.
    this.width = 0;
    this.height = 0;
    this.artScale = 1;
    this.artOffsetX = 0;
    this.artOffsetY = 0;
    this.artW = 0;
    this.artH = 0;
    this.artAspect = 16 / 9;
    // Optional fixed pixel size (canvas px per art px); 0 = automatic.
    this.pixelSize = opts.pixelSize > 0 ? Math.floor(opts.pixelSize) : 0;
    // Smooth mode (?pixels=false): full resolution, no posterise / dither /
    // stepped shadows / outline, so the real shading is visible.
    this.smooth = !!opts.smooth;
    if (this.smooth) this.pixelSize = 1;

    this.globalData = new Float32Array(64); // 256 bytes
    this.blitData = new Float32Array(8); // 32 bytes
    this.lightViewProj = mat4.identity();
  }

  /* ------------------------------------------------------------- lifecycle */

  async init(canvas) {
    if (!navigator.gpu) throw new Error('WebGPU is not available (navigator.gpu missing)');
    this.canvas = canvas;

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter available');
    this.adapter = adapter;

    const device = await adapter.requestDevice();
    this.device = device;

    device.lost.then((info) => {
      this.lost = true;
      this.lostReason = info.message || String(info.reason);
      console.error('WebGPU device lost (' + info.reason + '):', info.message);
      if (typeof this.onDeviceLost === 'function') this.onDeviceLost(info);
    });

    device.addEventListener('uncapturederror', (ev) => {
      console.error('WebGPU uncaptured error:', ev.error.message);
    });

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('canvas.getContext("webgpu") returned null');
    this.context = context;

    this.format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: this.format, alphaMode: 'opaque' });

    // Global uniform: 256 bytes, written every frame.
    this.globalBuffer = device.createBuffer({
      label: 'globals',
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Blit uniform: integer scale + art rect offset, written on resize.
    this.blitBuffer = device.createBuffer({
      label: 'blit-params',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Shadow map.
    this.shadowTexture = device.createTexture({
      label: 'shadow-map',
      size: { width: SHADOW_SIZE, height: SHADOW_SIZE, depthOrArrayLayers: 1 },
      format: DEPTH_FORMAT,
      sampleCount: 1,
      mipLevelCount: 1,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowView = this.shadowTexture.createView({ label: 'shadow-map-view' });

    this.shadowSampler = device.createSampler({
      label: 'shadow-sampler',
      compare: 'less-equal',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // Nearest, clamp: the blit must never interpolate.
    this.nearestSampler = device.createSampler({
      label: 'art-nearest',
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.globalBindGroupLayout = device.createBindGroupLayout({
      label: 'globals-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', minBindingSize: 256 },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'depth', viewDimension: '2d', multisampled: false },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'comparison' },
        },
      ],
    });

    this.objectBindGroupLayout = device.createBindGroupLayout({
      label: 'object-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: SLOT },
        },
      ],
    });

    this.pipelineLayout = device.createPipelineLayout({
      label: 'scene-layout',
      bindGroupLayouts: [this.globalBindGroupLayout, this.objectBindGroupLayout],
    });

    // The sky needs no per-object data, so it gets a layout with group 0 only.
    this.skyPipelineLayout = device.createPipelineLayout({
      label: 'sky-layout',
      bindGroupLayouts: [this.globalBindGroupLayout],
    });

    // The shadow pass writes the shadow map while it is bound, so it must not
    // bind the shadow map as a texture: group 0 there is the uniform alone.
    this.shadowBindGroupLayout = device.createBindGroupLayout({
      label: 'shadow-globals-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform', minBindingSize: 256 },
        },
      ],
    });
    this.shadowPipelineLayout = device.createPipelineLayout({
      label: 'shadow-layout',
      bindGroupLayouts: [this.shadowBindGroupLayout, this.objectBindGroupLayout],
    });

    // Blit: globals (for the depth unprojection), blit params, art colour,
    // nearest sampler, art depth.
    this.blitBindGroupLayout = device.createBindGroupLayout({
      label: 'blit-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', minBindingSize: 256 },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', minBindingSize: 32 },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'depth', viewDimension: '2d', multisampled: false },
        },
      ],
    });
    this.blitPipelineLayout = device.createPipelineLayout({
      label: 'blit-pass-layout',
      bindGroupLayouts: [this.blitBindGroupLayout],
    });

    this.globalBindGroup = device.createBindGroup({
      label: 'globals',
      layout: this.globalBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.globalBuffer } },
        { binding: 1, resource: this.shadowView },
        { binding: 2, resource: this.shadowSampler },
      ],
    });

    this.shadowBindGroup = device.createBindGroup({
      label: 'shadow-globals',
      layout: this.shadowBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.globalBuffer } }],
    });

    const skyModule = device.createShaderModule({ label: 'sky', code: skyWGSL });
    const shadowModule = device.createShaderModule({ label: 'shadow', code: shadowWGSL });
    const litModule = device.createShaderModule({ label: 'lit', code: litWGSL });
    const blitModule = device.createShaderModule({ label: 'blit', code: blitWGSL });

    for (const [name, mod] of [
      ['sky', skyModule],
      ['shadow', shadowModule],
      ['lit', litModule],
      ['blit', blitModule],
    ]) {
      const info = await mod.getCompilationInfo();
      for (const msg of info.messages) {
        if (msg.type === 'error') console.error('WGSL ' + name + ' ' + msg.lineNum + ':' + msg.linePos + ' ' + msg.message);
        else if (msg.type === 'warning') console.warn('WGSL ' + name + ' ' + msg.lineNum + ':' + msg.linePos + ' ' + msg.message);
      }
    }

    this.skyPipeline = await device.createRenderPipelineAsync({
      label: 'sky',
      layout: this.skyPipelineLayout,
      vertex: { module: skyModule, entryPoint: 'vs', buffers: [] },
      fragment: { module: skyModule, entryPoint: 'fs', targets: [{ format: ART_COLOR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: false,
        depthCompare: 'always',
      },
      multisample: { count: 1 },
    });

    this.shadowPipeline = await device.createRenderPipelineAsync({
      label: 'shadow',
      layout: this.shadowPipelineLayout,
      vertex: { module: shadowModule, entryPoint: 'vs', buffers: [VERTEX_LAYOUT] },
      fragment: { module: shadowModule, entryPoint: 'fs', targets: [] },
      primitive: { topology: 'triangle-list', cullMode: 'front', frontFace: 'ccw' },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: 'less',
        depthBias: 0,
        depthBiasSlopeScale: 0,
        depthBiasClamp: 0,
      },
    });

    this.litPipeline = await device.createRenderPipelineAsync({
      label: 'lit',
      layout: this.pipelineLayout,
      vertex: { module: litModule, entryPoint: 'vs', buffers: [VERTEX_LAYOUT] },
      fragment: { module: litModule, entryPoint: 'fs', targets: [{ format: ART_COLOR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: { count: 1 },
    });

    this.blitPipeline = await device.createRenderPipelineAsync({
      label: 'blit',
      layout: this.blitPipelineLayout,
      vertex: { module: blitModule, entryPoint: 'vs', buffers: [] },
      fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
    });

    this._updateLightMatrix();
    this.resize();
    return this;
  }

  /* ---------------------------------------------------------------- scene */

  setScene(parts) {
    const device = this.device;
    if (!device) throw new Error('Renderer.init() must be awaited before setScene()');
    this.parts = Array.isArray(parts) ? parts : [];

    // Dedupe by mesh object identity: 24 spokes built once share one range.
    const meshMap = new Map();
    for (const part of this.parts) {
      if (!part || !part.mesh) continue;
      if (!meshMap.has(part.mesh)) meshMap.set(part.mesh, null);
    }
    const meshes = Array.from(meshMap.keys());

    let vertexTotal = 0;
    let indexTotal = 0;
    for (const mesh of meshes) {
      vertexTotal += mesh.positions.length / 3;
      indexTotal += mesh.indices.length;
    }

    // Release the previous scene's GPU buffers.
    if (this.vertexBuffer) this.vertexBuffer.destroy();
    if (this.indexBuffer) this.indexBuffer.destroy();
    if (this.objectBuffer) this.objectBuffer.destroy();
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.objectBuffer = null;
    this.meshRanges = new Array(this.parts.length).fill(null);

    if (vertexTotal > 0 && indexTotal > 0) {
      const vertices = new Float32Array(vertexTotal * 6);
      const indices = new Uint32Array(indexTotal);
      let vBase = 0;
      let iBase = 0;

      for (const mesh of meshes) {
        const count = mesh.positions.length / 3;
        const pos = mesh.positions;
        const nor = mesh.normals;
        for (let i = 0; i < count; i++) {
          const d = (vBase + i) * 6;
          const s = i * 3;
          vertices[d + 0] = pos[s + 0];
          vertices[d + 1] = pos[s + 1];
          vertices[d + 2] = pos[s + 2];
          vertices[d + 3] = nor[s + 0];
          vertices[d + 4] = nor[s + 1];
          vertices[d + 5] = nor[s + 2];
        }
        const idx = mesh.indices;
        for (let i = 0; i < idx.length; i++) indices[iBase + i] = idx[i] + vBase;
        meshMap.set(mesh, { firstIndex: iBase, indexCount: idx.length, baseVertex: vBase });
        vBase += count;
        iBase += idx.length;
      }

      this.vertexBuffer = device.createBuffer({
        label: 'vertices',
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(this.vertexBuffer, 0, vertices);

      this.indexBuffer = device.createBuffer({
        label: 'indices',
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(this.indexBuffer, 0, indices);
    }

    for (let i = 0; i < this.parts.length; i++) {
      const part = this.parts[i];
      if (part && part.mesh && meshMap.has(part.mesh)) this.meshRanges[i] = meshMap.get(part.mesh);
    }

    // One uniform buffer for every object, 256 bytes apart.
    const slots = Math.max(this.parts.length, 1);
    this.objectData = new Float32Array(slots * (SLOT / 4));
    this.objectBuffer = device.createBuffer({
      label: 'objects',
      size: slots * SLOT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.objectBindGroup = device.createBindGroup({
      label: 'objects',
      layout: this.objectBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.objectBuffer, offset: 0, size: SLOT } }],
    });

    this._uploadObjects();
  }

  _uploadObjects() {
    const parts = this.parts;
    const data = this.objectData;
    if (!data) return;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const base = i * (SLOT / 4);
      const model = part.model;
      if (model) {
        for (let k = 0; k < 16; k++) data[base + k] = model[k];
      } else {
        for (let k = 0; k < 16; k++) data[base + k] = k % 5 === 0 ? 1 : 0;
      }
      const color = part.color || [1, 1, 1];
      data[base + 16] = color[0];
      data[base + 17] = color[1];
      data[base + 18] = color[2];
      data[base + 19] = 1;
      const params = part.params || [0, 1, 0];
      data[base + 20] = params[0];
      data[base + 21] = params[1];
      data[base + 22] = params[2];
      data[base + 23] = 0;
    }
    if (parts.length > 0) this.device.queue.writeBuffer(this.objectBuffer, 0, data, 0, parts.length * (SLOT / 4));
  }

  /* ---------------------------------------------------------------- light */

  _updateLightMatrix() {
    const dir = this.lightDir;
    const c = this.shadowCenter;
    const d = this.shadowDistance;
    const eye = [c[0] + dir[0] * d, c[1] + dir[1] * d, c[2] + dir[2] * d];
    let up = [0, 1, 0];
    if (Math.abs(dir[1]) > 0.98) up = [0, 0, 1];
    const view = mat4.lookAt(eye, c, up);
    const e = this.shadowExtent;
    const proj = ortho(-e, e, -e, e, 0.5, d * 2.2);
    this.lightViewProj = mat4.multiply(proj, view);
  }

  setLight(dir, ambient) {
    if (dir) this.lightDir = vec3.normalize(dir);
    if (ambient) this.ambient = ambient;
    this._updateLightMatrix();
  }

  /* ----------------------------------------------------------- art target */

  // (Re)creates the art colour/depth targets at artW x artH and the blit bind
  // group that samples them.
  _createArtTargets(artW, artH) {
    const device = this.device;
    if (this.artColorTexture) this.artColorTexture.destroy();
    if (this.artDepthTexture) this.artDepthTexture.destroy();

    this.artColorTexture = device.createTexture({
      label: 'art-color',
      size: { width: artW, height: artH, depthOrArrayLayers: 1 },
      format: ART_COLOR_FORMAT,
      sampleCount: 1,
      mipLevelCount: 1,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.artColorView = this.artColorTexture.createView({ label: 'art-color-view' });

    this.artDepthTexture = device.createTexture({
      label: 'art-depth',
      size: { width: artW, height: artH, depthOrArrayLayers: 1 },
      format: DEPTH_FORMAT,
      sampleCount: 1,
      mipLevelCount: 1,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.artDepthView = this.artDepthTexture.createView({ label: 'art-depth-view' });

    this.blitBindGroup = device.createBindGroup({
      label: 'blit',
      layout: this.blitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.globalBuffer } },
        { binding: 1, resource: { buffer: this.blitBuffer } },
        { binding: 2, resource: this.artColorView },
        { binding: 3, resource: this.nearestSampler },
        { binding: 4, resource: this.artDepthView },
      ],
    });

    this.artW = artW;
    this.artH = artH;
    this.artAspect = artW / artH;
  }

  /* --------------------------------------------------------------- resize */

  // Fills the canvas without stretching: one art pixel is `scale` x `scale`
  // canvas pixels, the art target matches the canvas aspect, and it is centred
  // with any partial edge pixel cropped.
  resize() {
    const canvas = this.canvas;
    if (!canvas || !this.device) return false;
    const w = canvas.width | 0;
    const h = canvas.height | 0;
    if (w <= 0 || h <= 0) return false;
    if (w === this.width && h === this.height && this.artColorTexture) return true;

    const scale = this.pixelSize || Math.max(1, Math.round(Math.sqrt((w * h) / ART_PIXELS)));
    const artW = Math.max(1, Math.ceil(w / scale));
    const artH = Math.max(1, Math.ceil(h / scale));
    const offX = Math.floor((w - artW * scale) / 2);
    const offY = Math.floor((h - artH * scale) / 2);

    if (artW !== this.artW || artH !== this.artH || !this.artColorTexture) {
      this._createArtTargets(artW, artH);
    }

    this.width = w;
    this.height = h;
    this.artScale = scale;
    this.artOffsetX = offX;
    this.artOffsetY = offY;

    this.blitData[0] = scale;
    this.blitData[1] = offX;
    this.blitData[2] = offY;
    this.blitData[3] = artW;
    this.blitData[4] = artH;
    this.blitData[5] = 0;
    this.blitData[6] = 0;
    this.blitData[7] = 0;
    this.device.queue.writeBuffer(this.blitBuffer, 0, this.blitData);
    return true;
  }

  /* ----------------------------------------------------------------- draw */

  draw(viewProj, cameraPos) {
    const device = this.device;
    if (!device || this.lost) return;
    if (!this.artColorTexture && !this.resize()) return; // zero-sized canvas

    // ---- global uniforms
    const g = this.globalData;
    g.set(viewProj, 0);
    g[16] = cameraPos[0];
    g[17] = cameraPos[1];
    g[18] = cameraPos[2];
    g[19] = 1;
    g[20] = this.lightDir[0];
    g[21] = this.lightDir[1];
    g[22] = this.lightDir[2];
    g[23] = this.sunDiscElev;
    g[24] = this.ambient[0];
    g[25] = this.ambient[1];
    g[26] = this.ambient[2];
    g[27] = 0;
    g.set(this.lightViewProj, 28);
    g[44] = 1 / SHADOW_SIZE;
    g[45] = this.shadowBias;
    g[46] = this.shadowSlopeBias;
    g[47] = this.smooth ? 1 : 0;
    g.set(mat4.invert(viewProj), 48);
    device.queue.writeBuffer(this.globalBuffer, 0, g);

    // ---- per-object uniforms (model matrices change every frame)
    this._uploadObjects();

    const encoder = device.createCommandEncoder({ label: 'frame' });
    const hasGeometry = this.parts.length > 0 && this.vertexBuffer && this.indexBuffer;

    // ---- pass 1: shadow map
    if (hasGeometry) {
      const pass = encoder.beginRenderPass({
        label: 'shadow-pass',
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      pass.setPipeline(this.shadowPipeline);
      pass.setBindGroup(0, this.shadowBindGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      pass.setIndexBuffer(this.indexBuffer, 'uint32');
      for (let i = 0; i < this.parts.length; i++) {
        const range = this.meshRanges[i];
        if (!range || range.indexCount === 0) continue;
        pass.setBindGroup(1, this.objectBindGroup, [i * SLOT]);
        pass.drawIndexed(range.indexCount, 1, range.firstIndex, 0, 0);
      }
      pass.end();
    }

    // ---- pass 2: the art target (canvas aspect), no MSAA, depth stored for the blit
    const artPass = encoder.beginRenderPass({
      label: 'art-pass',
      colorAttachments: [
        {
          view: this.artColorView,
          clearValue: { r: 0.557, g: 0.792, b: 0.902, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.artDepthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    artPass.setPipeline(this.skyPipeline);
    artPass.setBindGroup(0, this.globalBindGroup);
    artPass.draw(3, 1, 0, 0);

    if (hasGeometry) {
      artPass.setPipeline(this.litPipeline);
      artPass.setBindGroup(0, this.globalBindGroup);
      artPass.setVertexBuffer(0, this.vertexBuffer);
      artPass.setIndexBuffer(this.indexBuffer, 'uint32');
      for (let i = 0; i < this.parts.length; i++) {
        const range = this.meshRanges[i];
        if (!range || range.indexCount === 0) continue;
        artPass.setBindGroup(1, this.objectBindGroup, [i * SLOT]);
        artPass.drawIndexed(range.indexCount, 1, range.firstIndex, 0, 0);
      }
    }

    artPass.end();

    // ---- pass 3: nearest-neighbour integer-scale blit + 1px outline
    const blitPass = encoder.beginRenderPass({
      label: 'blit-pass',
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    blitPass.setPipeline(this.blitPipeline);
    blitPass.setBindGroup(0, this.blitBindGroup);
    blitPass.draw(3, 1, 0, 0);
    blitPass.end();

    device.queue.submit([encoder.finish()]);
  }

  destroy() {
    if (this.artColorTexture) this.artColorTexture.destroy();
    if (this.artDepthTexture) this.artDepthTexture.destroy();
    if (this.shadowTexture) this.shadowTexture.destroy();
    if (this.vertexBuffer) this.vertexBuffer.destroy();
    if (this.indexBuffer) this.indexBuffer.destroy();
    if (this.objectBuffer) this.objectBuffer.destroy();
    if (this.globalBuffer) this.globalBuffer.destroy();
    if (this.blitBuffer) this.blitBuffer.destroy();
    this.artColorTexture = this.artDepthTexture = this.shadowTexture = null;
    this.artColorView = this.artDepthView = this.shadowView = null;
    this.vertexBuffer = this.indexBuffer = this.objectBuffer = this.globalBuffer = null;
    this.blitBuffer = null;
    this.parts = [];
    this.width = this.height = 0;
  }
}
