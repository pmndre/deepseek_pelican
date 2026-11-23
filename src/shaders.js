// shaders.js — WGSL sources for the pelican scene, as inline strings.
//
// Pixel-art look: everything is drawn into a low-resolution offscreen target
// (about 384x216 worth of pixels, matching the canvas aspect) and then blitted
// to the canvas with a nearest sampler at an integer scale. Colour is
// quantised to 6 levels per channel with a 4x4 Bayer dither evaluated in ART
// PIXEL coordinates (the art target is 1:1 with the fragments, so
// @builtin(position) IS the art pixel), the sky is a handful of flat bands,
// shadows are a flat 2-level patch, and the blit pass draws a 1px dark outline
// just outside every silhouette.
//
// Smooth mode (?pixels=false, shadowParams.w = 1) renders at full resolution
// and skips the quantising, the stepped sky/shadows and the outline.
//
// Uniform layouts (must match renderer.js byte for byte):
//
//   Globals (bind group 0, binding 0) — 256 bytes
//     @0    mat4x4<f32> viewProj        camera view-projection (clip z in [0,1])
//     @64   vec4<f32>   cameraPos       xyz = eye position, w unused
//     @80   vec4<f32>   lightDir        xyz = direction TO the light (normalized),
//                                         w = elevation (rad) of the visible sun disc, 0 = use xyz
//     @96   vec4<f32>   ambient         rgb = ambient intensity, w unused
//     @112  mat4x4<f32> lightViewProj   light view-projection (orthographic)
//     @176  vec4<f32>   shadowParams    x = 1/shadowMapSize, y = constant depth bias,
//                                       z = slope-scaled bias, w = 1 for smooth mode
//     @192  mat4x4<f32> invViewProj     inverse of viewProj (sky ray reconstruction)
//     total 256 bytes
//
//   Obj (bind group 1, binding 0, dynamic offset, 256-byte stride)
//     @0    mat4x4<f32> model
//     @64   vec4<f32>   color           rgb = albedo (sRGB 0..1), a unused
//     @80   vec4<f32>   params          x = specular, y = shininess, z = emissive, w unused
//     total 96 bytes used, slot padded to 256
//
//   Blit (bind group 0, binding 1) — 32 bytes
//     @0  f32 scale     integer blit scale
//     @4  f32 offX      left edge of the art rect, in canvas pixels
//     @8  f32 offY      top edge of the art rect, in canvas pixels
//     @12 f32 artW      art target width
//     @16 f32 artH      art target height
//     @20..31 padding
//
// Vertex layout everywhere: stride 24 bytes, position float32x3 @0, normal float32x3 @12.

const GLOBALS_WGSL = /* wgsl */ `
struct Globals {
  viewProj      : mat4x4<f32>,
  cameraPos     : vec4<f32>,
  lightDir      : vec4<f32>,
  ambient       : vec4<f32>,
  lightViewProj : mat4x4<f32>,
  shadowParams  : vec4<f32>,
  invViewProj   : mat4x4<f32>,
};
`;

const OBJECT_WGSL = /* wgsl */ `
struct Obj {
  model  : mat4x4<f32>,
  color  : vec4<f32>,
  params : vec4<f32>,
};
`;

/* ------------------------------------------------------- dither / quantise */
// The 4x4 ordered (Bayer) matrix, evaluated on the ART pixel index. Because the
// art pass writes the low-res target 1:1, the pattern is
// locked to the low-res grid: it is chunky and stable, never shimmering.
//
// posterize6 then gives 6 levels per channel (0, .2, .4, .6, .8, 1) with the
// dither breaking the boundaries — the classic pixel-art ramp.

const DITHER_WGSL = /* wgsl */ `
fn bayer4(px : vec2<i32>) -> f32 {
  let x = px.x & 3;
  let y = px.y & 3;
  var m = array<f32, 16>(
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0
  );
  return (m[y * 4 + x] + 0.5) / 16.0;
}

// G.shadowParams.w > 0.5 = smooth mode (?pixels=false): no quantising.
fn smoothMode() -> bool {
  return G.shadowParams.w > 0.5;
}

fn posterize6(c : vec3<f32>, b : f32) -> vec3<f32> {
  let v = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  if (smoothMode()) {
    return v;
  }
  return floor(v * 5.0 + b) / 5.0;
}
`;

/* --------------------------------------------------------------- sky pass */
// Fullscreen triangle, no vertex buffer. Depth test is 'always' and depth write
// is off, so the sky simply fills the art colour target before anything else and
// leaves the cleared depth (1.0) intact for the outline pass.
//
// The sky is 5 flat horizontal bands (horizon cream -> pale blue -> zenith blue,
// all palette stops or blends of them). The band index is dithered, so the band
// boundaries are a one-pixel checker transition instead of a hard staircase, and
// the whole colour is posterised with the same matrix. A dithered sun glow sits
// on the light direction.

export const skyWGSL = /* wgsl */ `
${GLOBALS_WGSL}
${DITHER_WGSL}

@group(0) @binding(0) var<uniform> G : Globals;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) ndc : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var o : VSOut;
  let q = p[vi];
  o.clip = vec4<f32>(q, 1.0, 1.0);
  o.ndc = q;
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let px = vec2<i32>(floor(in.clip.xy));
  let b = bayer4(px);

  // Reconstruct the world-space view ray through this pixel (far-plane point).
  let h = G.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
  let farPoint = h.xyz / h.w;
  let dir = normalize(farPoint - G.cameraPos.xyz);

  // Elevation, remapped so 0 is the horizon and 1 is the top of the frame
  // (the camera's 45 deg vertical fov puts ~+0.42 in y at the top edge).
  let elev = clamp(dir.y / 0.42, 0.0, 1.0);

  // 5 flat bands. The dither amplitude is deliberately tiny (0.05 of a band),
  // so it only bites within a couple of pixels of a boundary: the boundary is a
  // chunky one-pixel checker, the bands themselves stay perfectly flat.
  let band = clamp(floor(elev * 4.0 + 0.5 + (b - 0.5) * 0.05), 0.0, 4.0);

  var bands = array<vec3<f32>, 5>(
    vec3<f32>(1.000, 0.910, 0.714), // #ffe8b6 horizon
    vec3<f32>(0.863, 0.914, 0.961), // #dce9f5 pale sky (no grey blend)
    vec3<f32>(0.741, 0.878, 0.996), // #bde0fe sky mid
    vec3<f32>(0.640, 0.831, 0.944), // mid -> zenith
    vec3<f32>(0.557, 0.792, 0.902)  // #8ecae6 zenith
  );
  var col = bands[i32(band)];
  if (smoothMode()) {
    // continuous gradient through the same stops
    let fb = elev * 4.0;
    let i0 = i32(clamp(floor(fb), 0.0, 3.0));
    col = mix(bands[i0], bands[i0 + 1], clamp(fb - f32(i0), 0.0, 1.0));
  }

  // Below the horizon: the same horizon cream, so the ground dissolves into it.
  if (dir.y < 0.0) {
    col = bands[0];
  }

  // Sun glow toward the key light, in palette colours (#ffe9a8 -> #ffd166),
  // quantised to 4 flat rings with the same dither so it is a stepped halo.
  // The halo follows the VISIBLE sun: same compass heading as the light, but
  // at the disc's (lower) elevation when lightDir.w gives one.
  let L0 = normalize(G.lightDir.xyz);
  var L = L0;
  if (G.lightDir.w != 0.0) {
    let hz = normalize(vec2<f32>(L0.x, L0.z));
    let e = G.lightDir.w;
    L = vec3<f32>(hz.x * cos(e), sin(e), hz.y * cos(e));
  }
  let d = max(dot(dir, L), 0.0);
  let glow = clamp(pow(d, 22.0) * 0.90 + pow(d, 5.0) * 0.16, 0.0, 1.0);
  let glowCol = mix(vec3<f32>(1.000, 0.914, 0.659), vec3<f32>(1.000, 0.820, 0.400), pow(d, 60.0));
  let gq = select(floor(glow * 3.0 + 0.5 + (b - 0.5) * 0.16) / 3.0, glow, smoothMode());
  col = mix(col, glowCol, gq);

  return vec4<f32>(col, 1.0);
}
`;

/* ----------------------------------------------------------- shadow pass */
// Depth only: no colour targets, fragment stage writes nothing.
// Front-face culling is on (pipeline state), so open single-sided geometry such
// as the ground plane does not write depth into the map, and the depth stored is
// the far side of each solid — which is why the depth bias can stay small and
// contact shadows stay attached to the tyres.

export const shadowWGSL = /* wgsl */ `
${GLOBALS_WGSL}
${OBJECT_WGSL}

@group(0) @binding(0) var<uniform> G : Globals;
@group(1) @binding(0) var<uniform> obj : Obj;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
};

@vertex
fn vs(@location(0) position : vec3<f32>) -> VSOut {
  var o : VSOut;
  o.clip = G.lightViewProj * (obj.model * vec4<f32>(position, 1.0));
  return o;
}

@fragment
fn fs() {
}
`;

/* -------------------------------------------------------------- lit pass */
// Same warm-key / cool-fill / hemispheric-ambient rig as before (it is what
// makes a white bird read as volume), but the output is now flat: the shadow
// term is quantised to two levels with the Bayer matrix so a shadow is a hard
// dark patch with a dithered edge, and the final colour is posterised to 6
// levels per channel. No smooth gradients survive.

export const litWGSL = /* wgsl */ `
${GLOBALS_WGSL}
${OBJECT_WGSL}
${DITHER_WGSL}

@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var shadowMap : texture_depth_2d;
@group(0) @binding(2) var shadowSampler : sampler_comparison;
@group(1) @binding(0) var<uniform> obj : Obj;

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
};

struct VSOut {
  @builtin(position) clip      : vec4<f32>,
  @location(0)       worldPos  : vec3<f32>,
  @location(1)       normal    : vec3<f32>,
  @location(2)       shadowPos : vec4<f32>,
};

// Inverse-transpose of the model's upper 3x3, so non-uniform scaling
// (scaled spheres for the body, the pouch, the crest) keeps correct normals.
fn normalMatrix(m : mat4x4<f32>) -> mat3x3<f32> {
  let a = m[0].xyz;
  let b = m[1].xyz;
  let c = m[2].xyz;
  let det = dot(a, cross(b, c));
  let s = select(1.0, 1.0 / det, abs(det) > 1e-8);
  return mat3x3<f32>(cross(b, c) * s, cross(c, a) * s, cross(a, b) * s);
}

@vertex
fn vs(in : VSIn) -> VSOut {
  let world = obj.model * vec4<f32>(in.position, 1.0);
  var o : VSOut;
  o.clip = G.viewProj * world;
  o.worldPos = world.xyz;
  o.normal = normalMatrix(obj.model) * in.normal;
  o.shadowPos = G.lightViewProj * world;
  return o;
}

// 3x3 PCF with a slope-scaled depth bias. The kernel is deliberately tight
// (0.7 texel) so the tyre shadows keep a defined edge instead of smearing into
// a pale halo. Returns 1.0 = fully lit.
fn shadowFactor(sp : vec4<f32>, N : vec3<f32>, L : vec3<f32>) -> f32 {
  let ndc = sp.xyz / sp.w;
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return 1.0;
  }
  if (ndc.z < 0.0 || ndc.z > 1.0) {
    return 1.0;
  }
  let texel = G.shadowParams.x * 0.70;
  let slope = clamp(1.0 - dot(N, L), 0.0, 1.0);
  let bias = G.shadowParams.y + G.shadowParams.z * slope;
  let refDepth = ndc.z - bias;

  var sum = 0.0;
  for (var y : i32 = -1; y <= 1; y = y + 1) {
    for (var x : i32 = -1; x <= 1; x = x + 1) {
      let off = vec2<f32>(f32(x), f32(y)) * texel;
      sum = sum + textureSampleCompareLevel(shadowMap, shadowSampler, uv + off, refDepth);
    }
  }
  return sum / 9.0;
}

// Contact patch under each tyre. The bike is pinned at x = +-0.72, z = 0, so
// this stays valid while the world scrolls past.
fn contactPatch(p : vec3<f32>) -> f32 {
  let a = length(vec2<f32>((p.x + 0.60) / 0.17, p.z / 0.105)); // rear tyre (pelican.js REAR)
  let b = length(vec2<f32>((p.x - 0.78) / 0.17, p.z / 0.105)); // front tyre (pelican.js FRONT)
  return max(1.0 - smoothstep(0.28, 1.0, a), 1.0 - smoothstep(0.28, 1.0, b));
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let px = vec2<i32>(floor(in.clip.xy));
  let b = bayer4(px);

  let N = normalize(in.normal);
  let L = normalize(G.lightDir.xyz);
  let V = normalize(G.cameraPos.xyz - in.worldPos);
  let H = normalize(L + V);

  // Colours arrive authored in sRGB; lighting happens in linear space.
  let albedo = pow(max(obj.color.rgb, vec3<f32>(0.0)), vec3<f32>(2.2));
  let specular = obj.params.x;
  let shininess = max(obj.params.y, 1.0);
  let emissive = obj.params.z;

  /* ---- light rig ------------------------------------------------------ */
  let sunCol  = vec3<f32>(1.000, 0.925, 0.790);
  let fillCol = vec3<f32>(0.190, 0.270, 0.430);
  let rimCol  = vec3<f32>(0.520, 0.660, 0.920);

  // Horizontal axis pointing to camera-right, independent of the camera's pitch.
  let ch = vec3<f32>(V.z, 0.0, -V.x);
  let camRight = select(vec3<f32>(1.0, 0.0, 0.0), normalize(ch), length(ch) > 1e-4);
  let fillDir = normalize(camRight + vec3<f32>(0.0, 0.34, 0.0) - V * 0.45);
  let rimDir = normalize(camRight * 0.80 - V * 0.90 + vec3<f32>(0.0, 0.42, 0.0));

  let ndl = max(dot(N, L), 0.0);

  // 3x3 PCF, then quantised to two levels with the same dither: the penumbra
  // becomes a one-pixel checker edge and the shadow body is a flat dark patch.
  let shRaw = shadowFactor(in.shadowPos, N, L);
  let sh = select(floor(shRaw + b), shRaw, smoothMode());

  let shAmb = mix(0.42, 1.0, sh);

  // Hemispheric ambient: cool sky above, warm bounce from the ground below.
  let skyTint    = vec3<f32>(0.560, 0.700, 0.900);
  let groundTint = vec3<f32>(0.780, 0.680, 0.520);
  let upness = N.y * 0.5 + 0.5;
  let hemi = mix(groundTint, skyTint, upness) * G.ambient.rgb;

  // Cool where shadowed, warm where lit.
  let ambTint = mix(vec3<f32>(0.72, 0.84, 1.06), vec3<f32>(1.07, 1.00, 0.90), sh);

  var col = albedo * hemi * shAmb * ambTint;
  col = col + albedo * vec3<f32>(0.640, 0.520, 0.360) * max(-N.y, 0.0) * 0.32 * shAmb;
  col = col + albedo * sunCol * ndl * sh;
  col = col + albedo * fillCol * max(dot(N, fillDir), 0.0) * 0.80;

  let fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.6);
  col = col + (albedo * 0.55 + vec3<f32>(0.05)) * rimCol * fres * max(dot(N, rimDir), 0.0) * 1.15;

  /* ---- specular: tight highlight + broad sheen ------------------------ */
  let nh = max(dot(N, H), 0.0);
  let tight = pow(nh, shininess) * specular * ndl * sh;
  let broad = pow(nh, max(shininess * 0.32, 2.0)) * specular * 0.16;
  let sheenW = 0.02 + min(specular, 0.5) * 0.30;
  let sheen = fres * sheenW * shAmb;
  col = col + sunCol * tight + sunCol * broad + rimCol * sheen;

  col = col + albedo * emissive;

  /* ---- ground: contact patch + horizon haze --------------------------- */
  // Keyed on world height and an up-facing normal, so whatever ground quad the
  // world supplies is dressed here and nowhere else. The patches and stripes
  // themselves are world.js's business; the shader only grounds the bike and
  // dissolves the far plane into the horizon band.
  if (in.worldPos.y < 0.01 && N.y > 0.9) {
    let contact = contactPatch(in.worldPos);
    col = col * mix(1.0, 0.26, contact);

    let dcam = length(in.worldPos - G.cameraPos.xyz);
    let haze = smoothstep(15.0, 52.0, dcam);
    // #ffe8b6 horizon, in linear space.
    let hazeCol = vec3<f32>(0.980, 0.800, 0.470);
    col = mix(col, hazeCol, haze);
  }

  // Gentle highlight rolloff so white surfaces keep their gradient, plus a
  // small exposure lift so a white albedo actually lands on the top level.
  col = col * 1.12;
  col = col / (vec3<f32>(1.0) + max(col, vec3<f32>(0.0)) * 0.07);
  col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

  // Posterise AFTER gamma, in art-pixel space: 6 levels per channel, dithered.
  col = posterize6(col, b);
  return vec4<f32>(col, 1.0);
}
`;

/* -------------------------------------------------------------- blit pass */
// Fullscreen triangle over the canvas. For each canvas pixel we find the art
// pixel under it, take its colour with a nearest sampler, and draw a 1px
// outline (in ART pixels, so it is `scale` canvas pixels thick) just outside
// every silhouette. The art target always covers the whole canvas; the last
// partial art pixel at an edge is cropped.
//
// The depth comparison is done on linear camera distance, reconstructed from
// the art depth through invViewProj, so the threshold is in world units and does
// not explode with the non-linear distribution of a 0..1 clip depth.

export const blitWGSL = /* wgsl */ `
${GLOBALS_WGSL}

struct Blit {
  scale : f32,
  offX  : f32,
  offY  : f32,
  artW  : f32,
  artH  : f32,
  pad0  : f32,
  pad1  : f32,
  pad2  : f32,
};

@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var<uniform> P : Blit;
@group(0) @binding(2) var artColor : texture_2d<f32>;
@group(0) @binding(3) var artSampler : sampler;
@group(0) @binding(4) var artDepth : texture_depth_2d;

const OUTLINE_RGB : vec3<f32> = vec3<f32>(0.1059, 0.1059, 0.1686); // #1b1b2b
const OUTLINE_MIN  : f32 = 0.020; // metres, at the near end
const OUTLINE_K    : f32 = 0.030; // metres per metre of depth
const OUTLINE_FAR  : f32 = 16.0;  // no outlines past this; the distance haze owns it

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  return vec4<f32>(p[vi], 0.0, 1.0);
}

// Linear camera distance of an art pixel, from its stored clip depth.
fn linDepth(ip : vec2<i32>, d : f32) -> f32 {
  let uv = (vec2<f32>(f32(ip.x) + 0.5, f32(ip.y) + 0.5)) / vec2<f32>(P.artW, P.artH);
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let h = G.invViewProj * vec4<f32>(ndc, d, 1.0);
  return length(h.xyz / h.w - G.cameraPos.xyz);
}

fn depthAt(ip : vec2<i32>) -> f32 {
  let q = clamp(ip, vec2<i32>(0, 0), vec2<i32>(i32(P.artW) - 1, i32(P.artH) - 1));
  return textureLoad(artDepth, q, 0);
}

@fragment
fn fs(@builtin(position) frag : vec4<f32>) -> @location(0) vec4<f32> {
  let ax = floor((frag.x - P.offX) / P.scale);
  let ay = floor((frag.y - P.offY) / P.scale);
  let inside = ax >= 0.0 && ay >= 0.0 && ax < P.artW && ay < P.artH;

  // Clamped so the sample is legal everywhere; textureSample stays in uniform
  // control flow (no derivative hazard); out-of-range pixels are decided after it.
  let ip = clamp(vec2<i32>(i32(ax), i32(ay)), vec2<i32>(0, 0),
                 vec2<i32>(i32(P.artW) - 1, i32(P.artH) - 1));
  let uv = (vec2<f32>(f32(ip.x) + 0.5, f32(ip.y) + 0.5)) / vec2<f32>(P.artW, P.artH);
  var col = textureSample(artColor, artSampler, uv).rgb;

  if (!inside) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  // 1px OUTER outline: drawn on the background pixel just outside a nearer
  // object, never on the object itself, so small things (flowers, birds, far
  // people) keep their colour and get a dark rim instead of turning black.
  //
  // A pixel is outlined when one of its 4 neighbours is clearly nearer than
  // it. The second difference (Laplacian) is required as well: it cancels the
  // smooth depth slope of the ground seen at a grazing angle, while a real
  // silhouette edge, where one side is metres behind the other, keeps a large
  // response. Sky pixels (depth 1.0) get outlined when they touch geometry.
  if (G.shadowParams.w < 0.5) {
    let d0 = depthAt(ip);
    let p1 = ip + vec2<i32>(-1, 0);
    let p2 = ip + vec2<i32>( 1, 0);
    let p3 = ip + vec2<i32>(0, -1);
    let p4 = ip + vec2<i32>(0,  1);
    let d1 = depthAt(p1);
    let d2 = depthAt(p2);
    let d3 = depthAt(p3);
    let d4 = depthAt(p4);
    let dn = min(min(d1, d2), min(d3, d4));
    if (dn < 0.99999) {
      // nearest neighbour that is geometry
      var ln = 1e9;
      if (d1 < 0.99999) { ln = min(ln, linDepth(p1, d1)); }
      if (d2 < 0.99999) { ln = min(ln, linDepth(p2, d2)); }
      if (d3 < 0.99999) { ln = min(ln, linDepth(p3, d3)); }
      if (d4 < 0.99999) { ln = min(ln, linDepth(p4, d4)); }
      if (ln < OUTLINE_FAR) {
        if (d0 >= 0.99999) {
          col = OUTLINE_RGB; // sky touching a silhouette
        } else {
          let l0 = linDepth(ip, d0);
          let l1 = linDepth(p1, d1);
          let l2 = linDepth(p2, d2);
          let l3 = linDepth(p3, d3);
          let l4 = linDepth(p4, d4);
          let j = max(abs(2.0 * l0 - l1 - l2), abs(2.0 * l0 - l3 - l4));
          let th = OUTLINE_MIN + OUTLINE_K * ln;
          if (l0 - ln > th && j > th) {
            col = OUTLINE_RGB;
          }
        }
      }
    }
  }

  return vec4<f32>(col, 1.0);
}
`;
