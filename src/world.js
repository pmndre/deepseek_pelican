// world.js — the scrolling landscape the pelican rides through.
//
// The bike stands at the origin and never moves: the world slides toward -X at
// SCROLL_SPEED times the speed of the layer a piece lives on, and every item
// wraps through x in [-32, +32] so the strip is endless.
//
// Layers:
//   1  1.00  the ground and anything close to the road
//   2  1.00  trees, bushes, grass and flowers standing on the ground (they
//            must move with it, or they would slide over the grass)
//   3  0.30  far tree line and hills
//   4  0.10  clouds
//   The mountains and the sun are fixed: far enough away not to scroll.
//
// Exports buildWorld() -> { parts, update(t) } plus the sun direction used for
// lighting. update(t) rewrites every model matrix for time t in seconds.

import { mat4 } from './math.js';
import { box, sphere, cylinder, cone, merge } from './geometry.js';

const M = mat4;
const TAU = Math.PI * 2;

export const SCROLL_SPEED = 2.2;

// The one sun. SUN_DIR is the scene's original key light (renderer default,
// ~50° up, toward -X/+Z): main.js passes it to the renderer for lighting and
// shadows. A sun that high is above the camera's view (it sees up to ~30°),
// so the visible disc sits on the SAME compass heading but lower in the sky,
// at SUN_DISC_ELEV: shadows always point straight away from the sun you see.
const _L = [-0.55, 0.82, 0.42];
const _LL = Math.hypot(_L[0], _L[1], _L[2]);
export const SUN_DIR = [_L[0] / _LL, _L[1] / _LL, _L[2] / _LL];
const SUN_AZ = Math.atan2(SUN_DIR[2], SUN_DIR[0]);
export const SUN_DISC_ELEV = (22 * Math.PI) / 180;
const SUN_DISC_DIR = [
  Math.cos(SUN_DISC_ELEV) * Math.cos(SUN_AZ),
  Math.sin(SUN_DISC_ELEV),
  Math.cos(SUN_DISC_ELEV) * Math.sin(SUN_AZ),
];
const WRAP = 64;
const HALF = 32;

// Scroll speed per layer (fraction of SCROLL_SPEED).
const SP_GROUND = 1.0; // layer 1
const SP_NEAR = 1.0; // layer 2: stands on the ground, so moves with it
const SP_MID = 0.3; // layer 3
const SP_SKY = 0.1; // layer 4

const WIND = 1.15; // rad/s base frequency of the sway
const GROUND_Y = 0.004; // a hair above the pelican's own ground plane

/* ---------------------------------------------------------------- colours */
// sRGB hex exactly as the palette gives it; the shader linearises.

const hex = (h) => [
  parseInt(h.slice(1, 3), 16) / 255,
  parseInt(h.slice(3, 5), 16) / 255,
  parseInt(h.slice(5, 7), 16) / 255,
];

const COL = {
  grassLight: hex('#90be3a'),
  grassMid: hex('#7cb518'),
  grassDark: hex('#5c8a12'),
  dirt: hex('#c8a165'),
  stone: hex('#b0a99f'),
  leafA: hex('#3f9142'),
  leafB: hex('#2f7d32'),
  trunk: hex('#7a4a26'),
  trunkDark: hex('#5e3a1f'),
  birchBark: hex('#ebe6da'),
  leafDeep: hex('#2a6a2f'),
  leafLight: hex('#6aae45'),
  leafLime: hex('#8cc152'),
  pineDark: hex('#245c3a'),
  pineLight: hex('#3a7f4c'),
  autumnA: hex('#d9822b'),
  autumnB: hex('#e8b04a'),
  bush: hex('#4f9d3a'),
  flowerPink: hex('#ff8fa3'),
  flowerYellow: hex('#ffd166'),
  hillFar: hex('#9ccf7a'),
  hillNear: hex('#6fae4e'),
  cloud: hex('#ffffff'),
  cloudShade: hex('#dce9f5'),
  sun: hex('#ffd166'),
  rock: hex('#8a93a6'),
  rockAlt: hex('#9497a3'),
  rockDark: hex('#626a80'),
  snow: hex('#f4f8fc'),
  sunGlow: hex('#ffe9a8'),
};

const PRM = {
  ground: [0.02, 6, 0],
  flat: [0.03, 6, 0],
  leaf: [0.06, 10, 0],
  wood: [0.04, 8, 0],
  flower: [0.05, 8, 0.05],
  cloud: [0.02, 4, 0.62],
  sun: [0.0, 2, 0.9],
  glow: [0.0, 2, 0.55],
};

/* ---------------------------------------------------------------- helpers */

const T = (x, y, z) => M.translation(x, y, z);
const S = (x, y, z) => M.scaling(x, y, z);
const Rx = (a) => M.rotationX(a);
const Ry = (a) => M.rotationY(a);
const Rz = (a) => M.rotationZ(a);

function mul(...ms) {
  let out = ms[0];
  for (let i = 1; i < ms.length; i++) out = M.multiply(out, ms[i]);
  return out;
}

// Wrap into [-32, +32).
const wrap = (x) => ((((x + HALF) % WRAP) + WRAP) % WRAP) - HALF;

// Deterministic little LCG, so the layout is identical on every load.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// Normal matrix columns (inverse transpose of the upper 3x3).
function normalColumns(m) {
  const a = [m[0], m[1], m[2]];
  const b = [m[4], m[5], m[6]];
  const c = [m[8], m[9], m[10]];
  const det =
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0]);
  const s = Math.abs(det) > 1e-9 ? 1 / det : 1;
  const cross = (u, v) => [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
  return [cross(b, c).map((n) => n * s), cross(c, a).map((n) => n * s), cross(a, b).map((n) => n * s)];
}

// Bake a transform into a mesh so many small pieces can share one part.
function xform(mesh, m) {
  const src = mesh.positions;
  const n = src.length / 3;
  const positions = new Float32Array(src.length);
  const normals = new Float32Array(src.length);
  const [nx, ny, nz] = normalColumns(m);
  for (let i = 0; i < n; i++) {
    const k = i * 3;
    const x = src[k];
    const y = src[k + 1];
    const z = src[k + 2];
    positions[k] = m[0] * x + m[4] * y + m[8] * z + m[12];
    positions[k + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    positions[k + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    const ox = mesh.normals[k];
    const oy = mesh.normals[k + 1];
    const oz = mesh.normals[k + 2];
    const ax = nx[0] * ox + ny[0] * oy + nz[0] * oz;
    const ay = nx[1] * ox + ny[1] * oy + nz[1] * oz;
    const az = nx[2] * ox + ny[2] * oy + nz[2] * oz;
    const l = Math.hypot(ax, ay, az) || 1;
    normals[k] = ax / l;
    normals[k + 1] = ay / l;
    normals[k + 2] = az / l;
  }
  return { positions, normals, indices: mesh.indices.slice() };
}

// One flat quad in the XZ plane, facing +Y (CCW seen from above).
function quadXZ(x0, x1, z0, z1, y) {
  return {
    positions: new Float32Array([x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1]),
    normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    indices: new Uint32Array([0, 3, 2, 0, 2, 1]),
  };
}

/* ------------------------------------------------------------- mountains */

// One mountain range in unit space (main peak ~1 high, ~1 wide at the base,
// base at y = 0). Returns { rock, cliff, snow } flat-shaded meshes that share
// the same surface, so the three colours never z-fight.
function mountainMeshes(seed) {
  const r = rng(seed);
  const peaks = [{ x: 0, z: 0, h: 1, w: 1 }];
  const subs = 1 + Math.floor(r() * 2);
  for (let k = 0; k < subs; k++) {
    const side = k % 2 ? -1 : 1;
    peaks.push({
      x: side * (0.55 + r() * 0.35),
      z: (r() - 0.5) * 0.6,
      h: 0.5 + r() * 0.3,
      w: 0.6 + r() * 0.25,
    });
  }
  const out = { rock: [[], [], []], cliff: [[], [], []], snow: [[], [], []] };
  const SEG = 13;
  const RINGS = 6;
  const snowLine = Array.from({ length: SEG }, () => 0.6 + r() * 0.14);

  function tri(a, b, c, cx, cz, seg) {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const mx = (a[0] + b[0] + c[0]) / 3;
    const my = (a[1] + b[1] + c[1]) / 3;
    const mz = (a[2] + b[2] + c[2]) / 3;
    // outward = away from the peak's axis
    if (nx * (mx - cx) + nz * (mz - cz) + ny * 0.2 < 0) {
      const t = b;
      b = c;
      c = t;
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l;
    ny /= l;
    nz /= l;
    const bucket = my > snowLine[seg] ? out.snow : ny < 0.42 ? out.cliff : out.rock;
    const [P, N, I] = bucket;
    const base = P.length / 3;
    for (const v of [a, b, c]) {
      P.push(v[0], v[1], v[2]);
      N.push(nx, ny, nz);
    }
    I.push(base, base + 1, base + 2);
  }

  for (const pk of peaks) {
    // ring vertices with a noisy radius and height
    const rings = [];
    for (let j = 0; j < RINGS; j++) {
      const t = j / RINGS;
      const ring = [];
      for (let i = 0; i < SEG; i++) {
        const th = (i / SEG) * TAU + (r() - 0.5) * 0.25;
        const rad = pk.w * 0.55 * Math.pow(1 - t, 1.15) * (0.78 + r() * 0.44);
        const y = pk.h * t + (j > 0 ? (r() - 0.5) * 0.05 * pk.h : 0);
        ring.push([pk.x + Math.cos(th) * rad, Math.max(0, y), pk.z + Math.sin(th) * rad]);
      }
      rings.push(ring);
    }
    const apex = [pk.x + (r() - 0.5) * 0.06, pk.h, pk.z + (r() - 0.5) * 0.06];
    for (let j = 0; j < RINGS - 1; j++) {
      for (let i = 0; i < SEG; i++) {
        const i2 = (i + 1) % SEG;
        const a = rings[j][i], b = rings[j][i2], c = rings[j + 1][i2], d = rings[j + 1][i];
        tri(a, b, c, pk.x, pk.z, i);
        tri(a, c, d, pk.x, pk.z, i);
      }
    }
    const top = rings[RINGS - 1];
    for (let i = 0; i < SEG; i++) tri(top[i], top[(i + 1) % SEG], apex, pk.x, pk.z, i);
  }

  const mk = ([P, N, I]) => ({
    positions: new Float32Array(P),
    normals: new Float32Array(N),
    indices: new Uint32Array(I),
  });
  return { rock: mk(out.rock), cliff: mk(out.cliff), snow: mk(out.snow) };
}

/* ------------------------------------------------------------------ trees */

// Hash of a position, so vertices that coincide (seams, poles) get the same
// jitter and the low-poly shapes stay closed.
function hash3(x, y, z) {
  let h = Math.imul(Math.round(x * 997), 73856093) ^ Math.imul(Math.round(y * 997), 19349663) ^ Math.imul(Math.round(z * 997), 83492791);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

// Low-poly look: jitter each vertex radially (from `c`) by +-amt, then split
// every triangle into its own flat-shaded face.
function facet(mesh, amt = 0, c = [0, 0, 0]) {
  const P = mesh.positions;
  const I = mesh.indices;
  const pos = new Float32Array(I.length * 3);
  const nor = new Float32Array(I.length * 3);
  const idx = new Uint32Array(I.length);
  const jit = (k) => {
    const x = P[k * 3], y = P[k * 3 + 1], z = P[k * 3 + 2];
    const f = 1 + amt * (hash3(x, y, z) - 0.5) * 2;
    return [c[0] + (x - c[0]) * f, c[1] + (y - c[1]) * f, c[2] + (z - c[2]) * f];
  };
  for (let t = 0; t < I.length; t += 3) {
    const a = jit(I[t]), b = jit(I[t + 1]), d = jit(I[t + 2]);
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = d[0] - a[0], vy = d[1] - a[1], vz = d[2] - a[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    [a, b, d].forEach((v, k) => {
      pos.set(v, (t + k) * 3);
      nor.set([nx, ny, nz], (t + k) * 3);
      idx[t + k] = t + k;
    });
  }
  return { positions: pos, normals: nor, indices: idx };
}

// One tree variant, origin at the root. Species by index; everything else
// (height, lean, clump layout, colours) from the seed. Returns the trunk and
// a canopy split into a shaded (lower/inner) and a lit (upper) part.
function makeTree(i, seed) {
  const r = rng(seed);
  const SPECIES = ['pine', 'round', 'pine', 'birch', 'round', 'poplar', 'round', 'pine', 'autumn'];
  const kind = SPECIES[i % SPECIES.length];
  const sph = sphere(1, 7, 5);
  const trunk = [];
  const dark = [];
  const light = [];
  const lean = (r() - 0.5) * 0.12;

  const branch = (x0, y0, x1, y1, z1, rad) => {
    const dx = x1 - x0, dy = y1 - y0;
    const len = Math.hypot(dx, dy, z1);
    const ang = Math.atan2(dx, dy);
    const tilt = Math.atan2(z1, Math.hypot(dx, dy));
    return xform(
      cylinder(rad * 0.6, rad, len, 6),
      mul(T(x0, y0, 0), Rx(tilt), Rz(-ang), T(0, len / 2, 0))
    );
  };

  if (kind === 'pine') {
    const H = 2.6 + r() * 1.8;
    const tiers = 4 + Math.floor(r() * 3);
    trunk.push(xform(cylinder(0.07, 0.13, H * 0.45, 7), T(0, H * 0.22, 0)));
    for (let k = 0; k < tiers; k++) {
      const f = k / tiers;
      const rad = (0.95 - f * 0.7) * (0.9 + r() * 0.2);
      const h = 0.75 * (1 - f * 0.35);
      const y = H * 0.28 + f * H * 0.62 + h / 2;
      const c = xform(cone(rad, h, 9), mul(T(lean * y, y, 0), Ry(r() * TAU)));
      (k % 2 ? light : dark).push(facet(c, 0.12, [lean * y, y, 0]));
    }
  } else if (kind === 'poplar') {
    const H = 3.2 + r() * 1.4;
    trunk.push(xform(cylinder(0.06, 0.11, 0.9, 7), T(0, 0.45, 0)));
    const crown = xform(sph, mul(T(0, H * 0.58, 0), S(0.5, H * 0.44, 0.5)));
    dark.push(facet(crown, 0.14, [0, H * 0.58, 0]));
    const top = xform(sph, mul(T(0.06, H * 0.78, 0.05), S(0.36, H * 0.26, 0.36)));
    light.push(facet(top, 0.14, [0.06, H * 0.78, 0.05]));
  } else {
    // round, birch, autumn: trunk + 2-3 branches + a cloud of clumps
    const birch = kind === 'birch';
    const th = (birch ? 1.9 : 1.2) + r() * 0.6;
    const tr = birch ? 0.07 : 0.13 + r() * 0.04;
    trunk.push(xform(cylinder(tr * 0.65, tr, th, 7), mul(Rz(lean), T(0, th / 2, 0))));
    const top = [Math.sin(-lean) * th, th];
    const nb = 2 + Math.floor(r() * 2);
    const cx = top[0];
    const cy = th + (birch ? 0.45 : 0.6);
    for (let k = 0; k < nb; k++) {
      const a = (k / nb) * TAU + r();
      trunk.push(branch(top[0], th * 0.8, cx + Math.cos(a) * 0.45, cy + 0.1, Math.sin(a) * 0.45, tr * 0.5));
    }
    const n = (birch ? 5 : 7) + Math.floor(r() * 4);
    const spread = birch ? 0.5 : 0.75 + r() * 0.2;
    for (let k = 0; k < n; k++) {
      const a = r() * TAU;
      const rr = Math.sqrt(r()) * spread;
      const yy = (r() - 0.35) * (birch ? 0.9 : 0.7);
      const sz = (birch ? 0.28 : 0.36) + r() * (birch ? 0.16 : 0.26);
      const c = [cx + Math.cos(a) * rr, cy + yy, Math.sin(a) * rr];
      const m = xform(sph, mul(T(c[0], c[1], c[2]), S(sz * 1.1, sz * 0.9, sz)));
      (yy > 0.05 ? light : dark).push(facet(m, 0.16, c));
    }
    // a crown clump so the silhouette always has a top
    const crown = [cx, cy + 0.45, 0];
    light.push(facet(xform(sph, mul(T(crown[0], crown[1], 0), S(0.42, 0.36, 0.4))), 0.16, crown));
  }

  const leafCols = {
    pine: [COL.pineDark, COL.pineLight],
    round: r() < 0.5 ? [COL.leafDeep, COL.leafA] : [COL.leafB, COL.leafLight],
    birch: [COL.leafLight, COL.leafLime],
    poplar: [COL.leafDeep, COL.leafB],
    autumn: r() < 0.5 ? [COL.autumnA, COL.autumnB] : [hex('#c4552e'), COL.autumnA],
  }[kind];
  return {
    trunk: merge(trunk),
    trunkCol: kind === 'birch' ? COL.birchBark : r() < 0.5 ? COL.trunk : COL.trunkDark,
    dark: merge(dark),
    darkCol: leafCols[0],
    light: merge(light.length ? light : dark),
    lightCol: leafCols[1],
  };
}

/* ------------------------------------------------------------------ build */

export function buildWorld() {
  const parts = [];
  const entries = [];

  // A part plus the recipe update(t) uses to place it.
  // spec: { x0, x, y, z, speed, sway, phase, sx, sy, sz, rot, bob, fixed }
  function add(mesh, color, params, layer, spec) {
    const s = {
      x0: 0,
      x: 0,
      y: 0,
      z: 0,
      speed: 0,
      sway: 0,
      phase: 0,
      sx: 1,
      sy: 1,
      sz: 1,
      rot: 0,
      bob: 0,
      fixed: false,
      ...spec,
    };
    parts.push({ mesh, color, params, model: M.identity(), layer });
    entries.push({ idx: parts.length - 1, spec: s });
    return parts.length - 1;
  }

  /* -------------------------------------------------------------- meshes -- */

  const mSphere = sphere(1, 10, 7);
  const mCone = cone(1, 1, 8);
  const mBlade = box(1, 1, 1);
  const mDisc = cylinder(1, 1, 1, 10);

  // ---- grass tuft: five blades fanning out of one root, origin at the root
  function tuftMesh(seed) {
    const r = rng(seed);
    const bits = [];
    for (let k = 0; k < 4; k++) {
      const h = 0.22 + r() * 0.16;
      const lean = (k - 1.5) * 0.36 + (r() - 0.5) * 0.1;
      // Tip the blade away from the camera: its flat face then points at the
      // sky, so it takes the key light instead of going black.
      const back = -0.45 - r() * 0.25;
      bits.push(
        xform(
          mBlade,
          mul(Rz(lean), Rx(back), T((k - 1.5) * 0.05, h / 2, (r() - 0.5) * 0.05), S(0.11, h, 0.04))
        )
      );
    }
    return merge(bits);
  }

  const mTuftA = tuftMesh(11);
  const mTuftB = tuftMesh(29);
  const mTuftC = tuftMesh(47);

  // ---- flower patch: a few flowers of one kind, origin at the ground.
  // Each flower is a gently bent stem with a leaf or two, and a real head:
  // daisy / cosmos = a ring of petals round a centre disc, tulip = a cup of
  // upright petals, lavender = a spike of tiny buds. Returns three meshes
  // (green stems+leaves, petals, centres) so each gets its own colour.
  const mPetal = sphere(1, 7, 5);
  function flowerPatch(seed, kind) {
    const r = rng(seed);
    const stems = [];
    const petals = [];
    const centres = [];
    const n = 2 + Math.floor(r() * 3);
    for (let k = 0; k < n; k++) {
      const x = (r() - 0.5) * 0.42;
      const z = (r() - 0.5) * 0.3;
      const h = (kind === 'lavender' ? 0.3 : 0.2) + r() * 0.14;
      const bend = (r() - 0.5) * 0.35;
      const bendZ = (r() - 0.5) * 0.35;
      // two-piece stem, bending a little toward the top
      const midX = x + Math.sin(bend) * h * 0.25;
      const midZ = z + Math.sin(bendZ) * h * 0.25;
      const topX = x + Math.sin(bend) * h * 0.7;
      const topZ = z + Math.sin(bendZ) * h * 0.7;
      stems.push(xform(cylinder(0.013, 0.017, h * 0.5, 5), mul(T(x, h * 0.25, z), Rz(-bend * 0.5), Rx(bendZ * 0.5))));
      stems.push(xform(cylinder(0.011, 0.013, h * 0.52, 5), mul(T((midX + topX) / 2, h * 0.75, (midZ + topZ) / 2), Rz(-bend), Rx(bendZ))));
      // leaves: flattened, pointed lenses angled off the stem
      const nl = 1 + Math.floor(r() * 2);
      for (let l = 0; l < nl; l++) {
        const ay = r() * TAU;
        const ly = h * (0.15 + r() * 0.3);
        stems.push(
          xform(mPetal, mul(T(x, ly, z), Ry(ay), Rz(0.7), T(0.05, 0, 0), S(0.06, 0.012, 0.024)))
        );
      }
      const head = mul(T(topX, h, topZ), Rz(-bend * 1.2), Rx(bendZ * 1.2));
      if (kind === 'daisy' || kind === 'cosmos') {
        const np = kind === 'daisy' ? 9 : 6;
        const pw = kind === 'daisy' ? 0.024 : 0.034; // chunky enough to hold whole art pixels
        const pl = kind === 'daisy' ? 0.052 : 0.055;
        for (let q = 0; q < np; q++) {
          const a = (q / np) * TAU + r() * 0.2;
          petals.push(xform(mPetal, mul(head, Ry(a), Rz(0.25), T(pl * 0.9, 0.006, 0), S(pl, 0.02, pw))));
        }
        centres.push(xform(mPetal, mul(head, T(0, 0.016, 0), S(0.032, 0.022, 0.032))));
      } else if (kind === 'tulip') {
        for (let q = 0; q < 4; q++) {
          const a = (q / 4) * TAU + 0.4;
          petals.push(xform(mPetal, mul(head, Ry(a), T(0.022, 0.05, 0), Rz(-0.18), S(0.026, 0.065, 0.04))));
        }
        centres.push(xform(mPetal, mul(head, T(0, 0.02, 0), S(0.012, 0.012, 0.012))));
      } else {
        // lavender: a tapering spike of buds along the top of the stem
        for (let q = 0; q < 8; q++) {
          const f = q / 8;
          const bud = 0.028 * (1 - f * 0.45);
          petals.push(xform(mPetal, mul(head, T((r() - 0.5) * 0.01, -0.06 + f * 0.11, (r() - 0.5) * 0.01), S(bud, bud * 1.3, bud))));
        }
        centres.push(xform(mPetal, mul(head, T(0, -0.07, 0), S(0.006, 0.006, 0.006))));
      }
    }
    return { stems: merge(stems), petals: merge(petals), centres: merge(centres) };
  }

  // ---- trees: procedural low-poly species, see makeTree()
  const TREE_VARIANTS = Array.from({ length: 18 }, (_, i) => makeTree(i, 3301 + i * 97));

  // ---- bush: three squashed spheres, origin at the ground
  function bushMesh(seed) {
    const r = rng(seed);
    const bits = [];
    for (let k = 0; k < 3; k++) {
      const s = 0.24 + r() * 0.14;
      bits.push(
        xform(
          mSphere,
          mul(T((k - 1) * 0.3 + (r() - 0.5) * 0.08, s * 0.62, (r() - 0.5) * 0.2), S(s * 1.25, s * 0.8, s))
        )
      );
    }
    return merge(bits);
  }

  const mBushA = bushMesh(5);
  const mBushB = bushMesh(13);

  // ---- hill: a dome, most of it under the ground line
  const mHill = sphere(1, 12, 8);

  // ---- cloud: a lumpy bank of squashed spheres
  function cloudMesh(seed) {
    const r = rng(seed);
    const bits = [];
    const n = 3 + Math.floor(r() * 3);
    for (let k = 0; k < n; k++) {
      const s = 0.55 + r() * 0.55;
      bits.push(
        xform(
          mSphere,
          mul(
            T((k - n / 2) * 0.85 + (r() - 0.5) * 0.3, (r() - 0.4) * 0.35, (r() - 0.5) * 0.3),
            S(s * 1.35, s * 0.78, s)
          )
        )
      );
    }
    return merge(bits);
  }

  // ---- far tree line: a compact clump of small pines, merged into one part
  function farClump(seed) {
    const r = rng(seed);
    const bits = [];
    for (let k = 0; k < 5; k++) {
      const x = (k - 2) * 1.9 + (r() - 0.5) * 0.6;
      const z = (r() - 0.5) * 1.6;
      const s = 0.55 + r() * 0.35;
      bits.push(xform(mCone, mul(T(x, 0.75 * s, z), S(0.75 * s, 1.5 * s, 0.75 * s))));
      bits.push(xform(mCone, mul(T(x, 1.25 * s, z), S(0.5 * s, 1.1 * s, 0.5 * s))));
    }
    return merge(bits);
  }

  const mClumpA = farClump(3);
  const mClumpB = farClump(19);

  /* ------------------------------------------------------------- ground -- */
  // The lawn is built as coplanar bands of three greens: they tile the plane
  // edge to edge, so nothing z-fights and no band edge produces a depth jump
  // for the outline pass. The pattern repeats every 4 units and the wrap is 64,
  // so the whole strip can slide and wrap without a seam. 4 units (not 8) is
  // what the camera actually sees: the visible x window on the near lawn is
  // only ~5 units wide, so a longer period left the foreground a single flat
  // colour and the scroll was invisible.

  // The camera orbits all the way round, so the lawn runs to the horizon on
  // both sides of the road.
  const GZ0 = -96;
  const GZ1 = 96;
  const GX = 96;
  const PERIOD = 4; // divides WRAP (64) exactly -> the wrap jump is invisible

  const bandLight = [];
  const bandMid = [];
  const bandDark = [];
  for (let x = -GX; x < GX; x += PERIOD) {
    bandLight.push(quadXZ(x, x + 2.0, GZ0, GZ1, GROUND_Y));
    bandMid.push(quadXZ(x + 2.0, x + 3.0, GZ0, GZ1, GROUND_Y));
    bandDark.push(quadXZ(x + 3.0, x + 4.0, GZ0, GZ1, GROUND_Y));
  }
  add(merge(bandLight), COL.grassLight, PRM.ground, 1, { speed: SP_GROUND });
  add(merge(bandMid), COL.grassMid, PRM.ground, 1, { speed: SP_GROUND });
  add(merge(bandDark), COL.grassDark, PRM.ground, 1, { speed: SP_GROUND });

  // Dirt patches: three compact clumps, each wrapping on its own. They sit a
  // couple of millimetres proud of the lawn — enough to win the depth test,
  // too little to trip the outline pass.
  const dirtSeeds = [1, 2, 3, 4, 5];
  for (const sd of dirtSeeds) {
    const r = rng(sd * 97);
    const bits = [];
    for (let k = 0; k < 3; k++) {
      const rx = 1.1 + r() * 1.6;
      const rz = 0.7 + r() * 0.9;
      bits.push(xform(mDisc, mul(T((k - 1) * 2.6 + (r() - 0.5), 0.006, (r() - 0.5) * 2.2), S(rx, 0.02, rz))));
    }
    add(merge(bits), COL.dirt, PRM.flat, 1, {
      x0: -25 + sd * 13,
      z: -2.6 - sd * 1.6,
      speed: SP_GROUND,
    });
  }

  // Stones: clumps of small rocks, the nearest ones close enough to read as
  // individual pebbles in the foreground.
  for (let g = 0; g < 6; g++) {
    const r = rng(700 + g * 31);
    const bits = [];
    for (let k = 0; k < 4; k++) {
      const s = 0.1 + r() * 0.12;
      bits.push(
        xform(
          mSphere,
          mul(T((k - 1.5) * 1.1 + (r() - 0.5) * 0.4, s * 0.5, (r() - 0.5) * 1.4), S(s * 1.3, s * 0.7, s))
        )
      );
    }
    add(merge(bits), COL.stone, PRM.flat, 1, {
      x0: -30 + g * 11.5,
      z: -1.1 - (g % 3) * 1.9,
      speed: SP_GROUND,
    });
  }

  /* ---------------------------------------------------- weeds + flowers -- */
  // Scattered at random over both sides of the road instead of in rows:
  // random spot, size, turn, mesh and green. Two bands stay clear: the bike's
  // own line (|z| < 0.5) and the camera's orbit ring (|z| 3.4-5.4), so
  // nothing pops up in the lens. Within reach of the wheels they ride the
  // ground layer; further out the standing-object layer, like the trees.
  const rw = rng(2024);
  const scatterZ = (r, zMax) => {
    for (;;) {
      const z = (r() * 2 - 1) * zMax;
      const az = Math.abs(z);
      if (az < 0.5 || (az > 3.4 && az < 5.4)) continue;
      return z;
    }
  };
  const TUFT_MESHES = [mTuftA, mTuftB, mTuftC];
  const TUFT_COLS = [COL.grassLight, COL.grassMid, COL.grassDark];
  for (let i = 0; i < 130; i++) {
    const z = scatterZ(rw, 13);
    const near = Math.abs(z) < 3.4;
    const s = 0.6 + rw() * rw() * 1.1; // mostly small, a few big clumps
    add(TUFT_MESHES[Math.floor(rw() * 3)], TUFT_COLS[Math.floor(rw() * 3)], PRM.leaf, near ? 1 : 2, {
      x0: -HALF + rw() * WRAP,
      z,
      speed: near ? SP_GROUND : SP_NEAR,
      sway: 0.16 + rw() * 0.1,
      phase: rw() * TAU,
      rot: rw() * TAU,
      sx: s * (0.8 + rw() * 0.4),
      sy: s,
      sz: s * (0.8 + rw() * 0.4),
    });
  }

  // [kind, petal colour, centre colour]
  const FLOWER_TYPES = [
    ['daisy', hex('#f7f3ea'), COL.flowerYellow],
    ['daisy', hex('#fff2a8'), hex('#c98b2b')],
    ['cosmos', COL.flowerPink, COL.flowerYellow],
    ['cosmos', hex('#c77dff'), hex('#ffe066')],
    ['tulip', hex('#e63946'), hex('#ffd166')],
    ['tulip', COL.flowerYellow, hex('#e07a2f')],
    ['lavender', hex('#9b7fd4'), hex('#9b7fd4')],
  ];
  const FLOWER_PATCHES = [];
  for (let k = 0; k < 14; k++) {
    const [kind, petalCol, centreCol] = FLOWER_TYPES[k % FLOWER_TYPES.length];
    FLOWER_PATCHES.push({ ...flowerPatch(31 + k * 53, kind), petalCol, centreCol });
  }
  const rf = rng(4711);
  for (let i = 0; i < 46; i++) {
    const z = scatterZ(rf, 11);
    const near = Math.abs(z) < 3.4;
    const patch = FLOWER_PATCHES[Math.floor(rf() * FLOWER_PATCHES.length)];
    const s = 0.8 + rf() * 0.7;
    const spec = {
      x0: -HALF + rf() * WRAP,
      z,
      speed: near ? SP_GROUND : SP_NEAR,
      sway: 0.1 + rf() * 0.06, // gentle: big sways of tiny heads flicker in pixel mode
      phase: rf() * TAU,
      rot: rf() * TAU,
      sx: s,
      sy: s * (0.85 + rf() * 0.35),
      sz: s,
    };
    add(patch.stems, COL.leafA, PRM.leaf, near ? 1 : 2, spec);
    add(patch.petals, patch.petalCol, PRM.flower, near ? 1 : 2, spec);
    add(patch.centres, patch.centreCol, PRM.flower, near ? 1 : 2, spec);
  }

  /* --------------------------------------------------------------- trees -- */
  // Two belts of trees, one each side of the road, kept clear of the bands
  // where people walk and sheep graze. Each tree is three parts (trunk,
  // shaded canopy, lit canopy) sharing one placement so they sway together.
  function plantTree(v, x0, z, s, rot, phase) {
    const spec = { x0, z, speed: SP_NEAR, sx: s, sy: s, sz: s, rot };
    const leafSpec = { ...spec, sway: 0.03, phase };
    add(v.trunk, v.trunkCol, PRM.wood, 2, spec);
    add(v.dark, v.darkCol, PRM.leaf, 2, leafSpec);
    add(v.light, v.lightCol, PRM.leaf, 2, leafSpec);
  }
  const rt = rng(777);
  const FAR_TREES = 34;
  for (let i = 0; i < FAR_TREES; i++) {
    const v = TREE_VARIANTS[Math.floor(rt() * TREE_VARIANTS.length)];
    plantTree(
      v,
      -HALF + i * (WRAP / FAR_TREES) + rt() * 1.6,
      -6.0 - rt() * 10.0,
      0.8 + rt() * 0.55,
      rt() * TAU,
      rt() * TAU
    );
  }

  /* -------------------------------------------------------------- bushes -- */
  const rb = rng(4242);
  for (let i = 0; i < 12; i++) {
    const mesh = i % 2 ? mBushA : mBushB;
    const s = 0.7 + rb() * 0.6;
    add(mesh, COL.bush, PRM.leaf, 2, {
      x0: -HALF + i * (WRAP / 12) + rb() * 2.4,
      z: -5.8 - rb() * 3.2, // beyond the camera's orbit ring (|z| ~3.3-5.6 kept clear)
      speed: SP_NEAR,
      sway: 0.11,
      phase: rb() * TAU,
      sx: s,
      sy: s,
      sz: s,
    });
  }

  /* ------------------------------------------------ near side (z > 0) -- */
  // The camera circles the pelican at ~4 units, so the near side is dressed
  // from z = 5 outward: nothing ever stands between the lens and the rider.
  const rn = rng(5150);
  const NEAR_TREES = 30;
  for (let i = 0; i < NEAR_TREES; i++) {
    const v = TREE_VARIANTS[Math.floor(rn() * TREE_VARIANTS.length)];
    plantTree(
      v,
      -HALF + i * (WRAP / NEAR_TREES) + rn() * 1.6,
      9.8 + rn() * 8.0,
      0.8 + rn() * 0.55,
      rn() * TAU,
      rn() * TAU
    );
  }
  for (let i = 0; i < 12; i++) {
    const mesh = i % 2 ? mBushB : mBushA;
    const s = 0.7 + rn() * 0.6;
    add(mesh, COL.bush, PRM.leaf, 2, {
      x0: -HALF + i * (WRAP / 12) + rn() * 2.4,
      z: 6.0 + rn() * 2.8, // outside the camera's orbit ring
      speed: SP_NEAR,
      sway: 0.11,
      phase: rn() * TAU,
      sx: s,
      sy: s,
      sz: s,
    });
  }

  /* -------------------------------------------------- far tree line (L3) -- */
  for (let i = 0; i < 8; i++) {
    const near = i >= 4;
    const k = i % 4;
    add(k % 2 ? mClumpA : mClumpB, k % 2 ? COL.leafB : COL.leafA, PRM.leaf, 3, {
      x0: -24 + k * 16 + (near ? 8 : 0),
      z: near ? 14.0 + (k % 2) * 2.0 : -13.5 - (k % 2) * 2.0,
      speed: SP_MID,
      sx: 1.15,
      sy: 1.1,
      sz: 1.15,
    });
  }

  /* --------------------------------------------------------------- hills -- */
  const HILLS = [
    { x0: -26, z: -28, sx: 13, sy: 4.4, sz: 8, far: true },
    { x0: -8, z: -32, sx: 16, sy: 5.2, sz: 9, far: true },
    { x0: 6, z: -25, sx: 11, sy: 3.6, sz: 7, far: false },
    { x0: 19, z: -31, sx: 15, sy: 4.8, sz: 9, far: true },
    { x0: 31, z: -26, sx: 12, sy: 4.0, sz: 8, far: false },
    { x0: -20, z: 27, sx: 14, sy: 4.6, sz: 8, far: true },
    { x0: -2, z: 31, sx: 15, sy: 5.0, sz: 9, far: false },
    { x0: 14, z: 26, sx: 12, sy: 3.8, sz: 7, far: true },
    { x0: 28, z: 30, sx: 14, sy: 4.4, sz: 8, far: false },
  ];
  for (const h of HILLS) {
    add(mHill, h.far ? COL.hillFar : COL.hillNear, PRM.leaf, 3, {
      x0: h.x0,
      y: -h.sy * 0.4,
      z: h.z,
      speed: SP_MID,
      sx: h.sx,
      sy: h.sy,
      sz: h.sz,
    });
  }

  /* -------------------------------------------------------------- clouds -- */
  const rc = rng(9091);
  for (let i = 0; i < 18; i++) {
    const mesh = cloudMesh(100 + i);
    const side = i < 10 ? -1 : 1;
    const s = 0.9 + rc() * 0.9;
    add(mesh, i % 5 === 4 ? COL.cloudShade : COL.cloud, PRM.cloud, 4, {
      x0: -HALF + (i % 10) * (WRAP / 10) + rc() * 4.0 + (side > 0 ? 3.2 : 0),
      y: 4.4 + rc() * 3.0,
      z: side * (17 + rc() * 10),
      speed: SP_SKY + rc() * 0.05,
      sx: s,
      sy: s * 0.9,
      sz: s,
      bob: 0.12 + rc() * 0.1,
      phase: rc() * TAU,
    });
  }

  /* ----------------------------------------------------------- mountains -- */
  // A ring of snow-capped ranges on the horizon all the way round, fixed in
  // place (too far away to scroll). Each range is a main peak plus one or two
  // lower shoulders. Every peak is a jagged, flat-shaded lathe: noisy radius
  // per vertex, a wobbly apex, and the faces split three ways — snow above an
  // irregular snowline, dark rock on steep faces, light rock elsewhere.
  for (let i = 0; i < 26; i++) {
    const r = rng(8080 + i * 131);
    const a = (i / 26) * TAU + (r() - 0.5) * 0.16;
    const dist = 70 + r() * 14;
    const H = 12 + r() * 10;
    const W = H * (1.0 + r() * 0.35);
    const m = mountainMeshes(9000 + i * 17);
    const spec = {
      fixed: true,
      x: Math.cos(a) * dist,
      y: -1.2, // base sinks into the horizon haze
      z: Math.sin(a) * dist,
      rot: r() * TAU,
      sx: W,
      sy: H,
      sz: W,
    };
    add(m.rock, i % 2 ? COL.rock : COL.rockAlt, PRM.flat, 3, spec);
    add(m.cliff, COL.rockDark, PRM.flat, 3, spec);
    add(m.snow, COL.snow, PRM.flat, 3, spec);
  }

  /* ----------------------------------------------------------------- sun -- */
  // Placed on the light's compass heading (see SUN_DISC_DIR). The glow sits a
  // little further out on the same ray so the disc stays in front.
  const SUN_D = 46;
  add(mSphere, COL.sunGlow, PRM.glow, 4, {
    fixed: true,
    x: SUN_DISC_DIR[0] * (SUN_D + 4),
    y: SUN_DISC_DIR[1] * (SUN_D + 4),
    z: SUN_DISC_DIR[2] * (SUN_D + 4),
    sx: 3.9,
    sy: 3.9,
    sz: 3.9,
  });
  add(mSphere, COL.sun, PRM.sun, 4, {
    fixed: true,
    x: SUN_DISC_DIR[0] * SUN_D,
    y: SUN_DISC_DIR[1] * SUN_D,
    z: SUN_DISC_DIR[2] * SUN_D,
    sx: 2.4,
    sy: 2.4,
    sz: 2.4,
  });

  /* -------------------------------------------------------------- update -- */

  function place(spec, t) {
    const x = spec.fixed ? spec.x : wrap(spec.x0 - SCROLL_SPEED * spec.speed * t);
    let y = spec.y;
    if (spec.bob) y += spec.bob * Math.sin(t * 0.55 + spec.phase);
    let m = T(x, y, spec.z);
    if (spec.sway) {
      // Wind: a slow lean, a medium wobble and a quick shiver from gusts, in
      // two directions so things shake rather than tick like a metronome.
      // Gusts roll across the field (phase follows x) and come and go.
      const gust = 0.6 + 0.4 * Math.sin(t * 0.37 + x * 0.08);
      const w =
        0.62 * Math.sin(t * WIND + spec.phase) +
        0.24 * Math.sin(t * 2.35 + spec.phase * 1.7) +
        0.14 * gust * Math.sin(t * 7.3 + spec.phase * 2.9);
      const w2 = 0.5 * Math.sin(t * 1.7 + spec.phase * 1.3) + 0.2 * gust * Math.sin(t * 6.1 + spec.phase);
      m = M.multiply(m, Rz(spec.sway * w * (0.7 + 0.5 * gust)));
      m = M.multiply(m, Rx(spec.sway * w2 * 0.6));
    }
    if (spec.rot) m = M.multiply(m, Ry(spec.rot));
    if (spec.sx !== 1 || spec.sy !== 1 || spec.sz !== 1) m = M.multiply(m, S(spec.sx, spec.sy, spec.sz));
    return m;
  }

  function update(t) {
    for (let k = 0; k < entries.length; k++) {
      const e = entries[k];
      parts[e.idx].model = place(e.spec, t);
    }
  }

  update(0);
  return { parts, update };
}
