// pelican.js — the pelican and his bicycle, built procedurally and animated
// from one time value.
//
// World frame: the bike faces +X, wheels roll about Z, ground is y = 0.
// Wheel radius 0.42, wheel centres x = -0.60 / +0.78 at y = 0.42, saddle top
// 1.02. The bars steer about the head tube and the whole bike swerves and
// leans a little (see STEER). The bike does not move; the world scrolls toward -X at SCROLL_SPEED,
// so the wheels and the cranks turn at WHEEL_OMEGA = SCROLL_SPEED / 0.42.
//
// Everything here is layer 0: the pelican and the bike never scroll.
//
// Exports buildPelican() -> { parts, update(t) } where parts are the objects
// Renderer.setScene() takes: { mesh, color, model, params, layer }.

import { mat4, vec3 } from './math.js';
import { box, sphere, cylinder, torus, merge } from './geometry.js';

const M = mat4;
const TAU = Math.PI * 2;

/* ------------------------------------------------------------- constants */
// (HEAD_AXIS and rotAxis are defined after the frame points below.)

const WHEEL_R = 0.42;
const AXLE_Y = WHEEL_R;
const REAR = [-0.60, AXLE_Y, 0]; // tucked in close behind the seat tube
const FRONT = [0.78, AXLE_Y, 0]; // just far enough forward that the down tube clears the tyre
const BB = [-0.02, 0.36, 0]; // bottom bracket (crank centre)
const CRANK_LEN = 0.15;
const SADDLE_TOP = 1.02;

const SCROLL_SPEED = 2.2; // world units / s, matches world.js
const WHEEL_OMEGA = SCROLL_SPEED / WHEEL_R; // 5.238 rad/s: wheels and cranks

const SEAT_CLUSTER = [-0.22, 0.90, 0];
// Head tube: its bottom (the fork crown) sits ABOVE the front tyre (0.51 from
// the axle vs a 0.42 tyre), raked back ~18° like a real bike.
const HEAD_BOT = [0.58, 0.9, 0];
const HEAD_TOP = [0.58 - 0.11 * Math.tan((18 * Math.PI) / 180), 0.9 + 0.11, 0];
// Handlebar on a short stem: just above and slightly ahead of the head tube,
// where a normal bike has it. The pelican's wings reach forward to it.
const BAR_C = [0.585, 1.045, 0];
// How far the hands moved forward/up from the original grip (0.47, 1.02);
// the wing shapes below are stretched by this much toward the hand.
const HAND_SHIFT = [BAR_C[0] - 0.47, BAR_C[1] - 1.02];
// steering axis: up the head tube (raked back like a real bike)
const HEAD_AXIS = (() => {
  const d = [HEAD_TOP[0] - HEAD_BOT[0], HEAD_TOP[1] - HEAD_BOT[1], 0];
  const l = Math.hypot(d[0], d[1]);
  return [d[0] / l, d[1] / l, 0];
})();

// Rotation by `ang` about the unit axis `k` (Rodrigues), column-major.
function rotAxis(k, ang) {
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  const t = 1 - c;
  const [x, y, z] = k;
  const m = new Float32Array(16);
  m[0] = t * x * x + c;
  m[1] = t * x * y + s * z;
  m[2] = t * x * z - s * y;
  m[4] = t * x * y - s * z;
  m[5] = t * y * y + c;
  m[6] = t * y * z + s * x;
  m[8] = t * x * z + s * y;
  m[9] = t * y * z - s * x;
  m[10] = t * z * z + c;
  m[15] = 1;
  return m;
}
// Frame spread around the 6 cm tyres: tubes start narrow at the joints and
// spread to the dropouts, so their inner faces always clear the rubber.
const STAY_Z = 0.095; // rear dropouts (seat + chain stays end here)
const STAY_Z0 = 0.025; // where the stays leave the bottom bracket / seat cluster
const FORK_Z = 0.07; // front dropouts
const FORK_Z0 = 0.06; // fork blades at the crown
const DRIVE_Z = 0.07; // chainring / chain plane
const COG_Z = 0.062; // rear cog, inside the dropouts

// Steering and swerve. The bars turn about the head-tube axis by a slowly
// wandering angle (two sine waves, so it never looks mechanical). From that
// steer angle the bicycle model gives the heading change (yaw' = v/L * steer),
// the sideways drift it causes (z' = -v * yaw), and the lean into the turn.
// Every part (bike and rider) is then yawed, leaned about the ground line and
// shifted by that drift. Small numbers: a few degrees and a few centimetres.
const WHEELBASE = FRONT[0] - REAR[0];
const STEER = [
  { a: 0.055, w: (2 * Math.PI) / 4.3, p: 0.0 },
  { a: 0.025, w: (2 * Math.PI) / 1.9, p: 1.7 },
];
const LEAN_K = 0.75; // lean (rad) per rad of steer, a touch more than physics

/* --------------------------------------------------------------- helpers */

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

// Rotation whose +Y axis lands on the unit vector d (right-handed, det = +1).
function rotationFromY(d) {
  const y = vec3.normalize(d);
  let x = vec3.cross([0, 1, 0], y);
  if (Math.hypot(x[0], x[1], x[2]) < 1e-6) x = vec3.cross([0, 0, 1], y);
  x = vec3.normalize(x);
  const z = vec3.cross(x, y);
  const m = new Float32Array(16);
  m[0] = x[0];
  m[1] = x[1];
  m[2] = x[2];
  m[4] = y[0];
  m[5] = y[1];
  m[6] = y[2];
  m[8] = z[0];
  m[9] = z[1];
  m[10] = z[2];
  m[15] = 1;
  return m;
}

// Model matrix for a unit-height primitive (cylinder/cone along Y, centred on
// the origin) stretched from a to b, with cross-section radii sx (local X) and
// sz (local Z).
function seg(a, b, sx, sz) {
  const d = vec3.sub(b, a);
  const len = Math.hypot(d[0], d[1], d[2]) || 1e-6;
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  return mul(T(mid[0], mid[1], mid[2]), rotationFromY(d), S(sx, len, sz));
}

// Normal matrix (inverse transpose of the upper 3x3) as three column vectors.
function normalColumns(m) {
  const a = [m[0], m[1], m[2]];
  const b = [m[4], m[5], m[6]];
  const c = [m[8], m[9], m[10]];
  const det =
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0]);
  const s = Math.abs(det) > 1e-9 ? 1 / det : 1;
  return [
    vec3.scale(vec3.cross(b, c), s),
    vec3.scale(vec3.cross(c, a), s),
    vec3.scale(vec3.cross(a, b), s),
  ];
}

// Bake a transform into a mesh (positions + normals), so repeated elements can
// be merged into a single part instead of one part each.
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
    let ax = nx[0] * ox + ny[0] * oy + nz[0] * oz;
    let ay = nx[1] * ox + ny[1] * oy + nz[1] * oz;
    let az = nx[2] * ox + ny[2] * oy + nz[2] * oz;
    const l = Math.hypot(ax, ay, az) || 1;
    normals[k] = ax / l;
    normals[k + 1] = ay / l;
    normals[k + 2] = az / l;
  }
  return { positions, normals, indices: mesh.indices.slice() };
}

// Two-bone IK: knee position for a hip H, an ankle A, bone lengths L1/L2 and a
// pole vector that decides which way the joint bends.
function solveIK(H, A, L1, L2, pole) {
  const d = vec3.sub(A, H);
  let dist = Math.hypot(d[0], d[1], d[2]);
  const lo = Math.abs(L1 - L2) + 1e-3;
  const hi = L1 + L2 - 1e-3;
  dist = dist < lo ? lo : dist > hi ? hi : dist;
  const dir = vec3.normalize(d);
  const a = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist);
  const h = Math.sqrt(Math.max(L1 * L1 - a * a, 0));
  const base = vec3.add(H, vec3.scale(dir, a));
  let p = vec3.sub(pole, vec3.scale(dir, vec3.dot(pole, dir)));
  const pl = Math.hypot(p[0], p[1], p[2]);
  if (pl < 1e-6) p = vec3.normalize(vec3.cross(dir, [0, 0, 1]));
  else p = vec3.scale(p, 1 / pl);
  return vec3.add(base, vec3.scale(p, h));
}

/* ------------------------------------------------- mesh building helpers */

function makeMesh(positions, normals, indices) {
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
  };
}

function faceNormal(a, b, c) {
  const ux = b[0] - a[0];
  const uy = b[1] - a[1];
  const uz = b[2] - a[2];
  const vx = c[0] - a[0];
  const vy = c[1] - a[1];
  const vz = c[2] - a[2];
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const l = Math.hypot(nx, ny, nz) || 1;
  return [nx / l, ny / l, nz / l];
}

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// Push a flat-shaded quad, flipped so its normal agrees with `ref` (outward).
function pushQuad(pos, nor, idx, pts, ref) {
  let n = faceNormal(pts[0], pts[1], pts[2]);
  if (dot3(n, ref) < 0) {
    pts = [pts[3], pts[2], pts[1], pts[0]];
    n = [-n[0], -n[1], -n[2]];
  }
  const base = pos.length / 3;
  for (const p of pts) {
    pos.push(p[0], p[1], p[2]);
    nor.push(n[0], n[1], n[2]);
  }
  idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

function pushTri(pos, nor, idx, pts, ref) {
  let n = faceNormal(pts[0], pts[1], pts[2]);
  if (dot3(n, ref) < 0) {
    pts = [pts[2], pts[1], pts[0]];
    n = [-n[0], -n[1], -n[2]];
  }
  const base = pos.length / 3;
  for (const p of pts) {
    pos.push(p[0], p[1], p[2]);
    nor.push(n[0], n[1], n[2]);
  }
  idx.push(base, base + 1, base + 2);
}

// A tube swept along a polyline: smooth-shaded, elliptical rings (ry along the
// frame normal, rz along the binormal), capped at both ends. This is what
// gives the body its breast-and-back profile and the neck its tapered S.
function loft(spine, rings, seg = 18, capA = true, capB = true) {
  const n = spine.length;
  const frames = [];
  for (let i = 0; i < n; i++) {
    const a = spine[Math.max(0, i - 1)];
    const b = spine[Math.min(n - 1, i + 1)];
    let tx = b[0] - a[0];
    let ty = b[1] - a[1];
    let tz = b[2] - a[2];
    const tl = Math.hypot(tx, ty, tz) || 1;
    tx /= tl;
    ty /= tl;
    tz /= tl;
    const ref = Math.abs(ty) > 0.9 ? [0, 0, 1] : [0, 1, 0];
    let bx = ty * ref[2] - tz * ref[1];
    let by = tz * ref[0] - tx * ref[2];
    let bz = tx * ref[1] - ty * ref[0];
    const bl = Math.hypot(bx, by, bz) || 1;
    bx /= bl;
    by /= bl;
    bz /= bl;
    const nx = by * tz - bz * ty;
    const ny = bz * tx - bx * tz;
    const nz = bx * ty - by * tx;
    frames.push({ t: [tx, ty, tz], n: [nx, ny, nz], b: [bx, by, bz] });
  }

  const pos = [];
  const nor = [];
  for (let i = 0; i < n; i++) {
    const c = spine[i];
    const f = frames[i];
    const ry = rings[i][0];
    const rz = rings[i][1];
    for (let j = 0; j < seg; j++) {
      const a = (j / seg) * TAU;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      pos.push(
        c[0] + f.n[0] * ry * ca + f.b[0] * rz * sa,
        c[1] + f.n[1] * ry * ca + f.b[1] * rz * sa,
        c[2] + f.n[2] * ry * ca + f.b[2] * rz * sa
      );
      // ellipse normal: (rz*cos, ry*sin) in the (n, b) frame
      let ex = f.n[0] * rz * ca + f.b[0] * ry * sa;
      let ey = f.n[1] * rz * ca + f.b[1] * ry * sa;
      let ez = f.n[2] * rz * ca + f.b[2] * ry * sa;
      const el = Math.hypot(ex, ey, ez) || 1;
      nor.push(ex / el, ey / el, ez / el);
    }
  }

  const idx = [];
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < seg; j++) {
      const j2 = (j + 1) % seg;
      const a = i * seg + j;
      const b = i * seg + j2;
      const c = (i + 1) * seg + j2;
      const d = (i + 1) * seg + j;
      const fn = faceNormal(
        [pos[a * 3], pos[a * 3 + 1], pos[a * 3 + 2]],
        [pos[b * 3], pos[b * 3 + 1], pos[b * 3 + 2]],
        [pos[c * 3], pos[c * 3 + 1], pos[c * 3 + 2]]
      );
      const vn = [nor[a * 3], nor[a * 3 + 1], nor[a * 3 + 2]];
      if (dot3(fn, vn) >= 0) idx.push(a, b, c, a, c, d);
      else idx.push(a, c, b, a, d, c);
    }
  }

  // flat caps, welded to their own vertices so the rim stays crisp
  const cap = (station, ringBase, dir) => {
    const c = spine[station];
    const centre = pos.length / 3;
    pos.push(c[0], c[1], c[2]);
    nor.push(dir[0], dir[1], dir[2]);
    const first = pos.length / 3;
    for (let j = 0; j < seg; j++) {
      const k = (ringBase + j) * 3;
      pos.push(pos[k], pos[k + 1], pos[k + 2]);
      nor.push(dir[0], dir[1], dir[2]);
    }
    for (let j = 0; j < seg; j++) {
      const j2 = (j + 1) % seg;
      const p0 = [pos[centre * 3], pos[centre * 3 + 1], pos[centre * 3 + 2]];
      const p1 = [pos[(first + j) * 3], pos[(first + j) * 3 + 1], pos[(first + j) * 3 + 2]];
      const p2 = [pos[(first + j2) * 3], pos[(first + j2) * 3 + 1], pos[(first + j2) * 3 + 2]];
      let tri = [p0, p1, p2];
      const fn = faceNormal(tri[0], tri[1], tri[2]);
      if (dot3(fn, dir) < 0) tri = [p0, p2, p1];
      idx.push(centre, first + (dot3(fn, dir) < 0 ? j2 : j), first + (dot3(fn, dir) < 0 ? j : j2));
    }
  };
  if (capA) cap(0, 0, [-frames[0].t[0], -frames[0].t[1], -frames[0].t[2]]);
  if (capB) cap(n - 1, (n - 1) * seg, frames[n - 1].t);

  return makeMesh(pos, nor, idx);
}

// A closed slab between two edges (a = leading, b = trailing), `half` thick in
// Z. Flat shaded, every face oriented outward. Used for the wings, the tail
// and every feather.
function plate(stations, half) {
  const pos = [];
  const nor = [];
  const idx = [];
  const C = [];
  for (const st of stations) {
    const ax = st.a[0];
    const ay = st.a[1];
    const az = st.a[2];
    const bx = st.b[0];
    const by = st.b[1];
    const bz = st.b[2];
    C.push([
      [ax, ay, az - half],
      [bx, by, bz - half],
      [bx, by, bz + half],
      [ax, ay, az + half],
    ]);
  }
  const n = C.length;
  for (let i = 0; i < n - 1; i++) {
    // centre of the slab segment: all four corners of both stations, so a
    // face on the -Z side is oriented away from the middle of the plate
    const mid = [0, 0, 0];
    for (let q = 0; q < 4; q++) {
      mid[0] += (C[i][q][0] + C[i + 1][q][0]) / 8;
      mid[1] += (C[i][q][1] + C[i + 1][q][1]) / 8;
      mid[2] += (C[i][q][2] + C[i + 1][q][2]) / 8;
    }
    for (let k = 0; k < 4; k++) {
      const k2 = (k + 1) % 4;
      const p0 = C[i][k];
      const p1 = C[i][k2];
      const p2 = C[i + 1][k2];
      const p3 = C[i + 1][k];
      const ref = [
        (p0[0] + p1[0] + p2[0] + p3[0]) / 4 - mid[0],
        (p0[1] + p1[1] + p2[1] + p3[1]) / 4 - mid[1],
        (p0[2] + p1[2] + p2[2] + p3[2]) / 4 - mid[2],
      ];
      pushQuad(pos, nor, idx, [p0, p1, p2, p3], ref);
    }
  }
  const dir0 = [
    (C[1][0][0] + C[1][1][0]) / 2 - (C[0][0][0] + C[0][1][0]) / 2,
    (C[1][0][1] + C[1][1][1]) / 2 - (C[0][0][1] + C[0][1][1]) / 2,
    (C[1][0][2] + C[1][1][2]) / 2 - (C[0][0][2] + C[0][1][2]) / 2,
  ];
  const dirN = [
    (C[n - 1][0][0] + C[n - 1][1][0]) / 2 - (C[n - 2][0][0] + C[n - 2][1][0]) / 2,
    (C[n - 1][0][1] + C[n - 1][1][1]) / 2 - (C[n - 2][0][1] + C[n - 2][1][1]) / 2,
    (C[n - 1][0][2] + C[n - 1][1][2]) / 2 - (C[n - 2][0][2] + C[n - 2][1][2]) / 2,
  ];
  pushQuad(pos, nor, idx, C[0], [-dir0[0], -dir0[1], -dir0[2]]);
  pushQuad(pos, nor, idx, C[n - 1], dirN);
  return makeMesh(pos, nor, idx);
}

// A single tapered feather from base to tip, lying in the plane of its own
// direction (width taken across it), `half` thick.
function feather(base, tip, w0, w1, half) {
  const d = vec3.normalize(vec3.sub(tip, base));
  let perp = vec3.cross(d, [0, 0, 1]);
  if (Math.hypot(perp[0], perp[1], perp[2]) < 1e-6) perp = vec3.cross(d, [0, 1, 0]);
  perp = vec3.normalize(perp);
  const a0 = vec3.sub(base, vec3.scale(perp, w0));
  const b0 = vec3.add(base, vec3.scale(perp, w0));
  const a1 = vec3.sub(tip, vec3.scale(perp, w1));
  const b1 = vec3.add(tip, vec3.scale(perp, w1));
  return plate(
    [
      { a: a0, b: b0 },
      { a: a1, b: b1 },
    ],
    half
  );
}

// Catmull-Rom through control points, with a matching radius ramp. Returns
// { spine, rings } ready for loft().
function smooth(spinePts, radii, samples) {
  const p = (i) => spinePts[Math.max(0, Math.min(spinePts.length - 1, i))];
  const r = (i) => radii[Math.max(0, Math.min(radii.length - 1, i))];
  const spine = [];
  const rings = [];
  const segs = spinePts.length - 1;
  for (let s = 0; s <= samples; s++) {
    const u = (s / samples) * segs;
    const i = Math.min(segs - 1, Math.floor(u));
    const t = u - i;
    const p0 = p(i - 1);
    const p1 = p(i);
    const p2 = p(i + 1);
    const p3 = p(i + 2);
    const t2 = t * t;
    const t3 = t2 * t;
    const cr = (a, b, c, d) =>
      0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
    spine.push([
      cr(p0[0], p1[0], p2[0], p3[0]),
      cr(p0[1], p1[1], p2[1], p3[1]),
      cr(p0[2], p1[2], p2[2], p3[2]),
    ]);
    const rr = r(i) + (r(i + 1) - r(i)) * t;
    rings.push([rr, rr]);
  }
  return { spine, rings };
}

/* ---------------------------------------------------------------- colours */

const COL = {
  tyre: [0.169, 0.169, 0.169],
  rim: [0.788, 0.788, 0.788],
  spoke: [0.66, 0.66, 0.66],
  hub: [0.82, 0.82, 0.82],
  frame: [0.165, 0.616, 0.561],
  frameDark: [0.10, 0.42, 0.38],
  saddle: [0.239, 0.251, 0.357],
  grip: [0.15, 0.15, 0.17],
  chain: [0.42, 0.41, 0.39],
  metal: [0.69, 0.66, 0.62],
  body: [0.992, 0.992, 0.992],
  bodyShade: [0.84, 0.86, 0.90],
  wingShade: [0.70, 0.745, 0.825],
  beak: [0.949, 0.639, 0.235],
  pouch: [0.910, 0.537, 0.169],
  eye: [0.106, 0.106, 0.169],
  white: [1.0, 1.0, 1.0],
  feet: [0.949, 0.639, 0.235],
};

const PRM = {
  rubber: [0.10, 14, 0],
  metal: [0.95, 70, 0],
  paint: [0.55, 40, 0],
  leather: [0.18, 20, 0],
  feather: [0.16, 12, 0],
  tail: [0.16, 12, 0.30],
  beak: [0.40, 24, 0],
  eye: [1.40, 120, 0],
  shine: [0.30, 40, 0.45],
};

/* ----------------------------------------------------- drivetrain meshes */

const UNIT_BOX = box(1, 1, 1);

// A sprocket in the XY plane, axis Z: a ring, real teeth around it and a
// spider of arms. Baked into one mesh so it costs one part.
function sprocket(R, teeth, thickness, tubeR, arms) {
  const bits = [xform(torus(R - tubeR * 0.55, tubeR, 40, 8), Rx(Math.PI / 2))];
  const toothLen = (TAU * R) / teeth;
  for (let k = 0; k < teeth; k++) {
    const a = (k / teeth) * TAU;
    bits.push(
      xform(UNIT_BOX, mul(Rz(a), T(R + 0.007, 0, 0), S(0.032, toothLen * 0.52, thickness)))
    );
  }
  for (let k = 0; k < arms; k++) {
    const a = (k / arms) * TAU + 0.35;
    bits.push(
      xform(UNIT_BOX, mul(Rz(a), T(R * 0.48, 0, 0), S(R * 0.98, 0.020, thickness * 0.62)))
    );
  }
  return merge(bits);
}

// A thick chain: links laid along the closed loop that runs over the
// chainring and back around the rear cog, on the drive side.
function chainMesh(c1, r1, c2, r2, z, spacing) {
  const dx = c2[0] - c1[0];
  const dy = c2[1] - c1[1];
  const d = Math.hypot(dx, dy);
  const ux = dx / d;
  const uy = dy / d;
  const beta = Math.acos(Math.max(-0.999, Math.min(0.999, (r1 - r2) / d)));
  const rot = (a) => [ux * Math.cos(a) - uy * Math.sin(a), ux * Math.sin(a) + uy * Math.cos(a)];
  const [ax, ay] = rot(beta);
  const [bx, by] = rot(-beta);
  const T1a = [c1[0] + r1 * ax, c1[1] + r1 * ay];
  const T1b = [c1[0] + r1 * bx, c1[1] + r1 * by];
  const T2a = [c2[0] + r2 * ax, c2[1] + r2 * ay];
  const T2b = [c2[0] + r2 * bx, c2[1] + r2 * by];

  const pts = [];
  const line = (p, q, n) => {
    for (let i = 0; i <= n; i++) {
      const s = i / n;
      pts.push([p[0] + (q[0] - p[0]) * s, p[1] + (q[1] - p[1]) * s]);
    }
  };
  const arc = (c, r, a0, a1, n) => {
    for (let i = 0; i <= n; i++) {
      const a = a0 + ((a1 - a0) * i) / n;
      pts.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)]);
    }
  };
  const a1a = Math.atan2(T1a[1] - c1[1], T1a[0] - c1[0]);
  const a1b = Math.atan2(T1b[1] - c1[1], T1b[0] - c1[0]);
  const a2a = Math.atan2(T2a[1] - c2[1], T2a[0] - c2[0]);
  const a2b = Math.atan2(T2b[1] - c2[1], T2b[0] - c2[0]);

  line(T1b, T2b, 10); // top run
  let e = a2a;
  while (e < a2b) e += TAU;
  arc(c2, r2, a2b, e, 14); // around the back of the cog
  line(T2a, T1a, 10); // bottom run
  arc(c1, r1, a1a, a1b - TAU, 18); // around the back of the chainring

  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  }
  const total = cum[cum.length - 1];
  const count = Math.max(8, Math.round(total / spacing));
  const step = total / count;
  const bits = [];
  for (let k = 0; k < count; k++) {
    const s = k * step;
    let i = 1;
    while (i < cum.length - 1 && cum[i] < s) i++;
    const t = (s - cum[i - 1]) / Math.max(1e-6, cum[i] - cum[i - 1]);
    const px = pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t;
    const py = pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t;
    const ang = Math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0]);
    const thick = k % 2 === 0 ? 0.030 : 0.023;
    bits.push(xform(UNIT_BOX, mul(T(px, py, z), Rz(ang), S(step * 0.92, 0.027, thick))));
  }
  return merge(bits);
}

// A fan-shaped webbed foot: three toes joined by scalloped web, lying in XZ,
// the heel at the origin and the toes toward +X.
function webbedFoot(len, spread, thick) {
  const N = 24;
  const rim = [];
  for (let k = 0; k <= N; k++) {
    const a = -spread + (2 * spread * k) / N;
    const r = len * (0.88 + 0.12 * Math.cos(4 * Math.PI * a));
    rim.push([Math.cos(a) * r, Math.sin(a) * r]);
  }
  const pos = [];
  const nor = [];
  const idx = [];
  const top = (p) => [p[0], 0, p[1]];
  const bot = (p) => [p[0], -thick, p[1]];
  const heel = [0, 0, 0];
  const heelB = [0, -thick, 0];
  for (let k = 0; k < N; k++) {
    pushTri(pos, nor, idx, [heel, top(rim[k]), top(rim[k + 1])], [0, 1, 0]);
    pushTri(pos, nor, idx, [heelB, bot(rim[k]), bot(rim[k + 1])], [0, -1, 0]);
    const mx = (rim[k][0] + rim[k + 1][0]) / 2;
    const mz = (rim[k][1] + rim[k + 1][1]) / 2;
    pushQuad(
      pos,
      nor,
      idx,
      [top(rim[k]), top(rim[k + 1]), bot(rim[k + 1]), bot(rim[k])],
      [mx, 0, mz]
    );
  }
  // three toe ridges on top, so the toes read even at art resolution
  const toes = merge([
    xform(UNIT_BOX, mul(Ry(-0.5), T(len * 0.5, thick * 0.5, 0), S(len * 0.98, 0.020, 0.026))),
    xform(UNIT_BOX, mul(T(len * 0.52, thick * 0.5, 0), S(len * 1.02, 0.020, 0.026))),
    xform(UNIT_BOX, mul(Ry(0.5), T(len * 0.5, thick * 0.5, 0), S(len * 0.98, 0.020, 0.026))),
  ]);
  return merge([makeMesh(pos, nor, idx), toes]);
}

/* -------------------------------------------------------------- wings --- */

// Leading edge from shoulder to hand, stretched so the hand reaches the
// handlebar: each point moves by HAND_SHIFT scaled by how far along the wing
// it is (shoulder fixed, hand moves the full amount).
const WING_LEAD = [
  [-0.03, 1.415],
  [0.05, 1.345],
  [0.15, 1.255],
  [0.26, 1.175],
  [0.36, 1.115],
  [0.44, 1.075],
].map(([x, y]) => {
  const f = (x + 0.03) / 0.47;
  return [x + HAND_SHIFT[0] * f, y + HAND_SHIFT[1] * f];
});
const WING_CHORD = [0.15, 0.2, 0.23, 0.22, 0.18, 0.12];
const WING_Z = [0.185, 0.215, 0.235, 0.245, 0.24, 0.225];

// One continuous wing per side: a folded plate from the shoulder to the hand,
// its trailing edge fringed with feathers. Returns { main, fringe }.
function wingMeshes(s) {
  const stations = [];
  const perps = [];
  for (let i = 0; i < WING_LEAD.length; i++) {
    const a = WING_LEAD[Math.max(0, i - 1)];
    const b = WING_LEAD[Math.min(WING_LEAD.length - 1, i + 1)];
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    const l = Math.hypot(dx, dy) || 1;
    dx /= l;
    dy /= l;
    let px = -dy;
    let py = dx;
    if (px > 0) {
      px = -px;
      py = -py;
    }
    perps.push([px, py]);
    const L = WING_LEAD[i];
    stations.push({
      a: [L[0], L[1], s * WING_Z[i]],
      b: [L[0] + px * WING_CHORD[i], L[1] + py * WING_CHORD[i], s * WING_Z[i] * 0.9],
    });
  }
  const main = plate(stations, 0.024);
  const fringe = [];
  for (let i = 0; i < stations.length - 1; i++) {
    const t0 = stations[i].b;
    const t1 = stations[i + 1].b;
    const base = [(t0[0] + t1[0]) / 2, (t0[1] + t1[1]) / 2, (t0[2] + t1[2]) / 2];
    const p = perps[i];
    const len = 0.062 - i * 0.005;
    const tip = [base[0] + p[0] * len, base[1] + p[1] * len, base[2] * 0.95];
    const w0 = Math.hypot(t1[0] - t0[0], t1[1] - t0[1]) * 0.6;
    fringe.push(feather(base, tip, w0, 0.012, 0.014));
  }
  return { main, fringe: merge(fringe) };
}

// The primaries: the wing tip closes around the grip, four feathers fanning
// from above the bar round to below it.
function primaries(s) {
  const bits = [];
  for (let k = 0; k < 4; k++) {
    const a = 1.95 - k * 0.82;
    const tip = [BAR_C[0] + Math.cos(a) * 0.088, BAR_C[1] + Math.sin(a) * 0.088];
    const base = [0.445 + HAND_SHIFT[0], 1.095 + HAND_SHIFT[1] - k * 0.013];
    bits.push(
      feather([base[0], base[1], s * 0.222], [tip[0], tip[1], s * 0.2], 0.03, 0.01, 0.013)
    );
  }
  return merge(bits);
}

/* ------------------------------------------------------------------ scene */

export function buildPelican() {
  const parts = [];
  const entries = []; // { idx, base, parent }

  function add(mesh, color, params, base, parent) {
    const model = base || M.identity();
    parts.push({ mesh, color, params, model, layer: 0 });
    entries.push({ idx: parts.length - 1, base: model, parent: parent || null });
    return parts.length - 1;
  }

  /* ---------------------------------------------------------- meshes ---- */
  const mSphere = sphere(1, 24, 16);
  const mCyl = cylinder(1, 1, 1, 14);
  const mTaper = cylinder(0.62, 1, 1, 14);

  const mTyre = torus(WHEEL_R - 0.030, 0.030, 40, 10);
  const mRim = torus(WHEEL_R - 0.072, 0.020, 40, 8);
  const mHub = cylinder(0.048, 0.048, 2 * STAY_Z + 0.01, 12); // rear: dropout to dropout
  const mHubF = cylinder(0.044, 0.044, 2 * FORK_Z + 0.01, 12);

  // 16 spokes baked into one mesh: one draw call for the whole wheel.
  const spokeBits = [];
  for (let k = 0; k < 16; k++) {
    const a = (k / 16) * TAU;
    spokeBits.push(xform(UNIT_BOX, mul(Rz(a), T(0, 0.2, 0), S(0.010, 0.3, 0.008))));
  }
  const mSpokes = merge(spokeBits);

  const mChainring = sprocket(0.105, 15, 0.016, 0.014, 4);
  const mCog = sprocket(0.048, 9, 0.014, 0.01, 3);
  const mChain = chainMesh([BB[0], BB[1]], 0.115, [REAR[0], REAR[1]], 0.056, (DRIVE_Z + COG_Z) / 2, 0.034);

  /* ----------------------------------------------------------- wheels --- */
  const wheelBase = Rx(Math.PI / 2); // torus axis Y -> Z, cylinder axis Y -> Z
  for (const parent of ['rear', 'front']) {
    add(mTyre, COL.tyre, PRM.rubber, wheelBase, parent);
    add(mRim, COL.rim, PRM.metal, wheelBase, parent);
    add(parent === 'rear' ? mHub : mHubF, COL.hub, PRM.metal, wheelBase, parent);
    add(mSpokes, COL.spoke, PRM.metal, M.identity(), parent);
  }
  // rear cog rides on the hub, so it turns with the wheel
  add(mCog, COL.metal, PRM.metal, T(0, 0, COG_Z), 'rear');

  /* ------------------------------------------------------------ frame --- */
  const tube = (a, b, r, color, parent) =>
    add(mCyl, color, PRM.paint, seg(a, b, r, r), parent || null);

  // main triangle
  tube(SEAT_CLUSTER, HEAD_TOP, 0.036, COL.frame);
  tube(BB, HEAD_BOT, 0.044, COL.frame);
  tube(BB, SEAT_CLUSTER, 0.040, COL.frame);
  tube(HEAD_TOP, HEAD_BOT, 0.048, COL.frame);

  // headset + stem (turn with the bars)
  const capTop = [HEAD_TOP[0] + HEAD_AXIS[0] * 0.035, HEAD_TOP[1] + HEAD_AXIS[1] * 0.035, 0];
  add(mCyl, COL.metal, PRM.metal, seg(HEAD_TOP, capTop, 0.034, 0.034), 'steer');
  tube(HEAD_TOP, BAR_C, 0.028, COL.metal, 'steer');

  // fork blades and crown
  add(UNIT_BOX, COL.frameDark, PRM.paint, mul(T(HEAD_BOT[0], HEAD_BOT[1] - 0.01, 0), S(0.07, 0.045, 2 * FORK_Z0 + 0.04)), 'steer');
  for (const s of [-1, 1]) {
    tube([HEAD_BOT[0], HEAD_BOT[1] - 0.01, s * FORK_Z0], [FRONT[0], FRONT[1], s * FORK_Z], 0.022, COL.frame, 'steer');
  }

  // rear triangle: chain stays + seat stays, paired
  for (const s of [-1, 1]) {
    tube([BB[0], BB[1], s * STAY_Z0], [REAR[0], REAR[1], s * STAY_Z], 0.021, COL.frame);
    tube([SEAT_CLUSTER[0], SEAT_CLUSTER[1], s * STAY_Z0], [REAR[0], REAR[1], s * STAY_Z], 0.019, COL.frame);
    // dropout plates at the axle ends
    add(UNIT_BOX, COL.frameDark, PRM.paint, mul(T(REAR[0], REAR[1], s * STAY_Z), S(0.07, 0.05, 0.012)));
  }

  // seat post and saddle: the saddle top is at SADDLE_TOP, where the bird sits
  tube(SEAT_CLUSTER, [-0.245, 0.965, 0], 0.024, COL.metal);
  add(mSphere, COL.saddle, PRM.leather, mul(T(-0.275, SADDLE_TOP - 0.038, 0), S(0.175, 0.038, 0.105)));
  add(mSphere, COL.saddle, PRM.leather, mul(T(-0.11, SADDLE_TOP - 0.032, 0), S(0.095, 0.032, 0.058)));

  // handlebar, grips
  add(mCyl, COL.metal, PRM.metal, mul(T(BAR_C[0], BAR_C[1], 0), Rx(Math.PI / 2), S(0.02, 0.52, 0.02)), 'steer');
  for (const s of [-1, 1]) {
    add(mCyl, COL.grip, PRM.leather, mul(T(BAR_C[0], BAR_C[1], s * 0.19), Rx(Math.PI / 2), S(0.03, 0.14, 0.03)), 'steer');
  }

  /* -------------------------------------------------------- drivetrain --- */
  add(mChainring, COL.metal, PRM.metal, T(0, 0, DRIVE_Z), 'crank');
  add(mChain, COL.chain, PRM.metal, M.identity());

  // crank arms turn with the chainring; the pedals stay level (per frame)
  for (const s of [-1, 1]) {
    add(
      UNIT_BOX,
      COL.metal,
      PRM.metal,
      mul(Rz(s > 0 ? 0 : Math.PI), T(0, CRANK_LEN / 2, s * (DRIVE_Z + 0.017)), S(0.028, CRANK_LEN, 0.02)),
      'crank'
    );
  }
  const iPedal = [];
  const iFoot = [];
  const iLeg = [];
  const mFoot = webbedFoot(0.15, 0.5, 0.016);
  for (let s = 0; s < 2; s++) {
    iPedal.push(add(UNIT_BOX, COL.metal, PRM.metal, M.identity()));
    iFoot.push(add(mFoot, COL.feet, PRM.beak, M.identity()));
    iLeg.push({
      thigh: add(mTaper, COL.body, PRM.feather, M.identity()),
      shin: add(mTaper, COL.feet, PRM.beak, M.identity()),
      knee: add(mSphere, COL.body, PRM.feather, M.identity()),
      ankle: add(mSphere, COL.feet, PRM.beak, M.identity()),
    });
  }

  /* ---------------------------------------------------------- pelican --- */
  // Body: one lofted ellipsoid. The spine drops toward the breast and the
  // radii swell there, so the underside touches the saddle at x = -0.24 and
  // hangs full and low in front of it; the top line is the back.
  const BODY_SPINE = [
    [-0.54, 1.315, 0],
    [-0.44, 1.305, 0],
    [-0.34, 1.288, 0],
    [-0.24, 1.272, 0],
    [-0.13, 1.258, 0],
    [-0.02, 1.25, 0],
    [0.08, 1.252, 0],
    [0.16, 1.272, 0],
    [0.22, 1.3, 0],
    [0.27, 1.33, 0],
  ];
  const BODY_RINGS = [
    [0.05, 0.044],
    [0.12, 0.098],
    [0.19, 0.158],
    [0.252, 0.2],
    [0.282, 0.222],
    [0.292, 0.23],
    [0.286, 0.224],
    [0.24, 0.19],
    [0.168, 0.14],
    [0.1, 0.09],
  ];
  add(loft(BODY_SPINE, BODY_RINGS, 22), COL.body, PRM.feather, M.identity(), 'body');

  // tail: a fan of flat feathers swept back and up
  const tailBits = [];
  for (let j = 0; j < 7; j++) {
    const d = j - 3;
    tailBits.push(
      feather(
        [-0.46, 1.315 + Math.abs(d) * 0.005, d * 0.012],
        [-0.70 - Math.abs(d) * 0.016, 1.375 + Math.abs(d) * 0.02, d * 0.05],
        0.052,
        0.012,
        0.012
      )
    );
  }
  add(merge(tailBits), COL.bodyShade, PRM.tail, M.identity(), 'body');

  // Neck: a smooth tapered S — out of the body, back, then forward into the
  // skull. One loft, so there is no seam anywhere along it.
  const neck = smooth(
    [
      [0.25, 1.3, 0],
      [0.215, 1.4, 0],
      [0.16, 1.5, 0],
      [0.155, 1.57, 0],
      [0.2, 1.625, 0],
      [0.245, 1.665, 0],
    ],
    [0.135, 0.108, 0.086, 0.074, 0.074, 0.084],
    18
  );
  add(loft(neck.spine, neck.rings, 16), COL.body, PRM.feather, M.identity(), 'body');

  // Skull: bigger than the neck, with a small occipital bump at the back.
  add(mSphere, COL.body, PRM.feather, mul(T(0.26, 1.7, 0), S(0.145, 0.132, 0.118)), 'head');
  add(mSphere, COL.body, PRM.feather, mul(T(0.155, 1.72, 0), S(0.07, 0.075, 0.078)), 'head');

  // Upper mandible: long, flat, tapering to a hooked tip.
  const beakSpine = [
    [0.32, 1.702, 0],
    [0.42, 1.692, 0],
    [0.52, 1.681, 0],
    [0.62, 1.669, 0],
    [0.7, 1.658, 0],
    [0.745, 1.648, 0],
  ];
  const beakRings = [
    [0.05, 0.062],
    [0.044, 0.056],
    [0.036, 0.048],
    [0.026, 0.036],
    [0.016, 0.024],
    [0.006, 0.009],
  ];
  add(loft(beakSpine, beakRings, 14), COL.beak, PRM.beak, M.identity(), 'head');

  // Gular pouch: the sac slung under the bill, deepest just past the middle
  // and gathered to a point at the tip.
  const pouchSec = [
    [0.3, 1.668, 0.055, 0.048],
    [0.38, 1.663, 0.09, 0.06],
    [0.47, 1.658, 0.098, 0.062],
    [0.56, 1.652, 0.078, 0.05],
    [0.64, 1.646, 0.048, 0.032],
    [0.7, 1.642, 0.016, 0.012],
  ];
  add(
    loft(
      pouchSec.map(([x, top, ry]) => [x, top - ry * 0.95, 0]),
      pouchSec.map(([, , ry, rz]) => [ry, rz]),
      16
    ),
    COL.pouch,
    PRM.beak,
    M.identity(),
    'head'
  );
  // throat: joins the pouch to the neck
  add(mSphere, COL.pouch, PRM.beak, mul(T(0.265, 1.635, 0), S(0.085, 0.08, 0.068)), 'head');

  // eyes with a highlight
  for (const s of [-1, 1]) {
    add(mSphere, COL.eye, PRM.eye, mul(T(0.325, 1.735, s * 0.098), S(0.03, 0.03, 0.03)), 'head');
    add(mSphere, COL.white, PRM.shine, mul(T(0.335, 1.745, s * 0.114), S(0.011, 0.011, 0.011)), 'head');
  }

  // crest
  // crest: three short rounded tufts swept back, not a single dark spike
  const crestBits = [];
  for (let j = 0; j < 3; j++) {
    const z = (j - 1) * 0.026;
    crestBits.push(
      xform(mSphere, mul(T(0.185, 1.805, z), Rz(0.55), S(0.030, 0.062, 0.026)))
    );
  }
  add(merge(crestBits), COL.bodyShade, PRM.tail, M.identity(), 'head');

  // Wings: continuous feathered surfaces from the shoulder to the hand, the
  // primaries closing around the grips. Static, so the hands stay on the bar
  // while the body bobs; the shoulder is buried in the body.
  for (const s of [-1, 1]) {
    // The wing has to read against the white flank: a white plate on a white
    // body is invisible, so the panel is grey-blue and the tips darker still.
    const w = wingMeshes(s);
    // the wings follow the bars so the hands stay on the grips (the shoulder
    // end moves a few cm, buried inside the body)
    add(w.main, COL.bodyShade, PRM.feather, M.identity(), 'steer');
    add(w.fringe, COL.wingShade, PRM.feather, M.identity(), 'steer');
    add(primaries(s), COL.wingShade, PRM.feather, M.identity(), 'steer');
    add(mSphere, COL.body, PRM.feather, mul(T(0.445 + HAND_SHIFT[0], 1.075 + HAND_SHIFT[1], s * 0.222), S(0.052, 0.05, 0.036)), 'steer');
  }

  /* ----------------------------------------------------------- update --- */

  function pedalPos(phi, side) {
    return [BB[0] - CRANK_LEN * Math.sin(phi), BB[1] + CRANK_LEN * Math.cos(phi), side * (DRIVE_Z + 0.05)];
  }

  function update(t) {
    // one gear, one speed: cranks and wheels both turn at WHEEL_OMEGA, so the
    // tyres track the ground exactly and the bike never looks like it skids.
    const phi = -WHEEL_OMEGA * t;
    const spin = -WHEEL_OMEGA * t;
    const bob = 0.012 * Math.sin(2 * phi);
    const headBob = 0.006 * Math.sin(2 * phi + 1.1);

    // steer angle and the swerve it produces (see STEER)
    let steer = 0;
    let yaw = 0;
    let drift = 0;
    for (const c of STEER) {
      const a = c.w * t + c.p;
      steer += c.a * Math.sin(a);
      yaw += -(SCROLL_SPEED / WHEELBASE) * (c.a / c.w) * Math.cos(a);
      drift += ((SCROLL_SPEED * SCROLL_SPEED) / WHEELBASE) * (c.a / (c.w * c.w)) * Math.sin(a);
    }
    const lean = -LEAN_K * steer;
    // the rotation about the head-tube axis, through its bottom point
    const steerM = mul(T(HEAD_BOT[0], HEAD_BOT[1], 0), rotAxis(HEAD_AXIS, steer), T(-HEAD_BOT[0], -HEAD_BOT[1], 0));

    const parents = {
      rear: mul(T(REAR[0], REAR[1], REAR[2]), Rz(spin)),
      front: mul(steerM, T(FRONT[0], FRONT[1], FRONT[2]), Rz(spin)),
      steer: steerM,
      crank: mul(T(BB[0], BB[1], BB[2]), Rz(phi)),
      body: T(0, bob, 0),
      head: T(0, bob + headBob, 0),
    };

    for (const e of entries) {
      const p = e.parent ? parents[e.parent] : null;
      parts[e.idx].model = p ? M.multiply(p, e.base) : e.base;
    }

    // pedals level, 180 degrees apart; webbed feet on top of them
    for (let s = 0; s < 2; s++) {
      const side = s === 0 ? -1 : 1;
      const ang = phi + (s === 0 ? 0 : Math.PI);
      const p = pedalPos(ang, side);
      parts[iPedal[s]].model = mul(T(p[0], p[1], p[2]), S(0.1, 0.022, 0.064));
      parts[iFoot[s]].model = mul(
        T(p[0] - 0.035, p[1] + 0.012, p[2]),
        Ry(-side * 0.1),
        S(1, 1, 1)
      );

      // two-bone leg from the hip to the ankle above the foot
      const hip = [-0.1, 1.03 + bob, side * 0.105];
      const ankle = [p[0] - 0.015, p[1] + 0.055, p[2]];
      const knee = solveIK(hip, ankle, 0.42, 0.42, [1, 0.2, side * 0.25]);
      const L = iLeg[s];
      parts[L.thigh].model = seg(hip, knee, 0.058, 0.05);
      parts[L.shin].model = seg(knee, ankle, 0.034, 0.034);
      parts[L.knee].model = mul(T(knee[0], knee[1], knee[2]), S(0.05, 0.05, 0.05));
      parts[L.ankle].model = mul(T(ankle[0], ankle[1], ankle[2]), S(0.036, 0.036, 0.036));
    }

    // swerve: drift sideways, turn to the new heading, lean about the ground
    const root = mul(T(0, 0, drift), Ry(yaw), Rx(lean));
    for (const part of parts) part.model = M.multiply(root, part.model);
  }

  update(0);
  return { parts, update };
}
