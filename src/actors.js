// actors.js — the living cast of the world: people walking, cats, sheep,
// birds, airplanes with contrails, and wind-blown leaves.
//
// Exports buildActors() -> { parts, update(t) }. Every part is
// { mesh, color, model, params, layer } as the renderer expects. `layer`
// picks the scroll speed (same scheme as world.js); each actor scrolls with
//   x = ((x0 - SCROLL_SPEED * t * LAYER_SPEED[layer] + 32) mod 64) - 32
// plus its own motion, so it loops seamlessly through x in [-32, 32).
//
// Nothing is shared between modules: each actor computes its own position from
// t, and update(t) rewrites every model matrix from scratch.

import { mat4 } from './math.js';
import { box, sphere, cone, merge } from './geometry.js';

const M = mat4;
const TAU = Math.PI * 2;

const SCROLL_SPEED = 2.2; // world units per second (world convention)
const LAYER_SPEED = [0, 1.0, 1.0, 0.30, 0.10]; // per layer 0..4 (2 stands on the ground: same as 1)
const HALF = 32;
const SPAN = 64;

// Walking gait with planted feet. Each leg is a straight 0.81 rod from the
// hip. Over one gait cycle (period 1/f) each leg spends half on the ground
// (stance) and half swinging forward. During stance the foot stays fixed on
// the ground while the hip travels half a stride over it, so the walker
// really treads instead of gliding; the hip dips to keep the stance foot on
// the ground. Stride per step = speed / (2 f).
const LEG_LEN = 0.81;
// Cat trot: legs 0.22 long swinging +-0.42 rad -> step 0.18, cycle 0.36.
const CAT_CADENCE = 9.0;
const CAT_TROT_SPEED = (2 * 2 * 0.22 * Math.sin(0.42) * CAT_CADENCE) / TAU;
const BIRD_SPEED = 7.5; // world units per second, the birds' own airspeed

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

// sRGB hex -> 0..1 floats. The shader linearizes, exactly like pelican.js.
const rgb = (hex) => [
  ((hex >> 16) & 255) / 255,
  ((hex >> 8) & 255) / 255,
  (hex & 255) / 255,
];

const PAL = {
  shirts: [0xef476f, 0x118ab2, 0xffd166, 0x06d6a0].map(rgb),
  trousers: rgb(0x4a6fa5),
  shoe: rgb(0x6b4f3a),
  skin: rgb(0xf1c27d),
  catA: rgb(0xf4a261),
  catB: rgb(0x3d405b),
  catEye: rgb(0xffe066),
  eyeWhite: rgb(0xfbfbf7),
  pupil: rgb(0x1b1b2b),
  lips: rgb(0xb5524a),
  bird: rgb(0x3d405b),
  leaves: [0x2f7d32, 0x3f9142, 0x5c8a12].map(rgb),
  beak: rgb(0xffd166),
  wool: rgb(0xf3efe6),
  sheepFace: rgb(0x3b3632),
};

// Scroll position of an actor that started at x0, on a given layer.
function wrapX(x0, t, layer) {
  const x = x0 - SCROLL_SPEED * t * LAYER_SPEED[layer];
  return (((x + HALF) % SPAN) + SPAN) % SPAN - HALF;
}

/* ------------------------------------------------------- mesh utilities */

// Ellipsoid: a unit sphere scaled per axis, with exact ellipsoid normals.
function ellipsoid(rx, ry, rz, seg = 12, rings = 8) {
  const s = sphere(1, seg, rings);
  const n = s.positions.length;
  const pos = new Float32Array(n);
  const nor = new Float32Array(n);
  const ix = 1 / (rx * rx);
  const iy = 1 / (ry * ry);
  const iz = 1 / (rz * rz);
  for (let i = 0; i < n; i += 3) {
    const x = s.positions[i] * rx;
    const y = s.positions[i + 1] * ry;
    const z = s.positions[i + 2] * rz;
    pos[i] = x;
    pos[i + 1] = y;
    pos[i + 2] = z;
    let ax = x * ix;
    let ay = y * iy;
    let az = z * iz;
    const l = Math.hypot(ax, ay, az) || 1;
    ax /= l;
    ay /= l;
    az /= l;
    nor[i] = ax;
    nor[i + 1] = ay;
    nor[i + 2] = az;
  }
  return { positions: pos, normals: nor, indices: s.indices };
}

// Bake a matrix into a mesh (positions + normals), for merged compound meshes.
function xform(mesh, mat) {
  const n = mesh.positions.length;
  const pos = new Float32Array(n);
  const nor = new Float32Array(n);
  for (let i = 0; i < n; i += 3) {
    const x = mesh.positions[i];
    const y = mesh.positions[i + 1];
    const z = mesh.positions[i + 2];
    pos[i] = mat[0] * x + mat[4] * y + mat[8] * z + mat[12];
    pos[i + 1] = mat[1] * x + mat[5] * y + mat[9] * z + mat[13];
    pos[i + 2] = mat[2] * x + mat[6] * y + mat[10] * z + mat[14];
    const nx = mesh.normals[i];
    const ny = mesh.normals[i + 1];
    const nz = mesh.normals[i + 2];
    let ax = mat[0] * nx + mat[4] * ny + mat[8] * nz;
    let ay = mat[1] * nx + mat[5] * ny + mat[9] * nz;
    let az = mat[2] * nx + mat[6] * ny + mat[10] * nz;
    const l = Math.hypot(ax, ay, az) || 1;
    nor[i] = ax / l;
    nor[i + 1] = ay / l;
    nor[i + 2] = az / l;
  }
  return { positions: pos, normals: nor, indices: mesh.indices };
}

const at = (mesh, x, y, z) => xform(mesh, T(x, y, z));

/* -------------------------------------------------------------- meshes */

// A point at distance r from the head centre, in direction (x, y, z).
function faceDir(x, y, z, r) {
  const l = Math.hypot(x, y, z) || 1;
  return [(x / l) * r, (y / l) * r, (z / l) * r];
}

// Hairstyles, in head space (head centre at the origin, radius 0.145, facing
// -X). Every style starts from a rounded cap that leaves the forehead, eyes
// and ears clear, then adds its own shape.
function hairCap(rx = 0.158, ry = 0.115) {
  return merge([
    at(ellipsoid(rx, ry, rx, 12, 8), 0.018, 0.058, 0), // crown
    at(ellipsoid(0.13, 0.13, 0.15, 10, 8), 0.05, -0.005, 0), // back of the head
  ]);
}
const HAIR_STYLES = [
  // 0 short crop
  hairCap(),
  // 1 long, straight: falls past the shoulders at the back and sides
  merge([
    hairCap(),
    at(ellipsoid(0.09, 0.23, 0.16, 10, 8), 0.085, -0.13, 0),
    at(ellipsoid(0.05, 0.19, 0.045, 8, 6), 0.0, -0.08, 0.135),
    at(ellipsoid(0.05, 0.19, 0.045, 8, 6), 0.0, -0.08, -0.135),
  ]),
  // 2 bun on top
  merge([hairCap(0.155, 0.11), at(sphere(0.075, 10, 8), 0.07, 0.15, 0)]),
  // 3 ponytail
  merge([
    hairCap(),
    at(sphere(0.04, 8, 6), 0.15, 0.03, 0),
    xform(ellipsoid(0.045, 0.15, 0.045, 8, 6), mul(T(0.2, -0.09, 0), Rz(0.35))),
  ]),
  // 4 curly / afro: a fuller cap studded with curls
  merge([
    hairCap(0.172, 0.13),
    ...Array.from({ length: 14 }, (_, k) => {
      const a = (k / 14) * Math.PI * 2;
      const up = k % 2 ? 0.55 : 0.2;
      const d = faceDir(Math.cos(a) * 0.9 + 0.25, up + 0.35, Math.sin(a), 0.165);
      return at(sphere(0.05, 7, 5), d[0], d[1] + 0.03, d[2]);
    }),
  ]),
  // 5 buzz cut: a thin, close cap
  merge([at(ellipsoid(0.15, 0.1, 0.15, 12, 8), 0.02, 0.06, 0)]),
];
const HAIR_COLORS = [0x2b2622, 0x5a3a22, 0xd9b45f, 0xa8492a, 0xb9b9b9, 0x3d2b1f].map(rgb);

// People are built facing -X (the direction the world scrolls), so the camera
// sees their left flank in profile: the best angle for a leg swing.
const PERSON_MESH = {
  // Rounded torso: a broad chest and a narrower waist. People face -X, so the
  // shoulder width runs along Z (the old box had width and depth swapped,
  // which left the arms hanging beside a thin torso).
  torso: merge([
    at(ellipsoid(0.13, 0.2, 0.2, 12, 8), 0, 0.09, 0),
    at(ellipsoid(0.115, 0.2, 0.165, 12, 8), 0, -0.09, 0),
  ]),
  head: sphere(0.145, 12, 10),
  // Neck: bridges the gap between the top of the torso (1.43) and the head.
  neck: ellipsoid(0.06, 0.07, 0.06, 8, 6),
  // Face, in head space (head centre at the origin, facing -X). Eyes sit on
  // the sphere surface: white eyeballs with dark pupils just in front.
  eyeWhites: merge([
    at(sphere(0.027, 8, 6), ...faceDir(-0.86, 0.14, 0.40, 0.128)),
    at(sphere(0.027, 8, 6), ...faceDir(-0.86, 0.14, -0.40, 0.128)),
  ]),
  pupils: merge([
    at(sphere(0.016, 8, 6), ...faceDir(-0.86, 0.14, 0.38, 0.150)),
    at(sphere(0.016, 8, 6), ...faceDir(-0.86, 0.14, -0.38, 0.150)),
  ]),
  brows: merge([
    xform(box(0.016, 0.014, 0.06), mul(T(...faceDir(-0.85, 0.32, 0.40, 0.142)), Rx(0.15))),
    xform(box(0.016, 0.014, 0.06), mul(T(...faceDir(-0.85, 0.32, -0.40, 0.142)), Rx(-0.15))),
  ]),
  // Nose, ears: skin coloured.
  skinBits: merge([
    at(ellipsoid(0.03, 0.035, 0.025, 8, 6), ...faceDir(-1.0, -0.05, 0, 0.145)),
    at(ellipsoid(0.025, 0.04, 0.02, 8, 6), 0.0, -0.005, 0.142),
    at(ellipsoid(0.025, 0.04, 0.02, 8, 6), 0.0, -0.005, -0.142),
  ]),
  // A small smile: three tiny boxes on a curve below the nose.
  mouth: merge([
    at(box(0.014, 0.014, 0.03), ...faceDir(-0.90, -0.42, 0, 0.142)),
    at(box(0.014, 0.014, 0.026), ...faceDir(-0.90, -0.38, 0.22, 0.142)),
    at(box(0.014, 0.014, 0.026), ...faceDir(-0.90, -0.38, -0.22, 0.142)),
  ]),
  // hair: see HAIR_STYLES (head space, one mesh per style)
  arm: ellipsoid(0.052, 0.27, 0.052, 8, 6),
  // Pelvis in trouser colour: closes the gap between the waist and the legs,
  // whose tops sink into it.
  hips: ellipsoid(0.125, 0.12, 0.175, 10, 8),
  leg: ellipsoid(0.072, 0.44, 0.072, 8, 6),
  foot: box(0.25, 0.07, 0.14),
};

// Rounded limbs and tail (ellipsoids, not planks), and two small eyes sitting
// on the face; every joint starts inside the body so the cat reads as one
// animal instead of a pile of boxes.
const CAT_EYE_DIR = [-0.832, 0.223, 0.508]; // unit, forward-up-side on the head
const CAT_MESH = {
  body: ellipsoid(0.22, 0.115, 0.11, 12, 8),
  sitBody: ellipsoid(0.13, 0.17, 0.12, 12, 8),
  head: sphere(0.095, 10, 8),
  ear: cone(0.042, 0.09, 6),
  leg: ellipsoid(0.03, 0.12, 0.03, 8, 6),
  foreleg: ellipsoid(0.028, 0.13, 0.028, 8, 6),
  tail: ellipsoid(0.09, 0.024, 0.024, 8, 6),
  tailTip: ellipsoid(0.075, 0.021, 0.021, 8, 6),
  eyes: merge([
    at(sphere(0.017, 8, 6), CAT_EYE_DIR[0] * 0.088, CAT_EYE_DIR[1] * 0.088, CAT_EYE_DIR[2] * 0.088),
    at(sphere(0.017, 8, 6), CAT_EYE_DIR[0] * 0.088, CAT_EYE_DIR[1] * 0.088, -CAT_EYE_DIR[2] * 0.088),
  ]),
};

// Sheep: a lumpy wool body (merged spheres), a dark head and four dark legs,
// built facing -X with the feet on the ground.
const SHEEP_MESH = {
  wool: merge([
    at(ellipsoid(0.34, 0.22, 0.22, 12, 8), 0, 0.5, 0),
    at(sphere(0.17, 8, 6), -0.16, 0.6, 0.07),
    at(sphere(0.17, 8, 6), 0.14, 0.62, -0.06),
    at(sphere(0.16, 8, 6), 0.02, 0.64, 0.1),
    at(sphere(0.15, 8, 6), 0.22, 0.52, 0.1),
  ]),
  head: merge([
    ellipsoid(0.12, 0.08, 0.075, 10, 7),
    at(ellipsoid(0.04, 0.02, 0.06, 6, 4), 0.06, 0.06, 0.08),
    at(ellipsoid(0.04, 0.02, 0.06, 6, 4), 0.06, 0.06, -0.08),
  ]),
  legs: merge([
    at(ellipsoid(0.035, 0.17, 0.035, 6, 5), -0.2, 0.17, 0.11),
    at(ellipsoid(0.035, 0.17, 0.035, 6, 5), -0.2, 0.17, -0.11),
    at(ellipsoid(0.035, 0.17, 0.035, 6, 5), 0.2, 0.17, 0.11),
    at(ellipsoid(0.035, 0.17, 0.035, 6, 5), 0.2, 0.17, -0.11),
  ]),
};

// Airliner, built facing -X, ~3.2 long. Two engines under the wings leave
// the contrails.
const PLANE_ENGINE_Z = 0.78;
const PLANE_MESH = {
  body: merge([
    ellipsoid(1.6, 0.19, 0.19, 14, 8),
    xform(box(0.5, 0.045, 1.55), mul(T(0.4, -0.05, 0.72), Ry(0.42))),
    xform(box(0.5, 0.045, 1.55), mul(T(0.4, -0.05, -0.72), Ry(-0.42))),
    xform(box(0.32, 0.03, 0.5), mul(T(1.42, 0.06, 0.26), Ry(0.5))),
    xform(box(0.32, 0.03, 0.5), mul(T(1.42, 0.06, -0.26), Ry(-0.5))),
  ]),
  tail: xform(box(0.42, 0.55, 0.045), mul(T(1.38, 0.32, 0), Rz(-0.45))),
  engines: merge([
    at(ellipsoid(0.24, 0.085, 0.085, 10, 6), 0.12, -0.15, PLANE_ENGINE_Z),
    at(ellipsoid(0.24, 0.085, 0.085, 10, 6), 0.12, -0.15, -PLANE_ENGINE_Z),
  ]),
  puff: ellipsoid(1, 1, 1, 8, 6),
};
// Contrail: TRAIL_N puffs per engine, each TRAIL_SEG long, widening as the
// trail spreads and blending into the sky colour so it fades out after
// roughly TRAIL_N * TRAIL_SEG units.
const TRAIL_N = 16;
const TRAIL_SEG = 1.35;
const SKY_FADE = rgb(0xbde0fe);

const BIRD_MESH = {
  body: ellipsoid(0.15, 0.07, 0.07, 10, 8),
  wing: box(0.13, 0.016, 0.30),
  // Rotate first (cone +Y -> -X, the flight direction), then move it to the
  // front of the head; the base sits inside the body so it reads as attached.
  beak: xform(cone(0.022, 0.09, 5), mul(T(-0.17, 0.015, 0), Rz(Math.PI / 2))),
  tail: box(0.10, 0.014, 0.08),
};

// A leaf is a flattened lens, not a slab: a box silhouette reads as a plank
// lying on the grass once it is a few pixels wide.
const LEAF_MESH = ellipsoid(0.085, 0.014, 0.046, 8, 5);

/* ---------------------------------------------------------------- build */

export function buildActors() {
  const parts = [];

  // Adds a part and returns it, so the update loop can keep a reference.
  const add = (mesh, color, params, layer) => {
    const part = { mesh, color, model: M.identity(), params, layer };
    parts.push(part);
    return part;
  };

  /* ------------------------------------------------------------- people */
  // Walkers on both sides of the road: torso, hips, neck, head with a face and
  // swinging legs with feet. Cadence and phase differ so they never march.
  const people = [
    // |z| between ~3.3 and ~5.6 is the camera's orbit ring: nobody walks there
    { x0: 7.5, z: -2.5, shirt: 0, phase: 0.0, cadence: 7.6 },
    { x0: 1.5, z: -3.0, shirt: 3, phase: 3.6, cadence: 7.9 },
    { x0: -10.5, z: -6.0, shirt: 1, phase: 2.2, cadence: 6.7 },
    { x0: -16.0, z: -6.6, shirt: 2, phase: 1.1, cadence: 7.3 },
    { x0: 22.0, z: -7.2, shirt: 2, phase: 4.1, cadence: 8.2 },
    { x0: -24.0, z: -8.0, shirt: 3, phase: 5.4, cadence: 7.1 },
    // near side (z > 0): outside the camera's orbit (~4 units), so the
    // circling camera never passes through anyone
    { x0: 4.0, z: 6.0, shirt: 1, phase: 0.8, cadence: 7.4 },
    { x0: -8.0, z: 6.6, shirt: 0, phase: 2.9, cadence: 7.0 },
    { x0: 15.0, z: 7.2, shirt: 3, phase: 4.4, cadence: 7.8 },
    { x0: -20.0, z: 7.8, shirt: 2, phase: 1.6, cadence: 7.2 },
    { x0: 26.0, z: 8.4, shirt: 0, phase: 3.3, cadence: 7.5 },
  ].map((cfg, i) => {
    const layer = 2;
    const shirt = PAL.shirts[cfg.shirt];
    // deterministic variety: style and colour cycle at different rates
    const hairMesh = HAIR_STYLES[(i * 5 + 1) % HAIR_STYLES.length];
    const hairCol = HAIR_COLORS[(i * 7 + 2) % HAIR_COLORS.length];
    return {
      ...cfg,
      layer,
      torso: add(PERSON_MESH.torso, shirt, [0.08, 6, 0], layer),
      head: add(PERSON_MESH.head, PAL.skin, [0.12, 10, 0], layer),
      neck: add(PERSON_MESH.neck, PAL.skin, [0.12, 10, 0], layer),
      skinBits: add(PERSON_MESH.skinBits, PAL.skin, [0.12, 10, 0], layer),
      eyeWhites: add(PERSON_MESH.eyeWhites, PAL.eyeWhite, [0.30, 24, 0.10], layer),
      pupils: add(PERSON_MESH.pupils, PAL.pupil, [0.40, 32, 0], layer),
      brows: add(PERSON_MESH.brows, hairCol, [0.08, 6, 0], layer),
      mouth: add(PERSON_MESH.mouth, PAL.lips, [0.10, 8, 0], layer),
      hair: add(hairMesh, hairCol, [0.10, 10, 0.02], layer),
      armL: add(PERSON_MESH.arm, shirt, [0.08, 6, 0], layer),
      armR: add(PERSON_MESH.arm, shirt, [0.08, 6, 0], layer),
      hips: add(PERSON_MESH.hips, PAL.trousers, [0.08, 6, 0], layer),
      legL: add(PERSON_MESH.leg, PAL.trousers, [0.08, 6, 0], layer),
      legR: add(PERSON_MESH.leg, PAL.trousers, [0.08, 6, 0], layer),
      footL: add(PERSON_MESH.foot, PAL.shoe, [0.10, 8, 0], layer),
      footR: add(PERSON_MESH.foot, PAL.shoe, [0.10, 8, 0], layer),
    };
  });

  /* --------------------------------------------------------------- cats */
  // Trotting cats and sitting cats on both sides of the road, all facing -X.
  // None closer than |z| = 1.5: at z = 1.15 a cat passed between the camera
  // and the bike and filled a third of the frame.
  const makeWalkCat = (x0, z, phase, color, layer) => ({
    x0,
    z,
    layer,
    phase,
    body: add(CAT_MESH.body, color, [0.10, 8, 0], layer),
    head: add(CAT_MESH.head, color, [0.10, 8, 0], layer),
    earL: add(CAT_MESH.ear, color, [0.10, 8, 0], layer),
    earR: add(CAT_MESH.ear, color, [0.10, 8, 0], layer),
    eyes: add(CAT_MESH.eyes, PAL.catEye, [0.30, 24, 0.10], layer),
    legFL: add(CAT_MESH.leg, color, [0.10, 8, 0], layer),
    legFR: add(CAT_MESH.leg, color, [0.10, 8, 0], layer),
    legBL: add(CAT_MESH.leg, color, [0.10, 8, 0], layer),
    legBR: add(CAT_MESH.leg, color, [0.10, 8, 0], layer),
    tail: add(CAT_MESH.tail, color, [0.10, 8, 0], layer),
  });
  const makeSitCat = (x0, z, phase, color, layer) => ({
    x0,
    z,
    layer,
    phase,
    body: add(CAT_MESH.sitBody, color, [0.10, 8, 0], layer),
    head: add(CAT_MESH.head, color, [0.10, 8, 0], layer),
    earL: add(CAT_MESH.ear, color, [0.10, 8, 0], layer),
    earR: add(CAT_MESH.ear, color, [0.10, 8, 0], layer),
    eyes: add(CAT_MESH.eyes, PAL.catEye, [0.30, 24, 0.10], layer),
    foreL: add(CAT_MESH.foreleg, color, [0.10, 8, 0], layer),
    foreR: add(CAT_MESH.foreleg, color, [0.10, 8, 0], layer),
    tail1: add(CAT_MESH.tail, color, [0.10, 8, 0], layer),
    tail2: add(CAT_MESH.tailTip, color, [0.10, 8, 0], layer),
  });

  const walkCats = [
    makeWalkCat(3.5, -1.5, 0.7, PAL.catA, 1),
    makeWalkCat(-12.0, 6.0, 2.4, PAL.catB, 2),
    makeWalkCat(18.0, 6.5, 4.1, PAL.catA, 2),
  ];
  const sitCats = [
    makeSitCat(-6.5, -1.9, 1.9, PAL.catB, 2),
    makeSitCat(9.0, 6.2, 0.4, PAL.catA, 2),
  ];

  /* -------------------------------------------------------------- sheep */
  // A few sheep grazing on both sides: woolly body, dark face and legs, the
  // head dipping to the grass now and then. They drift with their layer.
  const sheep = [
    { x0: -3.0, z: -6.5, phase: 0.0, s: 1.0 },
    { x0: -1.6, z: -7.4, phase: 2.1, s: 0.9 },
    { x0: 13.0, z: -6.0, phase: 4.0, s: 1.05 },
    { x0: -14.0, z: 7.8, phase: 1.2, s: 1.0 },
    { x0: -12.6, z: 8.8, phase: 3.3, s: 0.85 },
    { x0: 6.0, z: 9.2, phase: 5.1, s: 1.0 },
  ].map((cfg) => {
    const layer = 2;
    return {
      ...cfg,
      layer,
      wool: add(SHEEP_MESH.wool, PAL.wool, [0.02, 4, 0], layer),
      head: add(SHEEP_MESH.head, PAL.sheepFace, [0.08, 8, 0], layer),
      legs: add(SHEEP_MESH.legs, PAL.sheepFace, [0.08, 8, 0], layer),
    };
  });

  /* -------------------------------------------------------------- birds */
  // Birds cross the high sky on layer 4, wings flapping out of phase.
  // High in the sky, and big: at 15-17 units out a life-size bird is four
  // pixels across and reads as a speck of dirt on the lens.
  const birds = [
    { x0: -24.0, y: 10.4, z: -16.0, flap: 9.4, phase: 0.0, yaw: 0.18, s: 4.2 },
    { x0: -18.0, y: 11.0, z: -14.5, flap: 8.1, phase: 1.7, yaw: -0.12, s: 3.6 },
    { x0: -11.0, y: 9.8, z: -15.5, flap: 10.8, phase: 3.1, yaw: 0.30, s: 4.6 },
    { x0: -5.0, y: 10.8, z: -16.5, flap: 7.3, phase: 4.6, yaw: -0.22, s: 3.9 },
    { x0: 1.0, y: 10.0, z: -14.0, flap: 8.8, phase: 5.9, yaw: 0.10, s: 4.4 },
  ].map((cfg) => {
    const layer = 4;
    return {
      ...cfg,
      layer,
      body: add(BIRD_MESH.body, PAL.bird, [0.14, 10, 0], layer),
      beak: add(BIRD_MESH.beak, PAL.beak, [0.14, 10, 0], layer),
      tail: add(BIRD_MESH.tail, PAL.bird, [0.14, 10, 0], layer),
      wingL: add(BIRD_MESH.wing, PAL.bird, [0.14, 10, 0], layer),
      wingR: add(BIRD_MESH.wing, PAL.bird, [0.14, 10, 0], layer),
    };
  });

  /* ------------------------------------------------------------ planes */
  // Airliners crossing the sky on straight lines at different headings and
  // heights, each pulling two contrails that fade behind it.
  const planes = [
    { heading: 0.35, d: 42, y: 15.5, speed: 10.0, ph: 0, s: 1.0, tail: 0xef476f },
    { heading: 2.4, d: 55, y: 19.0, speed: 8.5, ph: 120, s: 1.15, tail: 0x118ab2 },
    { heading: 4.3, d: 48, y: 17.0, speed: 9.2, ph: 210, s: 1.05, tail: 0xffd166 },
  ].map((cfg) => {
    const layer = 4;
    const trails = [];
    for (const side of [1, -1]) {
      for (let k = 0; k < TRAIL_N; k++) {
        const f = k / (TRAIL_N - 1);
        const fade = Math.pow(f, 1.2);
        const col = [
          1 + (SKY_FADE[0] - 1) * fade,
          1 + (SKY_FADE[1] - 1) * fade,
          1 + (SKY_FADE[2] - 1) * fade,
        ];
        trails.push({ side, k, part: add(PLANE_MESH.puff, col, [0.0, 2, 0.35 * (1 - fade)], layer) });
      }
    }
    return {
      ...cfg,
      body: add(PLANE_MESH.body, rgb(0xf4f6f8), [0.25, 24, 0], layer),
      tail: add(PLANE_MESH.tail, rgb(cfg.tail), [0.2, 16, 0], layer),
      engines: add(PLANE_MESH.engines, rgb(0x9aa3b0), [0.35, 24, 0], layer),
      trails,
    };
  });

  /* ------------------------------------------------------------- leaves */
  // Wind-blown leaves tumble near the camera (layer 1) and a little further
  // back (layer 2), each with its own spin and bob.
  const leaves = [];
  for (let i = 0; i < 22; i++) {
    const near = i % 3 !== 2;
    const layer = near ? 1 : 2;
    leaves.push({
      x0: -32 + i * (64 / 22) + (i % 2) * 0.9,
      y0: 0.55 + ((i * 0.73) % 1) * 1.45,
      z: near ? -0.35 - ((i * 0.37) % 1) * 1.25 : -2.0 - ((i * 0.51) % 1) * 1.9,
      layer,
      phase: i * 1.37,
      spin: 1.6 + (i % 5) * 0.42,
      bob: 1.1 + (i % 4) * 0.35,
      part: add(LEAF_MESH, PAL.leaves[i % 3], [0.10, 6, 0], layer),
    });
  }

  /* ------------------------------------------------------------- update */
  function update(t) {
    /* people: planted-foot walk cycle, arms counter-swing, hip dip */
    for (const p of people) {
      // Walk forward (-X, the way they face) over the scrolling ground.
      const f = p.cadence / 8.0; // gait cycles per second (~0.85-1.0)
      const v = 1.15 + (p.cadence - 7) * 0.25; // walking speed over the ground
      const x = wrapX(p.x0 - v * t, t, p.layer);
      const step = v / (2 * f); // distance the hip covers per half cycle
      const cyc = t * f + p.phase / TAU;
      // forward foot offset (toward -X) of a leg at gait phase u in [0, 1)
      const footOffset = (u) => {
        u -= Math.floor(u);
        if (u < 0.5) return step / 2 - step * (u / 0.5); // planted: slides back under the hip
        const w = (u - 0.5) / 0.5; // swing: eased forward
        return -step / 2 + step * (w * w * (3 - 2 * w));
      };
      const dL = footOffset(cyc);
      const dR = footOffset(cyc + 0.5);
      // leg angles (Rz: positive swings the foot to +X, i.e. backward)
      const thL = -Math.asin(Math.max(-0.95, Math.min(0.95, dL / LEG_LEN)));
      const thR = -Math.asin(Math.max(-0.95, Math.min(0.95, dR / LEG_LEN)));
      // hip height so the stance foot just touches the ground
      const uL = cyc - Math.floor(cyc);
      const stanceTh = uL < 0.5 ? thL : thR;
      const bob = LEG_LEN * (Math.cos(stanceTh) - 1);
      const root = mul(T(x, bob, p.z), Rz(0.05));

      p.torso.model = mul(root, T(0, 1.14, 0));
      const headM = mul(root, T(0, 1.60, 0));
      p.head.model = headM;
      p.neck.model = mul(root, T(0, 1.46, 0));
      p.skinBits.model = headM;
      p.eyeWhites.model = headM;
      p.pupils.model = headM;
      p.brows.model = headM;
      p.mouth.model = headM;
      p.hair.model = headM;

      // arms swing opposite to the leg on the same side
      p.armL.model = mul(root, T(0, 1.36, 0.225), Rz(-thL * 0.9), T(0, -0.26, 0));
      p.armR.model = mul(root, T(0, 1.36, -0.225), Rz(-thR * 0.9), T(0, -0.26, 0));

      const hipL = mul(root, T(0, 0.84, 0.085), Rz(thL));
      const hipR = mul(root, T(0, 0.84, -0.085), Rz(thR));
      p.hips.model = mul(root, T(0, 0.87, 0));
      p.legL.model = mul(hipL, T(0, -0.37, 0));
      p.legR.model = mul(hipR, T(0, -0.37, 0));
      p.footL.model = mul(hipL, T(-0.06, -0.805, 0));
      p.footR.model = mul(hipR, T(-0.06, -0.805, 0));
    }

    /* walking cats: diagonal gait, tail held up and swaying */
    for (const c of walkCats) {
      // trot forward at the speed the stride covers, on top of the scroll
      const x = wrapX(c.x0 - CAT_TROT_SPEED * t, t, c.layer);
      const ph = t * CAT_CADENCE + c.phase;
      const root = mul(T(x, Math.abs(Math.sin(ph * 2)) * 0.012, c.z), Rz(0.02));

      c.body.model = mul(root, T(0, 0.26, 0));
      const headBase = mul(root, T(-0.26, 0.34, 0), Rz(Math.sin(ph * 0.5) * 0.06));
      c.head.model = headBase;
      c.earL.model = mul(headBase, T(0, 0.085, 0.05));
      c.earR.model = mul(headBase, T(0, 0.085, -0.05));
      c.eyes.model = headBase;

      const a = Math.sin(ph) * 0.42;
      const hipFL = mul(root, T(-0.13, 0.22, 0.06), Rz(a));
      const hipFR = mul(root, T(-0.13, 0.22, -0.06), Rz(-a));
      const hipBL = mul(root, T(0.13, 0.22, 0.06), Rz(-a));
      const hipBR = mul(root, T(0.13, 0.22, -0.06), Rz(a));
      c.legFL.model = mul(hipFL, T(0, -0.11, 0));
      c.legFR.model = mul(hipFR, T(0, -0.11, 0));
      c.legBL.model = mul(hipBL, T(0, -0.11, 0));
      c.legBR.model = mul(hipBR, T(0, -0.11, 0));

      c.tail.model = mul(
        root,
        T(0.18, 0.29, 0),
        Rz(0.55 + Math.sin(t * 2.6 + c.phase) * 0.22),
        T(0.08, 0, 0)
      );
    }

    /* sitting cats: upright, head turning, tail tip flicking */
    for (const c of sitCats) {
      const x = wrapX(c.x0, t, c.layer);
      const root = mul(T(x, 0, c.z), Rz(-0.04));

      c.body.model = mul(root, T(0, 0.20, 0));
      const headBase = mul(root, T(-0.02, 0.40, 0), Ry(Math.sin(t * 0.9 + c.phase) * 0.22));
      c.head.model = headBase;
      c.earL.model = mul(headBase, T(0, 0.085, 0.05));
      c.earR.model = mul(headBase, T(0, 0.085, -0.05));
      c.eyes.model = headBase;

      c.foreL.model = mul(root, T(-0.07, 0.12, 0.05));
      c.foreR.model = mul(root, T(-0.07, 0.12, -0.05));

      // tail rooted inside the rump, lying along the grass; the tip flicks up
      const base = mul(root, T(0.06, 0.08, 0), Rz(0.15));
      c.tail1.model = mul(base, T(0.09, 0, 0));
      const flick = Math.sin(t * 4.2 + c.phase) * 0.35;
      c.tail2.model = mul(base, T(0.17, 0, 0), Rz(flick + 0.6), T(0.065, 0, 0));
    }

    /* sheep: standing, head dipping down to graze every few seconds */
    for (const sh of sheep) {
      const x = wrapX(sh.x0, t, sh.layer);
      const root = mul(T(x, 0, sh.z), S(sh.s, sh.s, sh.s));
      const graze = Math.max(0, Math.sin(t * 0.5 + sh.phase)) ** 2; // 0 up .. 1 down
      sh.wool.model = root;
      sh.legs.model = root;
      sh.head.model = mul(root, T(-0.3, 0.5, 0), Rz(graze * 1.0), T(-0.1, -0.02, 0));
    }

    /* birds: glide with a slow bob and a shallow bank, wings flapping */
    for (const b of birds) {
      // Own airspeed toward -X (the way the beak points), varied per bird, on
      // top of the slow sky scroll; the sky scroll alone barely moved them.
      const x = wrapX(b.x0 - BIRD_SPEED * (0.85 + 0.3 * Math.sin(b.phase * 2.3)) * t, t, b.layer);
      const y = b.y + Math.sin(t * 0.9 + b.phase) * 0.18;
      const bank = Math.sin(t * 0.7 + b.phase) * 0.12;
      const root = mul(T(x, y, b.z), Ry(b.yaw), Rz(bank), S(b.s, b.s, b.s));

      b.body.model = root;
      b.beak.model = root;
      b.tail.model = mul(root, T(0.19, 0.005, 0), Rz(Math.sin(t * 3.0 + b.phase) * 0.12));

      const flap = Math.sin(t * b.flap + b.phase) * 0.62;
      b.wingL.model = mul(root, T(0, 0.01, 0.05), Rx(flap), T(0, 0, 0.15));
      b.wingR.model = mul(root, T(0, 0.01, -0.05), Rx(-flap), T(0, 0, -0.15));
    }

    /* planes: straight flight on a long loop, contrails trailing behind */
    for (const pl of planes) {
      const dx = Math.cos(pl.heading);
      const dz = Math.sin(pl.heading);
      const L = 260; // loop length: the wrap happens far beyond the far plane
      const along = (((t * pl.speed + pl.ph) % L) + L) % L - L / 2;
      // closest pass at distance d from the pelican, perpendicular to heading
      const px = -dz * pl.d + dx * along;
      const pz = dx * pl.d + dz * along;
      const yaw = Math.atan2(dz, -dx); // local -X (the nose) -> heading
      const pm = mul(T(px, pl.y, pz), Ry(yaw), S(pl.s, pl.s, pl.s));
      pl.body.model = pm;
      pl.tail.model = pm;
      pl.engines.model = pm;
      for (const tr of pl.trails) {
        const f = tr.k / (TRAIL_N - 1);
        // widen as it spreads, then pinch off at the very end
        const w = (0.07 + 0.16 * f) * (1 - Math.max(0, (f - 0.75) / 0.25) * 0.85);
        const cx = 0.38 + (tr.k + 0.5) * TRAIL_SEG;
        tr.part.model = mul(pm, T(cx, -0.15, tr.side * PLANE_ENGINE_Z), S(TRAIL_SEG * 0.72, w, w));
      }
    }

    /* leaves: drift with the wind, bob and tumble end over end */
    for (const l of leaves) {
      const x = wrapX(l.x0, t, l.layer);
      const y = l.y0 + Math.sin(t * l.bob + l.phase) * 0.32;
      const sp = t * l.spin + l.phase;
      l.part.model = mul(
        T(x, y, l.z),
        Rz(sp * 0.8),
        Rx(sp * 1.35 + l.phase),
        Ry(sp * 0.55)
      );
    }
  }

  update(0);
  return { parts, update };
}
