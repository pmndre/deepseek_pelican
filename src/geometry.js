// geometry.js — procedural mesh builders.
// Every builder returns a NEW object: {positions: Float32Array (xyz),
// normals: Float32Array (xyz), indices: Uint32Array}.
// Vertex stride is 24 bytes: position at offset 0, normal at offset 12.
// Front faces are CCW (outward normals). No imports: self-contained.

const TAU = Math.PI * 2;

function mesh(positions, normals, indices) {
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
  };
}

/* ------------------------------------------------------------------- box */
// Axis-aligned box centred on the origin. 24 verts / 36 indices, flat normals.
export function box(w, h, d) {
  const hx = w / 2;
  const hy = h / 2;
  const hz = d / 2;

  // Each face: normal + 4 corners in CCW order seen from outside.
  const faces = [
    // +Z
    { n: [0, 0, 1], v: [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]] },
    // -Z
    { n: [0, 0, -1], v: [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]] },
    // +X
    { n: [1, 0, 0], v: [[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]] },
    // -X
    { n: [-1, 0, 0], v: [[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]] },
    // +Y
    { n: [0, 1, 0], v: [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]] },
    // -Y
    { n: [0, -1, 0], v: [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]] },
  ];

  const positions = [];
  const normals = [];
  const indices = [];
  for (const f of faces) {
    const base = positions.length / 3;
    for (const p of f.v) {
      positions.push(p[0], p[1], p[2]);
      normals.push(f.n[0], f.n[1], f.n[2]);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return mesh(positions, normals, indices);
}

/* ---------------------------------------------------------------- sphere */
// UV sphere centred on the origin; smooth normals equal normalized position.
export function sphere(r, seg = 24, rings = 16) {
  const S = Math.max(3, Math.floor(seg));
  const R = Math.max(2, Math.floor(rings));
  const positions = [];
  const normals = [];
  const indices = [];

  for (let i = 0; i <= R; i++) {
    const phi = (i / R) * Math.PI;
    const sp = Math.sin(phi);
    const cp = Math.cos(phi);
    for (let j = 0; j <= S; j++) {
      const theta = (j / S) * TAU;
      const x = sp * Math.cos(theta);
      const y = cp;
      const z = sp * Math.sin(theta);
      positions.push(x * r, y * r, z * r);
      normals.push(x, y, z);
    }
  }

  const row = S + 1;
  for (let i = 0; i < R; i++) {
    for (let j = 0; j < S; j++) {
      const a = i * row + j;
      const b = a + 1;
      const c = a + row + 1;
      const d = a + row;
      // CCW outward
      indices.push(a, b, d, b, c, d);
    }
  }
  return mesh(positions, normals, indices);
}

/* -------------------------------------------------------------- cylinder */
// Frustum along Y, centred on the origin. Smooth side normals, flat caps.
export function cylinder(rTop, rBot, h, seg = 24, capped = true) {
  const S = Math.max(3, Math.floor(seg));
  const hy = h / 2;
  const positions = [];
  const normals = [];
  const indices = [];

  // side: bottom ring then top ring, both with smooth normals
  const slopeY = rBot - rTop; // normal y component (see derivation below)
  for (let j = 0; j <= S; j++) {
    const theta = (j / S) * TAU;
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    // outward normal of the conical side: normalize(ct*h, rBot-rTop, st*h)
    let nx = ct * h;
    let ny = slopeY;
    let nz = st * h;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l;
    ny /= l;
    nz /= l;
    positions.push(ct * rBot, -hy, st * rBot);
    normals.push(nx, ny, nz);
  }
  for (let j = 0; j <= S; j++) {
    const theta = (j / S) * TAU;
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    let nx = ct * h;
    let ny = slopeY;
    let nz = st * h;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l;
    ny /= l;
    nz /= l;
    positions.push(ct * rTop, hy, st * rTop);
    normals.push(nx, ny, nz);
  }

  const topOff = S + 1;
  for (let j = 0; j < S; j++) {
    const a = j; // bottom j
    const b = j + 1; // bottom j+1
    const c = topOff + j + 1; // top j+1
    const d = topOff + j; // top j
    // CCW outward
    indices.push(a, d, c, a, c, b);
  }

  if (capped) {
    // top cap, normal +Y
    if (rTop > 1e-9) {
      const centre = positions.length / 3;
      positions.push(0, hy, 0);
      normals.push(0, 1, 0);
      const ringStart = positions.length / 3;
      for (let j = 0; j <= S; j++) {
        const theta = (j / S) * TAU;
        positions.push(Math.cos(theta) * rTop, hy, Math.sin(theta) * rTop);
        normals.push(0, 1, 0);
      }
      for (let j = 0; j < S; j++) {
        indices.push(centre, ringStart + j + 1, ringStart + j);
      }
    }
    // bottom cap, normal -Y
    if (rBot > 1e-9) {
      const centre = positions.length / 3;
      positions.push(0, -hy, 0);
      normals.push(0, -1, 0);
      const ringStart = positions.length / 3;
      for (let j = 0; j <= S; j++) {
        const theta = (j / S) * TAU;
        positions.push(Math.cos(theta) * rBot, -hy, Math.sin(theta) * rBot);
        normals.push(0, -1, 0);
      }
      for (let j = 0; j < S; j++) {
        indices.push(centre, ringStart + j, ringStart + j + 1);
      }
    }
  }
  return mesh(positions, normals, indices);
}

/* ------------------------------------------------------------------ cone */
// Apex at +Y, base at -Y, centred on the origin. Smooth sides, flat base.
export function cone(r, h, seg = 24) {
  const S = Math.max(3, Math.floor(seg));
  const hy = h / 2;
  const positions = [];
  const normals = [];
  const indices = [];

  // base ring
  for (let j = 0; j < S; j++) {
    const theta = (j / S) * TAU;
    positions.push(Math.cos(theta) * r, -hy, Math.sin(theta) * r);
    let nx = Math.cos(theta) * h;
    let ny = r;
    let nz = Math.sin(theta) * h;
    const l = Math.hypot(nx, ny, nz) || 1;
    normals.push(nx / l, ny / l, nz / l);
  }
  // apex, duplicated per segment so each side gets its own normal
  for (let j = 0; j < S; j++) {
    const theta = (j / S) * TAU;
    positions.push(0, hy, 0);
    let nx = Math.cos(theta) * h;
    let ny = r;
    let nz = Math.sin(theta) * h;
    const l = Math.hypot(nx, ny, nz) || 1;
    normals.push(nx / l, ny / l, nz / l);
  }
  for (let j = 0; j < S; j++) {
    const b0 = j;
    const b1 = (j + 1) % S;
    const ap = S + j;
    // CCW outward
    indices.push(b1, b0, ap);
  }

  // base cap, normal -Y
  const centre = positions.length / 3;
  positions.push(0, -hy, 0);
  normals.push(0, -1, 0);
  const ringStart = positions.length / 3;
  for (let j = 0; j <= S; j++) {
    const theta = (j / S) * TAU;
    positions.push(Math.cos(theta) * r, -hy, Math.sin(theta) * r);
    normals.push(0, -1, 0);
  }
  for (let j = 0; j < S; j++) {
    indices.push(centre, ringStart + j, ringStart + j + 1);
  }
  return mesh(positions, normals, indices);
}

/* ----------------------------------------------------------------- torus */
// Torus in the XZ plane (axis = Y), centred on the origin. Smooth normals.
export function torus(R, r, seg = 48, sides = 16) {
  const S = Math.max(3, Math.floor(seg));
  const T = Math.max(3, Math.floor(sides));
  const positions = [];
  const normals = [];
  const indices = [];

  for (let i = 0; i <= S; i++) {
    const u = (i / S) * TAU;
    const cu = Math.cos(u);
    const su = Math.sin(u);
    for (let j = 0; j <= T; j++) {
      const v = (j / T) * TAU;
      const cv = Math.cos(v);
      const sv = Math.sin(v);
      const nx = cv * cu;
      const ny = sv;
      const nz = cv * su;
      positions.push((R + r * cv) * cu, r * sv, (R + r * cv) * su);
      normals.push(nx, ny, nz);
    }
  }

  const row = T + 1;
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < T; j++) {
      const a = i * row + j;
      const b = a + 1;
      const c = a + row + 1;
      const d = a + row;
      // CCW outward
      indices.push(a, b, d, b, c, d);
    }
  }
  return mesh(positions, normals, indices);
}

/* ----------------------------------------------------------------- merge */
// Concatenate meshes, offsetting each mesh's indices by its vertex base.
export function merge(meshes) {
  let vertCount = 0;
  let idxCount = 0;
  for (const m of meshes) {
    vertCount += m.positions.length / 3;
    idxCount += m.indices.length;
  }

  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const indices = new Uint32Array(idxCount);

  let vBase = 0;
  let iOff = 0;
  for (const m of meshes) {
    const vCount = m.positions.length / 3;
    positions.set(m.positions, vBase * 3);
    normals.set(m.normals, vBase * 3);
    const idx = m.indices;
    for (let k = 0; k < idx.length; k++) indices[iOff + k] = idx[k] + vBase;
    vBase += vCount;
    iOff += idx.length;
  }
  return { positions, normals, indices };
}
