// math.js — column-major 4x4 matrices (Float32Array(16)) and vec3 helpers.
// WebGPU conventions: clip z in [0,1], right-handed lookAt (camera looks down -Z).
// No imports: this module is self-contained.

const EPS = 1e-12;

/* ------------------------------------------------------------------ mat4 */

export const mat4 = {
  identity() {
    const m = new Float32Array(16);
    m[0] = 1;
    m[5] = 1;
    m[10] = 1;
    m[15] = 1;
    return m;
  },

  /** returns a * b (column-major, so out = a·b applies b first). */
  multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4 + 0];
      const b1 = b[c * 4 + 1];
      const b2 = b[c * 4 + 2];
      const b3 = b[c * 4 + 3];
      out[c * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return out;
  },

  /** WebGPU projection: z maps near -> 0, far -> 1. */
  perspective(fovY, aspect, near, far) {
    const f = 1 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    const m = new Float32Array(16);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = far * nf;
    m[11] = -1;
    m[14] = far * near * nf;
    m[15] = 0;
    return m;
  },

  /** right-handed view matrix; camera at eye looking toward target. */
  lookAt(eye, target, up) {
    let zx = eye[0] - target[0];
    let zy = eye[1] - target[1];
    let zz = eye[2] - target[2];
    let l = Math.hypot(zx, zy, zz);
    if (l < EPS) {
      zx = 0;
      zy = 0;
      zz = 1;
      l = 1;
    }
    zx /= l;
    zy /= l;
    zz /= l;

    // x = normalize(cross(up, z))
    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    l = Math.hypot(xx, xy, xz);
    if (l < EPS) {
      // up parallel to view direction: pick any orthogonal axis
      xx = 1;
      xy = 0;
      xz = 0;
      l = 1;
    }
    xx /= l;
    xy /= l;
    xz /= l;

    // y = cross(z, x)
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    const m = new Float32Array(16);
    m[0] = xx;
    m[1] = yx;
    m[2] = zx;
    m[3] = 0;
    m[4] = xy;
    m[5] = yy;
    m[6] = zy;
    m[7] = 0;
    m[8] = xz;
    m[9] = yz;
    m[10] = zz;
    m[11] = 0;
    m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    m[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    m[15] = 1;
    return m;
  },

  translation(x, y, z) {
    const m = new Float32Array(16);
    m[0] = 1;
    m[5] = 1;
    m[10] = 1;
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m[15] = 1;
    return m;
  },

  rotationX(rad) {
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const m = new Float32Array(16);
    m[0] = 1;
    m[5] = c;
    m[6] = s;
    m[9] = -s;
    m[10] = c;
    m[15] = 1;
    return m;
  },

  rotationY(rad) {
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const m = new Float32Array(16);
    m[0] = c;
    m[2] = -s;
    m[5] = 1;
    m[8] = s;
    m[10] = c;
    m[15] = 1;
    return m;
  },

  rotationZ(rad) {
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const m = new Float32Array(16);
    m[0] = c;
    m[1] = s;
    m[4] = -s;
    m[5] = c;
    m[10] = 1;
    m[15] = 1;
    return m;
  },

  scaling(x, y, z) {
    const m = new Float32Array(16);
    m[0] = x;
    m[5] = y;
    m[10] = z;
    m[15] = 1;
    return m;
  },

  /** scale, then rotX, then rotY, then rotZ, then translate. */
  compose(pos, rotEuler, scale) {
    const p = pos || [0, 0, 0];
    const r = rotEuler || [0, 0, 0];
    const s = scale || [1, 1, 1];
    const sx = Math.sin(r[0]);
    const cx = Math.cos(r[0]);
    const sy = Math.sin(r[1]);
    const cy = Math.cos(r[1]);
    const sz = Math.sin(r[2]);
    const cz = Math.cos(r[2]);

    // Rz * Ry * Rx * S, expanded (column-major)
    const m = new Float32Array(16);
    m[0] = cz * cy * s[0];
    m[1] = sz * cy * s[0];
    m[2] = -sy * s[0];
    m[3] = 0;

    m[4] = (cz * sy * sx - sz * cx) * s[1];
    m[5] = (sz * sy * sx + cz * cx) * s[1];
    m[6] = cy * sx * s[1];
    m[7] = 0;

    m[8] = (cz * sy * cx + sz * sx) * s[2];
    m[9] = (sz * sy * cx - cz * sx) * s[2];
    m[10] = cy * cx * s[2];
    m[11] = 0;

    m[12] = p[0];
    m[13] = p[1];
    m[14] = p[2];
    m[15] = 1;
    return m;
  },

  /** affine transform of a point (perspective divide when w != 1). */
  transformPoint(m, p) {
    const x = p[0];
    const y = p[1];
    const z = p[2];
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    const ox = m[0] * x + m[4] * y + m[8] * z + m[12];
    const oy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const oz = m[2] * x + m[6] * y + m[10] * z + m[14];
    if (Math.abs(w) > EPS && Math.abs(w - 1) > EPS) return [ox / w, oy / w, oz / w];
    return [ox, oy, oz];
  },

  invert(m) {
    const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
    const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
    const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
    const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;

    const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (Math.abs(det) < EPS) return mat4.identity();
    const d = 1 / det;

    const out = new Float32Array(16);
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d;
    return out;
  },
};

/* ------------------------------------------------------------------ vec3 */

export const vec3 = {
  add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  },
  sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  },
  scale(a, s) {
    return [a[0] * s, a[1] * s, a[2] * s];
  },
  normalize(a) {
    const l = Math.hypot(a[0], a[1], a[2]);
    if (l < EPS) return [0, 0, 0];
    return [a[0] / l, a[1] / l, a[2] / l];
  },
  cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  },
  dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  },
  lerp(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  },
};
