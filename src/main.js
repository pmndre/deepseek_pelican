// main.js — entry point.
//
// Canvas sizing (devicePixelRatio capped at 2), an orbit camera that
// auto-rotates around the pelican and can be dragged / zoomed, a gentle mouse
// parallax on top, URL parameters (?t, ?still, ?pixels, ?px), the
// requestAnimationFrame loop, and WebGPU error reporting.
//
// The scene is three builders concatenated into one part list:
//   buildPelican() -> the rider and the bike (layer 0, never scrolls)
//   buildWorld()   -> ground, grass, flowers, trees, hills, mountains, clouds, sun
//   buildActors()  -> people, cats, sheep, birds, planes, leaves
// Every builder exposes update(t); main calls all three each frame with the
// same t, then hands the parts to the renderer.

import { Renderer } from './renderer.js';
import { buildPelican } from './pelican.js';
import { buildWorld, SUN_DIR, SUN_DISC_ELEV } from './world.js';
import { buildActors } from './actors.js';
import { mat4 } from './math.js';

const canvas = document.getElementById('gpu');
const hud = document.getElementById('hud');

// ---------------------------------------------------------------- camera ---
// Starting view: 3/4 from the front-right (the bike faces +X), eye just above
// the axles looking slightly up, so the bike and rider fill ~57% of the frame
// height and the horizon sits in the lower third. 45° lens, ~4.1 m away.
const EYE = [2.6, 0.46, 3.2];
const TARGET = [0.0, 1.0, 0.0];
const UP = [0.0, 1.0, 0.0];
const FOV_Y = (45 * Math.PI) / 180;
const NEAR = 0.05;
const FAR = 100.0;

// Orbit controls, in the spirit of three.js OrbitControls with autoRotate:
// the camera sits on a sphere around TARGET (azimuth, elevation, radius).
// It auto-rotates once per ORBIT_PERIOD; click-drag (or touch-drag) spins and
// tilts it with damped inertia, the wheel / pinch zooms, and auto-rotation
// carries on from wherever the drag left it.
const ORBIT_PERIOD = 60.0; // seconds per full auto-rotate turn
const ROTATE_SPEED = 1.0; // drag: 2*pi * rotateSpeed per window height
const DAMPING = 0.08; // fraction of the pending drag applied per 60 fps frame
const EL_MIN = (-10 * Math.PI) / 180; // eye stays above the grass
const EL_MAX = (60 * Math.PI) / 180;
const R_MIN = 2.2;
const R_MAX = 4.6; // near-side trees start at z = 5: don't back into them
const _off = [EYE[0] - TARGET[0], EYE[1] - TARGET[1], EYE[2] - TARGET[2]];
const R0 = Math.hypot(_off[0], _off[1], _off[2]);
const AZ0 = Math.atan2(_off[0], _off[2]);
const EL0 = Math.asin(_off[1] / R0);

// ------------------------------------------------------------- parallax ---
// The mouse moves the camera, not the world: the eye slides sideways and up/down
// around the fixed target, so real perspective gives the parallax and the
// rider stays planted on the ground. Offsets are in world units at full
// pointer deflection.
const CAM_SHIFT_X = 0.7;
const CAM_SHIFT_Y = 0.35;
const CAM_MIN_Y = 0.22; // never dip the eye into the grass
const PARALLAX_SMOOTH = 0.12; // per-frame lerp factor at 60 fps

// ------------------------------------------------------------ url params ---
const query = new URLSearchParams(location.search);
const STILL = query.get('still') === '1';
const tParam = query.has('t') ? Number(query.get('t')) : NaN;
const FIXED_TIME = Number.isFinite(tParam) ? tParam : (STILL ? 0.9 : null);
// ?pixels=false renders smooth at full resolution (no pixel-art pass).
const SMOOTH = query.get('pixels') === 'false' || query.get('pixels') === '0';
// ?px=<n> forces the art pixel size (canvas pixels per art pixel); 0 = auto.
const pxParam = query.has('px') ? Number(query.get('px')) : 0;
const PIXEL_SIZE = Number.isFinite(pxParam) && pxParam > 0 ? Math.floor(pxParam) : 0;

// The framing was designed for 16:9. On narrower windows the vertical field of
// view widens to keep the 16:9 horizontal coverage, capped at 70° so a tall
// window crops the sides a little instead of shrinking the rider to a speck.
// Wider windows simply see more to the sides.
const DESIGN_ASPECT = 16 / 9;
const FOV_Y_MAX = (70 * Math.PI) / 180;
function fovFor(aspect) {
  if (aspect >= DESIGN_ASPECT) return FOV_Y;
  return Math.min(FOV_Y_MAX, 2 * Math.atan(Math.tan(FOV_Y / 2) * (DESIGN_ASPECT / aspect)));
}

// ---------------------------------------------------------------- state ----
const DPR_CAP = 2;
let dpr = 1;
let lastW = 0;
let lastH = 0;
let renderer = null;
let sceneParts = null;
let builders = null;
let device = null;
let time = FIXED_TIME !== null ? FIXED_TIME : 0;
let lastNow = 0;
let scopeOpen = false;

// orbit state (see ORBIT_PERIOD). Starting azimuth follows ?t= so stills
// stay reproducible.
let orbitAz = AZ0 + (ORBIT_PERIOD > 0 ? (time / ORBIT_PERIOD) * Math.PI * 2 : 0);
let orbitEl = EL0;
let orbitR = R0;
let dragAz = 0; // drag input not yet applied (damped)
let dragEl = 0;
let zoomScale = 1; // pending zoom factor (damped)
let dragging = false;
let dragId = null;
let lastPX = 0;
let lastPY = 0;
const touches = new Map(); // pointerId -> {x, y} for pinch zoom
let pinchDist = 0;

// pointer state: raw target and smoothed value, both in [-1, 1]
let mxTarget = 0;
let myTarget = 0;
let mx = 0;
let my = 0;

// ---------------------------------------------------------------- sizing ---
// Sizes the backing store from the CSS box and the capped device pixel ratio.
// Returns false when the canvas has no area (nothing to draw this frame).
function sizeCanvas() {
  const cssW = canvas.clientWidth || 0;
  const cssH = canvas.clientHeight || 0;
  if (cssW <= 0 || cssH <= 0) return false;

  dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP);
  const w = Math.max(1, Math.floor(cssW * dpr));
  const h = Math.max(1, Math.floor(cssH * dpr));

  if (w === lastW && h === lastH) return true;

  canvas.width = w;
  canvas.height = h;
  lastW = w;
  lastH = h;
  if (renderer) renderer.resize();
  return true;
}

// ---------------------------------------------------------------- camera ---
// Advances the orbit by dt: auto-rotate (paused while dragging), then the
// damped share of any pending drag / zoom, then the limits.
function updateOrbit(dt) {
  const frames = dt * 60;
  if (!dragging && FIXED_TIME === null && ORBIT_PERIOD > 0) {
    orbitAz += (dt / ORBIT_PERIOD) * Math.PI * 2;
  }
  const k = 1 - Math.pow(1 - DAMPING, Math.max(frames, 0));
  orbitAz += dragAz * k;
  orbitEl += dragEl * k;
  dragAz *= 1 - k;
  dragEl *= 1 - k;
  const zk = Math.pow(zoomScale, k);
  orbitR *= zk;
  zoomScale /= zk;
  if (orbitEl < EL_MIN) orbitEl = EL_MIN;
  if (orbitEl > EL_MAX) orbitEl = EL_MAX;
  if (orbitR < R_MIN) orbitR = R_MIN;
  if (orbitR > R_MAX) orbitR = R_MAX;
}

function orbitEye() {
  const ce = Math.cos(orbitEl);
  return [
    TARGET[0] + orbitR * ce * Math.sin(orbitAz),
    TARGET[1] + orbitR * Math.sin(orbitEl),
    TARGET[2] + orbitR * ce * Math.cos(orbitAz),
  ];
}

// ------------------------------------------------------------ orbit input ---
function onPointerDown(event) {
  if (event.pointerType === 'mouse' && event.button !== 0) return;
  touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (touches.size === 2) {
    // second finger: switch from rotate to pinch zoom
    const [p, q] = [...touches.values()];
    pinchDist = Math.hypot(p.x - q.x, p.y - q.y);
    dragging = true;
    dragId = null;
    return;
  }
  dragging = true;
  dragId = event.pointerId;
  lastPX = event.clientX;
  lastPY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
  canvas.style.cursor = 'grabbing';
}

function onDragMove(event) {
  if (touches.has(event.pointerId)) touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (touches.size === 2) {
    const [p, q] = [...touches.values()];
    const d = Math.hypot(p.x - q.x, p.y - q.y);
    if (pinchDist > 0 && d > 0) zoomScale *= pinchDist / d;
    pinchDist = d;
    return;
  }
  if (!dragging || event.pointerId !== dragId) return;
  const h = canvas.clientHeight || window.innerHeight || 1;
  const dx = event.clientX - lastPX;
  const dy = event.clientY - lastPY;
  lastPX = event.clientX;
  lastPY = event.clientY;
  // OrbitControls convention: drag right spins the view right, drag down
  // raises the camera.
  dragAz -= ((2 * Math.PI * dx) / h) * ROTATE_SPEED;
  dragEl += ((2 * Math.PI * dy) / h) * ROTATE_SPEED;
}

function onPointerUp(event) {
  touches.delete(event.pointerId);
  if (touches.size === 1) {
    // back to one finger: continue rotating with it
    const [[id, p]] = [...touches.entries()];
    dragId = id;
    lastPX = p.x;
    lastPY = p.y;
    return;
  }
  if (touches.size > 0) return;
  dragging = false;
  dragId = null;
  canvas.style.cursor = 'grab';
}

function onWheel(event) {
  event.preventDefault();
  zoomScale *= Math.pow(0.95, -event.deltaY / 100);
}

// -------------------------------------------------------------- parallax ---
function onPointerMove(event) {
  if (FIXED_TIME !== null) return; // frozen pose: no mouse offset
  if (dragging) return; // the drag owns the camera; parallax holds still
  const w = window.innerWidth || 1;
  const h = window.innerHeight || 1;
  mxTarget = (event.clientX / w) * 2 - 1;
  myTarget = (event.clientY / h) * 2 - 1;
  if (mxTarget < -1) mxTarget = -1;
  else if (mxTarget > 1) mxTarget = 1;
  if (myTarget < -1) myTarget = -1;
  else if (myTarget > 1) myTarget = 1;
}

// Smooths the pointer toward its target, frame-rate independent.
function smoothPointer(dt) {
  if (FIXED_TIME !== null) {
    mx = 0;
    my = 0;
    return;
  }
  const k = 1 - Math.pow(1 - PARALLAX_SMOOTH, dt * 60);
  mx += (mxTarget - mx) * k;
  my += (myTarget - my) * k;
}

// Shifts the orbiting eye by the pointer along the camera's right and up axes.
function parallaxEye(eye) {
  const fx = TARGET[0] - eye[0];
  const fz = TARGET[2] - eye[2];
  const fl = Math.hypot(fx, fz) || 1;
  const rx = -fz / fl; // right = forward x up (horizontal)
  const rz = fx / fl;
  const out = [
    eye[0] + rx * mx * CAM_SHIFT_X,
    eye[1] - my * CAM_SHIFT_Y,
    eye[2] + rz * mx * CAM_SHIFT_X,
  ];
  if (out[1] < CAM_MIN_Y) out[1] = CAM_MIN_Y;
  return out;
}

// ------------------------------------------------------- near-camera fade --
// Everything scrolls along X while the camera circles the pelican, so now and
// then a weed, bush or walker passes right through the lens. Small objects
// (bounding radius under NEAR_MAX_R) that come within NEAR_FADE of the eye
// shrink smoothly into the ground and grow back once past. Builders rewrite
// every model matrix each frame, so scaling them in place never accumulates.
const NEAR_FADE = 1.6;
const NEAR_GONE = 0.6;
const NEAR_MAX_R = 2.5;

function meshRadius(mesh) {
  if (mesh._radius !== undefined) return mesh._radius;
  let r = 0;
  const P = mesh.positions;
  for (let i = 0; i < P.length; i += 3) r = Math.max(r, Math.hypot(P[i], P[i + 1], P[i + 2]));
  mesh._radius = r;
  return r;
}

function fadeNearCamera(parts, eye) {
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (!part.layer || !part.mesh || !part.model) continue; // the rider (layer 0) never fades
    const m = part.model;
    const sc = Math.hypot(m[0], m[1], m[2]);
    if (meshRadius(part.mesh) * sc > NEAR_MAX_R) continue; // ground sheets, hills, mountains
    const d = Math.hypot(m[12] - eye[0], m[14] - eye[2]);
    if (d >= NEAR_FADE) continue;
    let k = (d - NEAR_GONE) / (NEAR_FADE - NEAR_GONE);
    k = k <= 0 ? 0.0001 : k * k * (3 - 2 * k);
    for (const c of [0, 1, 2, 4, 5, 6, 8, 9, 10]) m[c] *= k;
  }
}

// -------------------------------------------------------------- reporting --
// The label only appears for errors (WebGPU missing or failing to start).
function setHud(text) {
  if (!hud) return;
  hud.textContent = text;
  hud.style.display = 'block';
}

function reportAdapterInfo(info) {
  if (!info) {
    console.log('WebGPU adapter: (no info)');
    return;
  }
  const vendor = info.vendor || 'unknown';
  const architecture = info.architecture || '';
  const deviceName = info.device || '';
  const description = info.description || '';
  console.log(
    'WebGPU adapter:',
    `vendor=${vendor}`,
    `architecture=${architecture || '-'}`,
    `device=${deviceName || '-'}`,
    description ? `description=${description}` : ''
  );
}

function watchDevice(dev) {
  if (!dev) return;
  if (typeof dev.addEventListener === 'function') {
    dev.addEventListener('uncapturederror', (event) => {
      const err = event && event.error;
      console.error('WebGPU uncaptured error:', (err && err.message) || String(err));
    });
  }
  if (dev.lost && typeof dev.lost.then === 'function') {
    dev.lost.then((info) => {
      console.error('WebGPU device lost:', info.reason, info.message || '');
    });
  }
}

// ------------------------------------------------------------------ loop ---
function frame(now) {
  requestAnimationFrame(frame);

  if (!renderer || !sceneParts) return;

  const dt = lastNow ? Math.min(Math.max((now - lastNow) / 1000, 0), 0.05) : 0;
  lastNow = now;
  if (FIXED_TIME === null) time += dt;

  if (!sizeCanvas()) return; // canvas collapsed to 0: skip the frame

  // The art target has the canvas's aspect (up to one cropped edge pixel).
  const aspect = renderer.artAspect || canvas.width / Math.max(1, canvas.height);
  smoothPointer(dt);
  updateOrbit(dt);
  const eye = parallaxEye(orbitEye());
  const proj = mat4.perspective(fovFor(aspect), aspect, NEAR, FAR);
  const view = mat4.lookAt(eye, TARGET, UP);
  const viewProj = mat4.multiply(proj, view);

  builders.pelican.update(time);
  builders.world.update(time);
  builders.actors.update(time);
  fadeNearCamera(sceneParts, eye);

  const scoped = !scopeOpen && device && typeof device.pushErrorScope === 'function';
  if (scoped) {
    device.pushErrorScope('validation');
    scopeOpen = true;
  }

  try {
    renderer.draw(viewProj, eye);
  } catch (err) {
    console.error('WebGPU draw failed:', err);
  }

  if (scoped) {
    scopeOpen = false;
    device
      .popErrorScope()
      .then((err) => {
        if (err) console.error('WebGPU validation error:', err.message);
      })
      .catch((err) => {
        console.error('WebGPU error scope failed:', err);
      });
  }
}

// ------------------------------------------------------------------ boot ---
async function main() {
  if (!navigator.gpu) {
    setHud('WebGPU unavailable');
    console.error('WebGPU unavailable: navigator.gpu is missing.');
    return;
  }

  let adapterInfo = null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) {
      adapterInfo = adapter.info || null;
      reportAdapterInfo(adapterInfo);
    } else {
      console.warn('WebGPU: requestAdapter() returned null.');
    }
  } catch (err) {
    console.warn('WebGPU: could not read adapter info:', err);
  }

  // Key light = the original light direction (shared with the sun disc in
  // world.js); shadow map extent/distance stay at the renderer defaults.
  renderer = new Renderer({
    pixelSize: PIXEL_SIZE,
    smooth: SMOOTH,
    lightDir: SUN_DIR,
    sunDiscElev: SUN_DISC_ELEV, // sky halo sits around the visible disc
  });
  try {
    await renderer.init(canvas);
  } catch (err) {
    setHud('WebGPU init failed');
    console.error('WebGPU init failed:', err);
    return;
  }

  device = renderer.device || null;
  if (!device) {
    console.warn('WebGPU: renderer did not expose a device; validation scopes are off.');
  }
  watchDevice(device);

  // ---- scene: pelican + world + actors, one flat part list
  builders = {
    pelican: buildPelican(),
    world: buildWorld(),
    actors: buildActors(),
  };
  sceneParts = [].concat(builders.pelican.parts, builders.world.parts, builders.actors.parts);
  builders.pelican.update(time);
  builders.world.update(time);
  builders.actors.update(time);
  renderer.setScene(sceneParts);


  window.addEventListener('pointermove', onPointerMove);
  canvas.style.cursor = 'grab';
  canvas.style.touchAction = 'none'; // touch drags orbit instead of scrolling
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onDragMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });

  // Real fullscreen: double-click or press F.
  const toggleFullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
  };
  window.addEventListener('dblclick', toggleFullscreen);
  window.addEventListener('keydown', (e) => {
    if ((e.key === 'f' || e.key === 'F') && !e.metaKey && !e.ctrlKey && !e.altKey) toggleFullscreen();
  });
  window.addEventListener('resize', () => {
    sizeCanvas(); // re-creates the render targets when the size really changed
  });

  sizeCanvas();
  requestAnimationFrame(frame);
}

main();
