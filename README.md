# Pelican on a Bicycle

A pelican rides a bicycle through an endless low-poly countryside, rendered in real time with **WebGPU** and an optional pixel-art look.

Everything is procedural: no models, no textures, no libraries besides the dev server. The bike, the rider, the trees, the mountains and the people walking by are all built from code.

## How it was made

This project was made by an agent swarm running **DeepSeek 4.1 Flash** in [pine](https://github.com/pmndre/pine), started with `/swarm`: a lead agent took the job and broke it into tasks on a board, sub-agents worked through them, and the result was checked before it was done.

## What's in the scene

- **The rider:** a pelican on a bicycle, pedalling with webbed feet, steering a little and leaning into the turns.
- **The world:** a scrolling meadow with scattered grass and flowers (daisies, cosmos, tulips, lavender), low-poly trees of several species, bushes, hills and a ring of snow-capped mountains.
- **Life:** people walking with a planted-foot gait and varied hairstyles, trotting and sitting cats, grazing sheep, birds, and airliners pulling fading contrails.
- **Wind:** grass, flowers, bushes and trees sway and shiver in gusts.
- **Light:** a key light with shadow mapping, hemispheric ambient, rim light, and a sun with a halo in a banded sky.
- **Pixel-art pass:** the scene renders at a low resolution matching the window's shape, is posterised with an ordered dither, and gets a one-pixel outline. Turn it off with `?pixels=false` to see the smooth shading.

## Running it

You need a browser with WebGPU: a current Chrome, Edge or Safari.

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

To build a static version and preview it:

```bash
npm run build
npm run preview
```

The build in `dist/` uses relative paths, so it can be hosted from any folder, GitHub Pages included.

## Controls

| Input | Action |
| --- | --- |
| Drag (mouse or one finger) | Orbit the camera around the rider |
| Scroll wheel / pinch | Zoom in and out |
| Move the mouse | Subtle parallax |
| Double-click or `F` | Toggle fullscreen |

The camera rotates slowly on its own and picks up again from wherever you leave it.

## URL parameters

| Parameter | Effect |
| --- | --- |
| `?pixels=false` | Full-resolution smooth rendering, without the pixel-art pass |
| `?px=<n>` | Force the pixel size (screen pixels per art pixel) |
| `?t=<seconds>` | Freeze the scene at a given moment |
| `?still=1` | Freeze the scene at a default moment |

## Project layout

| File | What it does |
| --- | --- |
| `src/main.js` | Entry point: canvas sizing, orbit camera, input, frame loop |
| `src/renderer.js` | WebGPU renderer: shadow pass, low-res art pass, upscale and outline pass |
| `src/shaders.js` | WGSL shaders: sky, shadows, lighting, posterise and dither, outline |
| `src/pelican.js` | The pelican and the bicycle, with pedalling, steering and lean |
| `src/world.js` | Ground, grass, flowers, trees, hills, mountains, clouds and the sun |
| `src/actors.js` | People, cats, sheep, birds, airplanes and leaves |
| `src/geometry.js` | Mesh primitives: box, sphere, cylinder, cone, torus, merge |
| `src/math.js` | Matrix and vector helpers |
