import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths, so the production build also works from a
  // sub-path such as GitHub Pages.
  base: './',
});
