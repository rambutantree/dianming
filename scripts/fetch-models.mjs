// Copies the face-api.js model weights bundled inside the npm package into
// public/models so Vite can serve them as static assets at runtime.
import { cp, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));
const src = path.join(root, "..", "node_modules", "@vladmandic", "face-api", "model");
const dest = path.join(root, "..", "public", "models");

if (!existsSync(src)) {
  console.warn(`[fetch-models] source not found at ${src}, skipping (did npm install run?)`);
  process.exit(0);
}

await mkdir(dest, { recursive: true });
await cp(src, dest, { recursive: true });
console.log(`[fetch-models] copied model weights to ${path.relative(process.cwd(), dest)}`);
