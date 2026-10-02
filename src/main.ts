import * as faceapi from "@vladmandic/face-api";
import "./style.css";
import type { Person, UnknownCluster } from "./types";
import { clearRoster, deletePerson, loadRoster, savePerson } from "./faceStore";
import { findBestMatch, foldDescriptor, LEARN_THRESHOLD, newPersonId } from "./recognizer";

const MODEL_URL = "/models";
const DETECT_INTERVAL_MS = 350;
const STALE_UNKNOWN_MS = 60_000;

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("overlay") as HTMLCanvasElement;
const videoWrap = document.querySelector(".video-wrap") as HTMLElement;
const statusEl = document.getElementById("status") as HTMLElement;
const cameraToggle = document.getElementById("camera-toggle") as HTMLButtonElement;
const clearDataBtn = document.getElementById("clear-data") as HTMLButtonElement;
const rosterUpload = document.getElementById("roster-upload") as HTMLInputElement;
const unknownListEl = document.getElementById("unknown-list") as HTMLElement;
const rosterListEl = document.getElementById("roster-list") as HTMLElement;
const liveCountEl = document.getElementById("count-live") as HTMLElement;
const knownCountEl = document.getElementById("count-known") as HTMLElement;
const sessionCountEl = document.getElementById("count-session") as HTMLElement;

let roster: Person[] = [];
let unknownClusters: UnknownCluster[] = [];
const sessionSeenIds = new Set<string>();

let stream: MediaStream | null = null;
let detectTimer: ReturnType<typeof setInterval> | null = null;
let busy = false;
let modelsReady = false;

function setStatus(text: string): void {
  statusEl.textContent = text;
}

function detectorOptions(): faceapi.TinyFaceDetectorOptions {
  return new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
}

async function loadModels(): Promise<void> {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);
  modelsReady = true;
}

async function cropThumbnail(
  input: HTMLVideoElement | HTMLImageElement,
  detection: faceapi.FaceDetection,
): Promise<string> {
  const [face] = await faceapi.extractFaces(input, [detection]);
  return face instanceof HTMLCanvasElement ? face.toDataURL("image/jpeg", 0.8) : "";
}

function updateCounts(liveCount: number): void {
  liveCountEl.textContent = String(liveCount);
  knownCountEl.textContent = String(roster.length);
  sessionCountEl.textContent = String(sessionSeenIds.size);
}

/**
 * Core identity logic shared by the live camera loop and the roster-photo
 * upload flow: match against known people first, then against faces already
 * seen-but-unnamed this session, and only create a new pending entry when
 * neither matches. This keeps a single learning pipeline regardless of
 * where the face came from.
 */
async function resolveSighting(
  descriptor: Float32Array,
  thumbnail: () => Promise<string>,
): Promise<{ label: string; kind: "known" | "unknown" }> {
  const knownMatch = findBestMatch(descriptor, roster);
  if (knownMatch) {
    const { item: person, distance } = knownMatch;
    person.updatedAt = Date.now();
    if (distance < LEARN_THRESHOLD) {
      person.descriptor = foldDescriptor(person, descriptor);
    }
    person.sampleCount += 1;
    sessionSeenIds.add(person.id);
    void savePerson(person);
    return { label: person.name, kind: "known" };
  }

  const unknownMatch = findBestMatch(descriptor, unknownClusters);
  if (unknownMatch) {
    const cluster = unknownMatch.item;
    cluster.descriptor = foldDescriptor(cluster, descriptor);
    cluster.sampleCount += 1;
    cluster.lastSeen = Date.now();
    sessionSeenIds.add(cluster.id);
    return { label: "Unknown", kind: "unknown" };
  }

  const cluster: UnknownCluster = {
    id: newPersonId(),
    descriptor: Array.from(descriptor),
    sampleCount: 1,
    thumbnail: await thumbnail(),
    lastSeen: Date.now(),
  };
  unknownClusters.push(cluster);
  sessionSeenIds.add(cluster.id);
  renderUnknownPanel();
  return { label: "Unknown", kind: "unknown" };
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  box: { x: number; y: number; width: number; height: number },
  label: string,
  color: string,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
  ctx.font = "16px system-ui, sans-serif";
  const textWidth = ctx.measureText(label).width;
  ctx.fillStyle = color;
  ctx.fillRect(box.x, box.y - 22, textWidth + 12, 22);
  ctx.fillStyle = "#0f1115";
  ctx.fillText(label, box.x + 6, box.y - 6);
}

async function detectAndRender(): Promise<void> {
  if (busy || video.readyState < 2) return;
  busy = true;
  try {
    const detections = await faceapi
      .detectAllFaces(video, detectorOptions())
      .withFaceLandmarks(true)
      .withFaceDescriptors();

    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    if (canvas.width !== displaySize.width || canvas.height !== displaySize.height) {
      faceapi.matchDimensions(canvas, displaySize);
    }
    const resized = faceapi.resizeResults(detections, displaySize);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < resized.length; i++) {
      const descriptor = detections[i].descriptor;
      const rawDetection = detections[i].detection;
      const result = await resolveSighting(descriptor, () => cropThumbnail(video, rawDetection));
      const color = result.kind === "known" ? "#3fcf8e" : "#e0a95a";
      drawBox(ctx, resized[i].detection.box, result.label, color);
    }

    pruneStaleUnknowns();
    updateCounts(resized.length);
    renderRosterPanel();
  } catch (err) {
    console.error("detection loop error", err);
  } finally {
    busy = false;
  }
}

function pruneStaleUnknowns(): void {
  const now = Date.now();
  const before = unknownClusters.length;
  unknownClusters = unknownClusters.filter((c) => now - c.lastSeen < STALE_UNKNOWN_MS);
  if (unknownClusters.length !== before) renderUnknownPanel();
}

function renderUnknownPanel(): void {
  unknownListEl.innerHTML = "";
  if (unknownClusters.length === 0) {
    unknownListEl.innerHTML = '<p class="empty-note">No unrecognized faces right now.</p>';
    return;
  }
  for (const cluster of unknownClusters) {
    const card = document.createElement("div");
    card.className = "unknown-card";

    const img = document.createElement("img");
    img.src = cluster.thumbnail;
    img.alt = "Unidentified face";
    card.appendChild(img);

    const form = document.createElement("form");
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Enter name";
    input.required = true;
    form.appendChild(input);

    const submit = document.createElement("button");
    submit.type = "submit";
    submit.textContent = "Remember";
    form.appendChild(submit);

    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.textContent = "✕";
    dismiss.title = "Dismiss without naming";
    form.appendChild(dismiss);

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      void nameCluster(cluster.id, input.value.trim());
    });
    dismiss.addEventListener("click", () => {
      unknownClusters = unknownClusters.filter((c) => c.id !== cluster.id);
      renderUnknownPanel();
    });

    card.appendChild(form);
    unknownListEl.appendChild(card);
  }
}

async function nameCluster(clusterId: string, name: string): Promise<void> {
  if (!name) return;
  const cluster = unknownClusters.find((c) => c.id === clusterId);
  if (!cluster) return;

  const person: Person = {
    id: cluster.id,
    name,
    descriptor: cluster.descriptor,
    sampleCount: cluster.sampleCount,
    thumbnail: cluster.thumbnail,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  roster.push(person);
  await savePerson(person);

  unknownClusters = unknownClusters.filter((c) => c.id !== clusterId);
  renderUnknownPanel();
  renderRosterPanel();
  updateCounts(Number(liveCountEl.textContent));
}

function relativeTime(ms: number): string {
  const seconds = Math.round((Date.now() - ms) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function renderRosterPanel(): void {
  rosterListEl.innerHTML = "";
  if (roster.length === 0) {
    rosterListEl.innerHTML = '<p class="empty-note">No one in the roster yet.</p>';
    return;
  }
  const sorted = [...roster].sort((a, b) => b.updatedAt - a.updatedAt);
  for (const person of sorted) {
    const row = document.createElement("div");
    row.className = "roster-row";

    const left = document.createElement("div");
    left.textContent = person.name;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `seen ${person.sampleCount}x · ${relativeTime(person.updatedAt)}`;

    const right = document.createElement("div");
    right.appendChild(meta);

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "✕";
    removeBtn.title = `Forget ${person.name}`;
    removeBtn.addEventListener("click", () => void forgetPerson(person.id));
    right.appendChild(removeBtn);

    row.appendChild(left);
    row.appendChild(right);
    rosterListEl.appendChild(row);
  }
}

async function forgetPerson(id: string): Promise<void> {
  roster = roster.filter((p) => p.id !== id);
  await deletePerson(id);
  renderRosterPanel();
  updateCounts(Number(liveCountEl.textContent));
}

async function startCamera(): Promise<void> {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    setStatus(`Couldn't access the camera: ${(err as Error).message}`);
    return;
  }

  video.srcObject = stream;
  await video.play();
  videoWrap.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;

  detectTimer = setInterval(() => void detectAndRender(), DETECT_INTERVAL_MS);
  cameraToggle.textContent = "Stop camera";
  setStatus("Camera on — looking for faces…");
}

function stopCamera(): void {
  if (detectTimer) clearInterval(detectTimer);
  detectTimer = null;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  liveCountEl.textContent = "0";
  cameraToggle.textContent = "Start camera";
  setStatus("Camera stopped.");
}

cameraToggle.addEventListener("click", () => {
  if (stream) stopCamera();
  else void startCamera();
});

clearDataBtn.addEventListener("click", async () => {
  if (!confirm("Forget everyone the app has learned? This can't be undone.")) return;
  await clearRoster();
  roster = [];
  unknownClusters = [];
  sessionSeenIds.clear();
  renderRosterPanel();
  renderUnknownPanel();
  updateCounts(Number(liveCountEl.textContent));
});

rosterUpload.addEventListener("change", async () => {
  const files = Array.from(rosterUpload.files ?? []);
  rosterUpload.value = "";
  if (files.length === 0 || !modelsReady) return;

  for (const file of files) {
    setStatus(`Reading ${file.name}…`);
    const img = await loadImageFile(file);
    const result = await faceapi
      .detectSingleFace(img, detectorOptions())
      .withFaceLandmarks(true)
      .withFaceDescriptor();

    if (!result) {
      setStatus(`No face found in ${file.name}, skipped.`);
      continue;
    }
    await resolveSighting(result.descriptor, () => cropThumbnail(img, result.detection));
  }
  renderRosterPanel();
  updateCounts(Number(liveCountEl.textContent));
  setStatus("Roster photo(s) processed. Name any new faces on the right.");
});

function loadImageFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Could not load ${file.name}`));
    };
    img.src = url;
  });
}

async function init(): Promise<void> {
  setStatus("Loading models…");
  cameraToggle.disabled = true;
  const [, loadedRoster] = await Promise.all([loadModels(), loadRoster()]);
  roster = loadedRoster;
  cameraToggle.disabled = false;
  renderRosterPanel();
  renderUnknownPanel();
  updateCounts(0);
  setStatus('Models loaded. Click "Start camera" to begin.');
}

void init();
