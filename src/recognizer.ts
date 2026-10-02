import * as faceapi from "@vladmandic/face-api";

interface Embedded {
  descriptor: number[];
  sampleCount: number;
}

/** Below this euclidean distance, a detection is considered the same person. */
export const MATCH_THRESHOLD = 0.5;

/**
 * Only fold a sighting into a person's running-average descriptor when the
 * match is this confident, so a borderline/ambiguous match can't drag their
 * stored embedding toward someone else's face.
 */
export const LEARN_THRESHOLD = 0.42;

/**
 * Cap on the effective sample weight. Without this, a person seen for
 * thousands of frames would make their embedding nearly immovable, even if
 * their appearance changes (lighting, glasses, haircut).
 */
const MAX_SAMPLE_WEIGHT = 50;

export function findBestMatch<T extends Embedded>(
  descriptor: Float32Array,
  candidates: T[],
): { item: T; distance: number } | null {
  let best: { item: T; distance: number } | null = null;
  for (const item of candidates) {
    const distance = faceapi.euclideanDistance(descriptor, item.descriptor);
    if (distance < MATCH_THRESHOLD && (!best || distance < best.distance)) {
      best = { item, distance };
    }
  }
  return best;
}

/**
 * Folds a new confirmed sighting into a person's stored descriptor using a
 * weighted running mean, so recognition keeps adapting to how someone
 * actually looks across sessions instead of freezing on the first photo.
 */
export function foldDescriptor(existing: Embedded, descriptor: Float32Array): number[] {
  const weight = Math.min(existing.sampleCount, MAX_SAMPLE_WEIGHT);
  const next = new Array(descriptor.length);
  for (let i = 0; i < descriptor.length; i++) {
    next[i] = (existing.descriptor[i] * weight + descriptor[i]) / (weight + 1);
  }
  return next;
}

export function newPersonId(): string {
  return crypto.randomUUID();
}
