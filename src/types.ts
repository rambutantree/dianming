export interface Person {
  id: string;
  name: string;
  /** Running mean of the 128-d face descriptor across all confirmed sightings. */
  descriptor: number[];
  /** How many sightings contributed to the mean (capped, see recognizer.ts). */
  sampleCount: number;
  /** Small data-URL face crop used as a thumbnail in the roster list. */
  thumbnail: string;
  createdAt: number;
  updatedAt: number;
}

/** An as-yet-unnamed face being tracked in memory only (not persisted). */
export interface UnknownCluster {
  id: string;
  descriptor: number[];
  sampleCount: number;
  thumbnail: string;
  lastSeen: number;
}

export interface MatchResult {
  person: Person;
  distance: number;
}
