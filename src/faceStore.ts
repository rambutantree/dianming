import { createStore, keys, get, set, del, values } from "idb-keyval";
import type { Person } from "./types";

const store = createStore("dianming-db", "roster");

export async function loadRoster(): Promise<Person[]> {
  return values<Person>(store);
}

export async function savePerson(person: Person): Promise<void> {
  await set(person.id, person, store);
}

export async function deletePerson(id: string): Promise<void> {
  await del(id, store);
}

export async function clearRoster(): Promise<void> {
  const ids = await keys(store);
  await Promise.all(ids.map((id) => del(id, store)));
}

export async function getPerson(id: string): Promise<Person | undefined> {
  return get(id, store);
}
