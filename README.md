# dianming

点名 (diǎnmíng) — "roll call / to call the register":

A browser app that watches your webcam, counts the faces currently in frame,
and learns people's names over time. Everything — face embeddings, names,
thumbnails — stays on your device; nothing is uploaded anywhere.

## How it works

- **Counting**: every ~350ms the app runs face detection on the webcam feed
  and draws a box around each face, with a live count of faces currently
  in frame.
- **Recognition**: each detected face is turned into a 128-dimension
  descriptor ([face-api.js](https://github.com/vladmandic/face-api)) and
  compared against everyone already in the roster. A close enough match is
  labelled with that person's name.
- **Progressive learning**: you can teach the app names two ways —
  - *Live*: when a face doesn't match anyone, it shows up in the
    "Unrecognized faces" panel with a thumbnail. Type a name once and the
    app remembers that face from then on.
  - *Preload*: upload one or more photos ("Preload roster photo") and the
    same naming panel appears for faces found in them.
  Every time a known person is seen again, their stored descriptor is
  nudged towards the new sighting (a capped running average), so matching
  stays robust to lighting, angle, and small appearance changes over time.
- **Storage**: the roster (name + descriptor + thumbnail) is kept in the
  browser's IndexedDB. It survives page reloads but never leaves your
  machine. "Clear learned data" wipes it.

## Getting started

```bash
npm install     # also copies face-api.js model weights into public/models
npm run dev      # starts a local dev server (prints the URL to open)
```

Open the printed URL in a browser, click **Start camera**, and allow camera
access. Chrome/Edge/Firefox on desktop work best; camera access requires
`https://` or `localhost`.

## Notes & limitations

- Face matching uses a fixed distance threshold tuned for face-api.js's
  recognition model; it's good for casual roll-call use, not a verified
  biometric system.
- Works with one camera at a time, in-browser, client-side only — no
  backend, no network calls after the initial page/model load.
- The "Unrecognized faces" panel keeps a face pending for up to a minute of
  inactivity before dropping it, so stepping out of frame briefly doesn't
  lose the in-progress match.
