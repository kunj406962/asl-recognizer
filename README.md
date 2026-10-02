# ASL Letters — Static ASL Alphabet Recognition + Live Webcam Spelling Game

> Status: **fresh start.** No code, no data, no environment set up yet. This
> document is the plan we build from. (Suggested repo name: `asl-letters` —
> rename if you'd rather call it something else, nothing below depends on
> the name.)

This is a from-scratch follow-up to an earlier, much heavier project
(sequence-based ASL *fingerspelling* recognition — LSTM + CTC over
video-frame landmark sequences). That project is fully deleted (~200GB of
data) and is **not** being resumed or referenced here except as a lesson
in scope: this project is deliberately smaller, faster to iterate on, and
built around a single static image → single letter prediction, not a
continuous signed sequence.

---

## 1. What we're building

A model that looks at **one photo of a hand making an ASL letter shape**
and predicts which letter it is — then a live webcam app that does the
same thing in real time, wrapped in a simple game: the app shows you a
word to spell, you sign it one letter at a time, and it confirms each
letter before moving to the next.

**What makes this a *trained-from-scratch* project**, not just gluing
together existing tools: the actual letter-classification model is one
we train ourselves, from zero, on labeled data. We do use one pretrained
third-party component — **MediaPipe Hands** — but only for hand
*localization/tracking* (it outputs 21 generic hand-joint keypoints; it
has no concept of ASL or letters at all). That's infrastructure, the same
role OpenCV or a face detector plays in any CV pipeline — not the thing
we're here to learn.

### Why keypoints instead of raw pixels (a CNN on the images directly)

Early look at the dataset (see §3) strongly suggests it's really just a
handful of distinct hand poses, each captured across thousands of
near-duplicate video frames (same background, same lighting, same wrist
angle, sequential filenames). A CNN trained directly on those pixels
would have an easy time memorizing *this specific background and
lighting*, and a hard time generalizing to a real webcam — classic
overfitting-to-nuisance-features trap.

Routing through MediaPipe's keypoints sidesteps most of that: a keypoint
is a set of (x, y, z) joint positions *relative to the hand itself* — it
mostly doesn't care what the background or lighting looks like, since
MediaPipe itself was trained on a huge, diverse set of real-world hands
already. We're not fully immune to the dataset's low diversity (still
worth testing on our own webcam, still worth collecting some of our own
data — that's V2), but it meaningfully de-risks the generalization
problem before we've written a line of our own training code.

It's also a much smaller, simpler model: 21 keypoints × (x, y, z) = 63
numbers in, a small MLP, 29-way softmax out. No convolutions, no
augmentation pipeline required for V1, trains in minutes.

---

## 2. Scope

### V1 — core pipeline (this phase)

1. Download + explore the dataset.
2. Run MediaPipe Hands over every training image, extract 21 keypoints
   per image, save as a lightweight feature table (not re-decoding raw
   images every time we touch this later).
3. Train a small MLP from scratch: 63 features → predicted letter (29
   classes: A–Z, SPACE, DEL, NOTHING).
4. Evaluate — including, critically, **testing against a few of our own
   webcam photos**, not just the dataset's own held-out split, since
   that's the real test of whether this generalizes at all.
5. Build the live webcam demo: MediaPipe Hands on the live feed → same
   keypoint extraction → trained MLP → predicted letter on screen.
6. Build the "spell the word" game on top of that: show a target word,
   highlight the current letter, require N consecutive correctly-predicted
   frames before advancing (debounce, so one lucky/noisy frame doesn't
   falsely trigger), celebrate on completion.

### V2 — robustness (after V1 works end-to-end)

- Collect a small set of our own webcam images per letter (even
  30–50/letter) to mix into training or fine-tune on, closing the
  generalization gap directly instead of hoping MediaPipe's invariance
  is enough on its own.
- Decide how SPACE / DEL / NOTHING should behave in the actual game UI
  (e.g. SPACE between words, DEL to redo a letter).
- Confidence thresholding / "no hand detected" handling in the demo.
- General polish — whatever V1 testing reveals is actually needed, rather
  than guessing now.

Nothing in V2 is being built yet. V1 is the whole focus until it works.

---

## 3. The dataset

**ASL Alphabet** (Kaggle, `grassknoted/asl-alphabet`):
<https://www.kaggle.com/datasets/grassknoted/asl-alphabet>

- 87,000 training images, 200×200 px, 29 classes: A–Z plus `space`,
  `del`, `nothing`.
- Test split is deliberately tiny (~29 images) — the dataset authors'
  own hint that you're meant to bring your own real-world test images
  rather than trust their split.
- **Known caveat (seen directly in a sample of the data before download
  even completed):** filenames and thumbnails for one letter's folder
  showed what looks like a single short clip split into thousands of
  sequential frames (same doorframe in shot, same lighting, same wrist
  angle, filenames like `B1.jpg, B2.jpg, ..., B1001.jpg...`). Treat the
  "87,000 images" figure as optimistic for actual pose diversity — this
  is the whole reason V1 routes through keypoints instead of raw pixels,
  and why V2 exists at all.
- Not downloaded yet. Expected to land in `data/raw/` (see §5) — exact
  subfolder names (`asl_alphabet_train/`, `asl_alphabet_test/`) to be
  confirmed once downloaded, documented here once known.

---

## 4. Environment

- Python virtual env (name TBD when we set it up — suggest
  `asl-letters-env` unless you'd rather reuse/name it differently).
- Expected core dependencies (exact versions pinned once installed):
  - `mediapipe` — hand keypoint detection/tracking
  - `opencv-python` — webcam capture, image I/O, drawing overlays
  - `torch` — model definition + training
  - `numpy`, `pandas` — keypoint feature tables
  - `scikit-learn` — train/val split, maybe a quick baseline classifier
    for sanity-checking the MLP against
- `requirements.txt` generated once the env is actually set up — not
  guessing exact versions in this doc ahead of time.

---

## 5. File structure

Same working process as last time: **notebooks first** for exploration,
diagnostics, and benchmarking (with markdown cells explaining *why*,
mistakes kept rather than deleted), then logic that's actually settled
gets promoted into a real importable `src/` module once it's proven out
— not written directly into a script from scratch.

```
asl-letters/
├── README.md                  # this file
├── .gitignore
├── requirements.txt           # pinned once env is set up
│
├── data/
│   ├── raw/                   # downloaded Kaggle dataset, untouched (gitignored)
│   └── processed/             # extracted keypoints + labels, train/val/test splits (gitignored)
│
├── notebooks/
│   ├── 01_explore_data.ipynb          # look at the raw data, confirm class folders/counts,
│   │                                   #   sample images, re-confirm the diversity caveat above
│   ├── 02_extract_keypoints.ipynb     # MediaPipe on one sample image first, then the full set;
│   │                                   #   benchmark before committing to the full 87k run
│   ├── 03_train_classifier.ipynb      # MLP training, sanity checks, val accuracy
│   ├── 04_own_webcam_eval.ipynb       # the real test: does it work on OUR hand/background
│   └── 05_webcam_demo_prototype.ipynb # live loop prototype before promoting to src/app
│
├── src/
│   └── asl_letters/
│       ├── __init__.py
│       ├── keypoints.py       # MediaPipe wrapper: extract_keypoints(image) -> np.ndarray | None
│       ├── dataset.py         # PyTorch Dataset over the extracted keypoint feature table
│       ├── model.py           # the MLP classifier definition
│       ├── train.py           # training loop, promoted from notebook 03 once settled
│       └── game.py            # "spell the word" state machine (current letter, debounce counter)
│
├── models/
│   └── checkpoints/           # saved .pt files (gitignored)
│
└── app/
    └── webcam_demo.py         # final live entrypoint: python app/webcam_demo.py
```

---

## 6. `.gitignore` (starting point)

```gitignore
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Virtual environment
asl-letters-env/
venv/
*.env

# Data — never commit the dataset or generated features
data/raw/
data/processed/

# Model checkpoints
models/checkpoints/

# OS cruft
.DS_Store
Thumbs.db
```

(Adjust the venv folder name here once it's actually created, if it ends
up named differently.)

---

## 7. Open questions / decisions still pending

- [ ] Confirm repo/folder name (`asl-letters` is just a placeholder).
- [ ] Confirm venv name.
- [ ] Download the dataset, confirm actual subfolder structure inside
      `data/raw/`, update §3 with the real paths.
- [ ] Decide exactly how many consecutive frames the game's debounce
      should require before confirming a letter (start with a guess,
      tune once we feel how it plays).

---

## 8. Explicitly not doing right now

- No sequence modeling / continuous fingerspelling recognition — that
  was the old project. This one is single static letter in, single
  letter out, every time.
- No training our own hand-detector/localizer from scratch — MediaPipe
  handles that; our training effort goes entirely into the
  keypoints → letter classifier.
- No data augmentation pipeline in V1 — if V1's own-webcam testing (step
  4 above) shows it's needed, that becomes part of V2, not bolted on
  speculatively now.
