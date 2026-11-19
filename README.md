# ASL Finger-Speller (`asl-letters`)

A real-time webcam game that recognizes static ASL fingerspelling letters. Spell the word on screen, one letter at a time, using your hand.

**Pipeline:** webcam frame → MediaPipe Hand Landmarker (21 hand keypoints) → normalization → small MLP trained from scratch in PyTorch → letter.

> *Add a short demo GIF here (`docs/demo.gif`).*

## Results

| Test | Result |
|---|---|
| Dataset validation (block split, same signer as training) | 93% accuracy, 0.92 macro-F1 over 28 classes |
| Logistic-regression baseline, same split | 0.90 macro-F1 |
| **My own webcam, fresh session (different day, place, clothes)** | **92.6%** system accuracy over 336 frames (about 96% excluding `del`) |

"System accuracy" counts a frame where no hand was detected as a wrong answer, which is what a player experiences. Of the 29 classes, 23 were perfect on the fresh session; the weak ones were `del`, T, M, O and S.

## How it works

1. **Keypoints, not pixels.** The dataset is video clips cut into frames, so a CNN on pixels would memorize backgrounds and lighting. MediaPipe's 21 landmarks discard those.
2. **Normalization.** Landmarks are converted to pixel units, translated so the wrist is the origin, and divided by palm length (wrist to middle knuckle). Hand position and distance from the camera stop mattering. Result: 63 features.
3. **Classifier.** MLP `63 → 128 → 64 → 28` with ReLU, dropout 0.2, weight decay, and inverse-frequency class weights. Output is 28 classes (A-Z, `del`, `space`). "No hand detected" is treated as NOTHING and never reaches the model.
4. **Game.** A letter is accepted after 10 consecutive frames with probability ≥ 0.7. Repeated letters (the LL in HELLO) require dropping the hand or changing sign in between.

## What I learned (decisions backed by measurements)

- **Random splits lie on video data.** A 1-nearest-neighbour model scored 0.985 on a random split but 0.874 on a block split, because near-twin frames land on both sides. Frames stay similar for hundreds of frames (measured in notebook 01), so validation uses early frames for training, late frames for validation, with a 200-frame buffer.
- **A strong baseline matters.** Logistic regression reached 0.90 macro-F1; the MLP added about 2 points. Removing dropout and weight decay hurt more (final validation accuracy 0.889 vs 0.925) than changing network size.
- **MediaPipe misses hands unevenly.** At its default confidence it found hands in only 45% of N frames and 51% of M. Lowering the detection and presence confidence to 0.3 recovered about 5% more frames (no false hands in a 100-frame sample of `nothing`, 14 in all 3,000 on the full run). Padding and 2x upscaling made detection worse.
- **Flip or not was decided by data.** Flipping webcam frames before detection gave 79.6% vs 8.5% system accuracy raw. The dataset behaves like a mirrored selfie view.
- **Validation is not deployment.** Evaluation on my own webcam exposed problems validation could not: hand placement (a hand cut off by the frame edge), and a bug in my own data-collection script that mislabeled bursts. Both were found by looking at the failure images.

## Known limitations

- **Works mainly with one hand.** The model was trained on one signer's single hand, so keypoints from the other hand are mirror images it has never seen. In my testing it works mainly with my left hand. (A fix is listed below.)
- **`del` and `space` are unreliable.** They are the dataset author's own signs, not standard ASL, and MediaPipe often cannot detect them. The game uses only A-Z.
- **T is weak** (often read as L), and **M/N/A** are confusable thumb-tucked fists. N is also detected in only about half of dataset frames.
- **J and Z** are motion letters; they are recognized as static shapes only.
- **One tester.** The "own webcam" results are one person, one camera, two sessions.
- Runs on CPU at interactive speed (about 16 ms per frame for detection).

## Setup (Windows)

```
python -m venv asl-letters-venv          # Python 3.10
asl-letters-venv\Scripts\activate
pip install mediapipe opencv-python numpy pandas scikit-learn torch matplotlib jupyter ipykernel
pip install -e .
```

1. Download the dataset from Kaggle (`grassknoted/asl-alphabet`) into `data/raw/` so you have `data/raw/asl_alphabet_train/<class>/` and `data/raw/asl_alphabet_test/`.
2. Download `hand_landmarker.task` from Google's MediaPipe Hand Landmarker docs into `models/mediapipe/`.
3. Run the notebooks in order (below), then `python app/webcam_demo.py`.

Data and model files are not committed.

## Project structure

```
notebooks/
  01_explore_data.ipynb        dataset checks, near-duplicate frame analysis
  02_extract_keypoints.ipynb   MediaPipe, detection-rate experiments, normalization, full extraction
  03_train_classifier.ipynb    block split, baselines, MLP, design experiments, confusions
  04_own_webcam_eval.ipynb     flip decision, accuracy on my own hand, thresholds
src/asl_letters/
  keypoints.py                 detector options, detect, normalize (shared by notebooks and app)
  model.py                     MLP definition and checkpoint loading
app/
  collect_webcam.py            record labelled frames from the webcam
  webcam_demo.py               the spelling game
data/{raw,processed,own_webcam}/   (gitignored)
models/                        (gitignored)
```

## Future work

- **Mirror augmentation** (negate the x coordinates of training features) so the model works with either hand.
- Train on my own webcam frames in addition to the dataset, to fix T and M.
- Sequence model for the motion letters J and Z.
- Handle SPACE/DEL with a deliberate gesture or the keyboard.

## Credits

Dataset: Kaggle `grassknoted/asl-alphabet`. Hand landmarks: Google MediaPipe Hand Landmarker. Built as a learning project, with Claude as a tutor and pair-programmer; I ran, debugged and validated every step, and the notebooks keep my mistakes on purpose.
