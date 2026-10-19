"""Live ASL spelling game.

Run from the project root (venv active):
    python app/webcam_demo.py

Spell the word letter by letter. A letter counts only after the model sees it
for --frames consecutive frames with probability >= --thresh. For repeated
letters (e.g. LL) drop your hand or change sign in between.
Keys: n = new word | r = restart word | Esc = quit
"""
import argparse
import random
import time
from pathlib import Path

import cv2

from asl_letters.keypoints import detect, make_options, normalize
from asl_letters.model import load_checkpoint, predict_letter
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
LANDMARKER_PATH = ROOT / "models" / "mediapipe" / "hand_landmarker.task"
CKPT_PATH = ROOT / "models" / "mlp_v1.pt"

# T and DEL/SPACE were the weakest on my own hand; edit freely
WORDS = ["CAB", "BEE", "FIG", "HAND", "SIGN", "WORLD", "HELLO", "LOVE", "GLOVE", "PLAY",
         "FROG", "WAVE", "BOOK", "SKY", "YOU"]


def draw_hand(img, hand):
    h, w = img.shape[:2]
    for i, p in enumerate(hand):
        cv2.circle(img, (int(p.x * w), int(p.y * h)), 3, (0, 0, 255) if i == 0 else (0, 255, 0), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.7, help="min probability")
    ap.add_argument("--frames", type=int, default=10, help="consecutive correct frames to accept")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    model, classes, mean, std = load_checkpoint(CKPT_PATH)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)       # Windows backend
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam")

    word, idx, streak = random.choice(WORDS), 0, 0
    last_accepted, released, done_until = None, True, 0.0
    t_prev, fps = time.perf_counter(), 0.0

    with vision.HandLandmarker.create_from_options(make_options(LANDMARKER_PATH)) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                       # same as training (decided in notebook 04)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # MediaPipe expects RGB
            res = detect(lm, rgb)

            label, prob, found = "nothing", 1.0, False
            if res.hand_landmarks:
                h, w = rgb.shape[:2]
                feats = normalize(res.hand_landmarks[0], w, h)
                if feats is not None:
                    label, prob = predict_letter(model, classes, mean, std, feats)
                    found = True
                    draw_hand(frame, res.hand_landmarks[0])

            now = time.perf_counter()
            if now < done_until:                             # word finished: celebrate, then next word
                cv2.putText(frame, "DONE!  " + word, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            else:
                if done_until:
                    word, idx, streak, last_accepted, released, done_until = random.choice(WORDS), 0, 0, None, True, 0.0
                target = word[idx]
                if label != last_accepted:
                    released = True                          # hand dropped or sign changed since last accept
                good = found and label == target and prob >= args.thresh and (target != last_accepted or released)
                streak = streak + 1 if good else 0
                if streak >= args.frames:
                    last_accepted, released, streak = target, False, 0
                    idx += 1
                    if idx == len(word):
                        done_until = now + 1.5

                # HUD
                shown = "".join(c if i < idx else "_" for i, c in enumerate(word))
                cv2.putText(frame, shown, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
                if idx < len(word):
                    cv2.putText(frame, f"sign: {word[idx]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.rectangle(frame, (20, 105), (20 + 200, 120), (80, 80, 80), 1)
                cv2.rectangle(frame, (20, 105), (20 + int(200 * min(streak / args.frames, 1)), 120), (0, 255, 0), -1)

            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(frame, f"seeing: {label} ({prob:.2f})   fps {fps:.0f}", (20, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("ASL finger-speller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("n"):
                word, idx, streak, last_accepted, released = random.choice(WORDS), 0, 0, None, True
            elif key == ord("r"):
                idx, streak, last_accepted, released = 0, 0, None, True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()