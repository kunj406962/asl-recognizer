"""Collect your own labelled webcam frames.

Run from the project root (venv active):
    python app/collect_webcam.py --session s1

Keys:  a-z = pick letter | 1 = space | 2 = del | 0 = nothing (empty scene)
       Enter = record a burst (hold the sign and move it slightly)
       Esc = quit
The preview is mirrored for comfort, but the SAVED frames are RAW (not flipped),
so notebook 04 can test raw vs flipped. Session names must not contain "_".
"""
import argparse
import time
from pathlib import Path

import cv2

LABEL_KEYS = {ord(c): c.upper() for c in "abcdefghijklmnopqrstuvwxyz"}
LABEL_KEYS.update({ord("1"): "space", ord("2"): "del", ord("0"): "nothing"})
OUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "own_webcam"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="s1")
    ap.add_argument("--burst", type=int, default=15, help="frames per burst")
    ap.add_argument("--interval", type=float, default=0.2, help="seconds between burst frames")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    assert "_" not in args.session, "session name must not contain '_'"

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)      # Windows backend
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam")

    label, burst_left, last_save, counts = None, 0, 0.0, {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if burst_left > 0 and now - last_save >= args.interval:
            d = OUT_ROOT / label
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{label}_{args.session}_{int(now * 1000)}.jpg"), frame)  # RAW frame
            counts[label] = counts.get(label, 0) + 1
            burst_left -= 1
            last_save = now

        view = cv2.flip(frame, 1)                           # mirrored preview only
        status = f"label: {label or '-'}  saved: {counts.get(label, 0) if label else 0}  session: {args.session}"
        cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if burst_left > 0:
            cv2.putText(view, f"REC {burst_left}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(view, "pick a key, Enter = record, Esc = quit", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("collect_webcam", view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                                       # Esc
            break
        elif key in LABEL_KEYS and burst_left == 0:
            label = LABEL_KEYS[key]
        elif key == 13 and label and burst_left == 0:       # Enter
            burst_left, last_save = args.burst, 0.0

    cap.release()
    cv2.destroyAllWindows()
    print("saved this run:", counts)


if __name__ == "__main__":
    main()
