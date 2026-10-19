"""Collect your own labelled webcam frames (v2: cannot silently mislabel).

Run from the project root (venv active):
    python app/collect_webcam.py --session s2

Keys:  a-z = pick letter | 1 = space | 2 = del | 0 = nothing (empty scene)
       Enter = record a burst (only after you picked a label; hold the sign still-ish)
       Esc = quit
- A burst needs a FRESH label key press, so pressing Enter twice can never save
  a second sign under the old label.
- A label key pressed while recording is queued and becomes active when the burst ends.
- The preview is mirrored for comfort, but the SAVED frames are RAW (not flipped).
  Notebook 04 flips them before detection, exactly like the live app must.
- Keep the WHOLE hand inside the frame, close to the camera, away from your face.
Session names must not contain "_".
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
    ap.add_argument("--session", default="s2")
    ap.add_argument("--burst", type=int, default=12, help="frames per burst")
    ap.add_argument("--interval", type=float, default=0.25, help="seconds between burst frames")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    assert "_" not in args.session, "session name must not contain '_'"

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)      # Windows backend
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam")

    label, armed, pending = None, False, None
    burst_left, last_save, counts = 0, 0.0, {}
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
            if burst_left == 0:                              # burst finished
                if pending:
                    label, armed, pending = pending, True, None
                else:
                    armed = False                            # must press a label key again

        view = cv2.flip(frame, 1)                            # mirrored preview only
        cv2.putText(view, str(label or "-"), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        if burst_left > 0:
            msg, col = f"RECORDING {burst_left}" + (f"  (next: {pending})" if pending else ""), (0, 0, 255)
        elif armed:
            msg, col = f"READY - press Enter to record {label}", (0, 255, 0)
        else:
            msg, col = "press a label key (a-z, 1=space, 2=del, 0=nothing)", (255, 255, 255)
        cv2.putText(view, msg, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(view, f"session {args.session} | saved {counts.get(label, 0) if label else 0} for {label or '-'}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("collect_webcam", view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                                        # Esc
            break
        elif key in LABEL_KEYS:
            if burst_left == 0:
                label, armed = LABEL_KEYS[key], True
            else:
                pending = LABEL_KEYS[key]
        elif key == 13 and burst_left == 0 and armed and label:   # Enter
            burst_left, last_save, armed = args.burst, 0.0, False

    cap.release()
    cv2.destroyAllWindows()
    print("saved this run:", counts)


if __name__ == "__main__":
    main()