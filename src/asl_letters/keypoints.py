"""Hand keypoint extraction shared by training notebooks and the live app.

The webcam app must call exactly these functions, with the same CONF, so live
features match the training features.
"""
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CONF = 0.3  # min detection + presence confidence; chosen in notebook 02


def make_options(model_path, conf=CONF):
    return vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
    )


def detect(landmarker, rgb):
    """Run the landmarker on an RGB uint8 array."""
    return landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


def normalize(hand, width, height):
    """21 landmarks -> 63 features: pixel units, wrist at origin, scaled by palm length."""
    pts = np.array([[p.x * width, p.y * height, p.z * width] for p in hand], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9, :2])
    if scale < 1e-6:
        return None
    return (pts / scale).reshape(-1)
