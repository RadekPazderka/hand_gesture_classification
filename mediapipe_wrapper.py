import numpy as np
import mediapipe as mp
from typing import List

from lib.coordinates.point import Point
from lib.coordinates.keypoints.hand_keypoints import HandKeypoints


class MediapipeWrapper(object):
    def __init__(self, max_num_hands: int = 1, detection_conf: float = 0.7, tracking_conf: float = 0.5):

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def _postprocess(self, landmarks, h, w):
        landmark_points = []

        # Keypoint
        for landmark in landmarks.landmark:
            landmark_x = max(0, min(int(landmark.x * w), w - 1))
            landmark_y = max(0, min(int(landmark.y * h), h - 1))
            # landmark_z = landmark.z
            landmark_points.append(Point(landmark_x, landmark_y))
        return HandKeypoints(landmark_points)

    def process(self, frame_rgb: np.ndarray) -> List[HandKeypoints]:
        res = self._hands.process(frame_rgb)
        h, w, _ = frame_rgb.shape

        hands = []
        if res.multi_hand_landmarks is not None:
            for landmarks in res.multi_hand_landmarks:
                hands.append(self._postprocess(landmarks, h, w))
        return hands
