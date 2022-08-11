from typing import List

import cv2
import numpy as np

from lib.coordinates.point import Point
from lib.coordinates.bbox_2d import BBox


class HandKeypoints(object):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    CONNECTIONS = [
        (WRIST, THUMB_CMC),
        (THUMB_CMC, THUMB_MCP),
        (THUMB_MCP, THUMB_IP),
        (THUMB_IP, THUMB_TIP),

        (WRIST, INDEX_FINGER_MCP),
        (INDEX_FINGER_MCP, INDEX_FINGER_PIP),
        (INDEX_FINGER_PIP, INDEX_FINGER_DIP),
        (INDEX_FINGER_DIP, INDEX_FINGER_TIP),

        (WRIST, MIDDLE_FINGER_MCP),
        (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP),
        (MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP),
        (MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP),

        (WRIST, RING_FINGER_MCP),
        (RING_FINGER_MCP, RING_FINGER_PIP),
        (RING_FINGER_PIP, RING_FINGER_DIP),
        (RING_FINGER_DIP, RING_FINGER_TIP),

        (WRIST, PINKY_MCP),
        (PINKY_MCP, PINKY_PIP),
        (PINKY_PIP, PINKY_DIP),
        (PINKY_DIP, PINKY_TIP),

        (INDEX_FINGER_MCP, MIDDLE_FINGER_MCP),
        (MIDDLE_FINGER_MCP, RING_FINGER_MCP),
        (RING_FINGER_MCP, PINKY_MCP),
    ]

    def __init__(self, points: List[Point]):
        assert len(points) == 21, "ERROR: Hand has only 21 keypoints / points"
        self._points = points

    def to_bbox(self) -> BBox:
        return BBox(
            Point(x=min(self._points, key=lambda point: point.x).x, y=min(self._points, key=lambda point: point.y).y),
            Point(x=max(self._points, key=lambda point: point.x).x, y=max(self._points, key=lambda point: point.y).y))

    def draw(self, image: np.ndarray, line_thickness=5):
        pt_color = (255, 0, 0)
        for pt in self._points:
            pt.draw(image, color=pt_color, radius=10, filled=True)
        line_color = (0, 255, 0)

        for start_id, end_id in self.CONNECTIONS:
            cv2.line(image, self._points[start_id].to_coord(), self._points[end_id].to_coord(), line_color,
                     thickness=line_thickness)

    def to_normalized_list_xy(self) -> List[float]:
        # Convert to relative coordinates by WRIST position
        base_point = self._points[self.WRIST]  # index 0
        relative_xy_coords = []
        for point in self._points:
            relative_xy_coords.append(point.x - base_point.x)
            relative_xy_coords.append(point.y - base_point.y)

        # Normalization
        max_value = max(list(map(abs, relative_xy_coords)))
        normalized_xy_coords = list(map(lambda x: x / max_value, relative_xy_coords))
        return normalized_xy_coords
