import cv2
import numpy as np
from typing import Tuple

from lib.coordinates.point import Point


class TextDrawer(object):
    def __init__(self, font_size=1.):
        #type: (float) -> None
        self._font_size = font_size
        self._text_data = {}

    def set_font_size(self, size: float):
        self._font_size = size

    def add_text(self, point, text, color=(0,0,255)):
        #type: (Point, str, Tuple[int, int, int]) -> None
        point_id = id(point)
        if point_id not in self._text_data:
            self._text_data[point_id] = {"point": point,
                                     "lines": []}
        self._text_data[point_id]["lines"].append({
            "txt": text,
            "color": color
        })

    def draw_text(self, frame: np.ndarray) -> None:
        for bbox_id, bbox_data in self._text_data.items():
            point = bbox_data["point"]
            xmin, ymin = point.to_coord()

            img_height, img_width, _ = frame.shape
            if (xmin >= img_width - 1) or (ymin >= img_height - 1):
                print("Skip drawining text...")
                continue
            text_blobs = [self._put_text(line["txt"], color=line["color"]) for line in bbox_data["lines"]]

            begin_y = max(0, ymin - sum(list(map(lambda x: x.shape[0], text_blobs))))
            for txt_blob in text_blobs:
                h, w, _ = txt_blob.shape
                xmin = max(0, xmin)
                pad_w = max(0, (xmin + w) - img_width)
                frame[begin_y:begin_y+h, xmin:xmin+w] = txt_blob[0:h, 0:w-pad_w]
                begin_y += h
        self._text_data = {}

    def _put_text(self, text, color=(0,0,255)):
        #type: (str, Tuple[int, int, int]) -> np.ndarray
        zeros = np.zeros((100, 400, 3), dtype=np.uint8)
        zeros = cv2.putText(zeros, "{}".format(text), (0, 50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=self._font_size, color=color, thickness=1, lineType=cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(zeros, cv2.COLOR_BGR2GRAY))
        return zeros[y:y+h, x:x+w]
