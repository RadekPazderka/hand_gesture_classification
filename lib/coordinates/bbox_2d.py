from __future__ import annotations

import math
import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple, List, Union

from lib.coordinates.point import Point
from lib.image_utils.text_drawer import TextDrawer


class BBox(object):

    @classmethod
    def from_kalman_x_bbox(cls, x: np.ndarray):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns BBox instance
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        xmin = x[0] - w / 2.
        ymin = x[1] - h / 2.
        xmax = x[0] + w / 2.
        ymax = x[1] + h / 2.
        return cls.from_min_max(xmin, ymin, xmax, ymax)

    @classmethod
    def from_cvat_annot(cls, cvat_data: Dict[str, Any]):
        min_point = Point(cvat_data["xtl"], cvat_data["ytl"])
        max_point = Point(cvat_data["xbr"], cvat_data["ybr"])
        return cls(min_point, max_point)

    @classmethod
    def from_min_max(cls, xmin: Union[int, float], ymin: Union[int, float],
                     xmax: Union[int, float], ymax: Union[int, float]):
        return cls(Point(xmin, ymin), Point(xmax, ymax))

    @classmethod
    def from_sort_array(cls, data: np.ndarray):
        min_point = Point(x=data[0], y=data[1])
        max_point = Point(x=data[2], y=data[3])
        return cls(min_point, max_point)

    @classmethod
    def from_list(cls, list_data: List[float]):
        xmin = list_data[0]
        ymin = list_data[1]
        xmax = list_data[2]
        ymax = list_data[3]
        return cls(min_point=Point(xmin, ymin),
                   max_point=Point(xmax, ymax))

    @classmethod
    def from_list_xymin_wh(cls, list_data: List[float]):
        xmin = list_data[0]
        ymin = list_data[1]
        w = list_data[2]
        h = list_data[3]
        return cls(min_point=Point(xmin, ymin),
                   max_point=Point(xmin + w, ymin + h))

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> BBox:
        return cls(min_point=Point.from_json(json_data["min_point"]),
                   max_point=Point.from_json(json_data["max_point"]))

    @classmethod
    def from_pd_json(cls, json_data: List[float]) -> BBox:
        x = json_data[0]
        y = json_data[1]
        width = json_data[2]
        height = json_data[3]
        return cls(min_point=Point(x, y),
                   max_point=Point(x + width, y + height))

    @classmethod
    def from_underscore_format(cls, str_data: str) -> BBox:
        splits = str_data.split("_")
        assert len(splits) == 4, "Need only 4 numbers separated by underscore !"
        xmin = float(splits[0])
        ymin = float(splits[1])
        xmax = float(splits[2])
        ymax = float(splits[3])
        return cls(Point(xmin, ymin), Point(xmax, ymax))

    def __init__(self, min_point: Point, max_point: Point, ground_point: Optional[Point] = None):
        self._min_point = min_point
        self._max_point = max_point
        self._ground_point = ground_point
        self._parrent = None

    def __ge__(self, bbox: BBox) -> bool:
        return self.area >= bbox.area

    def __lt__(self, bbox: BBox) -> bool:
        return self.area < bbox.area

    def __str__(self):
        return "{} - {}".format(str(self._min_point), str(self._max_point))

    @property
    def min_point(self) -> Point:
        return self._min_point

    @property
    def max_point(self) -> Point:
        return self._max_point

    @property
    def width(self) -> int:
        return self._max_point.int_x - self._min_point.int_x

    @property
    def width_float(self) -> float:
        return self._max_point.x - self._min_point.x

    @property
    def height(self) -> int:
        return self._max_point.int_y - self._min_point.int_y

    @property
    def height_float(self) -> float:
        return self._max_point.y - self._min_point.y

    @property
    def area_float(self) -> float:
        return self.width_float * self.height_float

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def ratio_size(self) -> float:
        return self.width_float / self.height_float

    @property
    def center_point(self) -> Point:
        middle_point_x = (self._min_point.x + self._max_point.x) / 2.0
        middle_point_y = (self._min_point.y + self._max_point.y) / 2.0
        return Point(middle_point_x, middle_point_y)

    @property
    def bottom_middle_point(self) -> Point:
        center_x = self.center_point.int_x
        max_y = self.max_point.int_y
        return Point(center_x, max_y)

    @property
    def ground_point(self) -> Optional[Point]:
        return self._ground_point

    def is_point_in(self, point: Point) -> bool:
        is_in_x = self._min_point.x < point.x < self._max_point.x
        is_in_y = self._min_point.y < point.y < self._max_point.y
        return is_in_x and is_in_y

    def is_in_border(self, bbox: BBox):
        xmin_check = bbox.min_point.x <= self._min_point.x
        ymin_check = bbox.min_point.y <= self._min_point.y
        xmax_check = bbox.max_point.x >= self._max_point.x
        ymax_check = bbox.max_point.y >= self._max_point.y
        return xmin_check and ymin_check and xmax_check and ymax_check


    def set_offset(self, offset: Point):
        self._min_point = Point(self._min_point.x + offset.x, self._min_point.y + offset.y)
        self._max_point = Point(self._max_point.x + offset.x, self._max_point.y + offset.y)

    def set_max_area(self, bbox_area: BBox):
        xmin = max(bbox_area.min_point.x + 1, self._min_point.x)
        ymin = max(bbox_area.min_point.y + 1, self._min_point.y)
        xmax = min(bbox_area.max_point.x - 1, self._max_point.x)
        ymax = min(bbox_area.max_point.y - 1, self._max_point.y)
        self._min_point = Point(xmin, ymin)
        self._max_point = Point(xmax, ymax)

    def set_ratio(self, ratio_x: float, ratio_y: float):
        self._min_point = Point(self._min_point.x * ratio_x, self._min_point.y * ratio_y)
        self._max_point = Point(self._max_point.x * ratio_x, self._max_point.y * ratio_y)

    def diagonal_length(self) -> float:
        return math.sqrt(self.width_float ** 2 + self.height_float ** 2)

    def draw(self, img: np.ndarray,
             color: Tuple[int, int, int] = (255, 0, 0),
             thickness: int = 1,
             txt_label: str = "",
             font_size: float = 1.0
             ) -> np.ndarray:
        img = cv2.rectangle(img, self._min_point.to_coord(), self._max_point.to_coord(), color, thickness)
        if txt_label:
            text_drawer = TextDrawer(font_size)
            text_drawer.add_text(self._max_point, txt_label)
            text_drawer.draw_text(img)
        return img

    def draw_blur(self, img):
        kernel_width = (self.width // 7) | 1
        kernel_height = (self.height // 7) | 1

        h, w, _ = img.shape
        xmin = min(max(0, self._min_point.int_x), w - 1)
        ymin = min(max(0, self._min_point.int_y), h - 1)
        xmax = min(max(0, self._max_point.int_x), w - 1)
        ymax = min(max(0, self._max_point.int_y), h - 1)

        blured = cv2.GaussianBlur(img[ymin: ymax, xmin: xmax], (kernel_width, kernel_height), 0)
        img[ymin: ymax, xmin: xmax] = blured
        return img

    def crop(self, img: np.ndarray, copy_result: bool = True) -> np.ndarray:
        h, w, _ = img.shape
        xmin = min(max(0, self._min_point.int_x), w - 1)
        ymin = min(max(0, self._min_point.int_y), h - 1)
        xmax = min(max(0, self._max_point.int_x), w - 1)
        ymax = min(max(0, self._max_point.int_y), h - 1)

        cropped = img[ymin: ymax, xmin: xmax]
        return cropped.copy() if copy_result else cropped

    def crop_expanded(self, img: np.ndarray, padding_percentage: Optional[float] = None):
        crop_bbox = self
        if padding_percentage is not None:
            expand_x = int(self.width_float * padding_percentage / 2)
            expand_y = int(self.height_float * padding_percentage / 2)

            expanded_bbox = self.expand_bbox_size(expand_x, expand_y)
            crop_bbox = expanded_bbox

        return crop_bbox.crop(img)

    def draw_cross(self, img: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 1) -> np.ndarray:

        img = cv2.line(img, self._min_point.to_coord(), self._max_point.to_coord(), color, thickness)
        return cv2.line(img, Point(self._min_point.x, self._max_point.y).to_coord(),
                        Point(self._max_point.x, self._min_point.y).to_coord(), color, thickness)

    def to_dict(self) -> Dict[str, Any]:
        return {"min_point": self._min_point.to_dict(),
                "max_point": self._max_point.to_dict(),
                "center_point": self.center_point.to_dict()}

    def serialize_pd(self):
        return [self._min_point.int_x, self._min_point.int_y, self.width, self.height]

    def to_cvat_box_dict(self):
        return {
            "xtl": self._min_point.x, "ytl": self._min_point.y,
            "xbr": self._max_point.x, "ybr": self._max_point.y,
        }

    def to_darknet_label(self, img_w: int, img_h: int) -> str:
        return "{} {} {} {}".format(self.center_point.x / float(img_w), self.center_point.y / float(img_h),
                                    self.width_float / float(img_w), self.height_float / float(img_h))

    def to_numpy_points(self) -> np.ndarray:
        return np.array([[self._min_point.x, self._min_point.y],
                         [self._max_point.x, self._min_point.y],
                         [self._max_point.x, self._max_point.y],
                         [self._min_point.x, self._max_point.y]], dtype=np.float32)

    def to_min_max_points_float(self) -> List[float]:
        return [self._min_point.x, self._min_point.y, self._max_point.x, self._max_point.y]

    def to_underscore_format(self) -> str:
        return "{}_{}".format(self._min_point.to_underscore_format(), self._max_point.to_underscore_format())

    def to_cvrect_format_list(self) -> List[int]:
        return [self._min_point.int_x, self._min_point.int_y, self.width, self.height]

    def to_xymin_wh_list(self) -> List[float]:
        l = [self._min_point.x, self._min_point.y, self.width_float, self.height_float]
        return list(map(lambda x: round(x, 2), l))

    def to_sort_z(self) -> np.ndarray:
        """
            Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r]
            where x,y is the centre of the box and s is the scale/area and r is the aspect ratio
        """
        return np.array([self.center_point.x, self.center_point.y, self.area_float, self.ratio_size]).reshape((4, 1))

    def to_sort_scored_z(self, score: float) -> List[float]:
        return [self.center_point.x, self.center_point.y, self.area_float, self.ratio_size, score]

    def to_sort_bbox(self, score: float = 1.0):
        return self.to_min_max_points_float() + [score]

    def repair_by_image_size(self, img_w: int, img_h: int):
        xmin = min(max(0.0, self._min_point.x), img_w - 1)
        ymin = min(max(0.0, self._min_point.y), img_h - 1)
        xmax = min(max(0.0, self._max_point.x), img_w - 1)
        ymax = min(max(0.0, self._max_point.y), img_h - 1)
        return BBox(Point(xmin, ymin), Point(xmax, ymax))

    def expand_bbox_size(self, px_size_x: int, px_size_y: int) -> "BBox":
        xmin = max(0.0, self._min_point.x - px_size_x)
        ymin = max(0.0, self._min_point.y - px_size_y)
        xmax = max(0.0, self._max_point.x + px_size_x)
        ymax = max(0.0, self._max_point.y + px_size_y)
        return BBox(Point(xmin, ymin), Point(xmax, ymax))
