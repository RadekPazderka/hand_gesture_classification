import math
import cv2
import numpy as np
from typing import Tuple, Dict, List, Union


class Point(object):

    @classmethod
    def from_ndarray(cls, data: np.ndarray):
        dims = len(data.shape)
        assert dims == 1 and data.shape[0] == 2, "Point shape: (2,) not {}".format(data.shape)
        return cls(float(data[0]), float(data[1]))

    @classmethod
    def from_json(cls, json_data):
        # type: (Dict[str, float]) -> Point
        return cls(x=json_data["x"], y=json_data["y"])

    @classmethod
    def from_tuple(cls, data):
        # type: (Tuple[Union[float, int, str], Union[float, int, str]]) -> Point
        return cls(float(data[0]), float(data[1]))

    def __init__(self, x, y):
        # type: (float, float) -> None
        self._x = float(x)
        self._y = float(y)

    def __eq__(self, point):
        # type: (Point) -> bool
        return (self.int_x == point.int_x) and (self.int_y == point.int_y)

    def __str__(self):
        return "({}, {})".format(self._x, self._y)

    def __repr__(self):
        return "({}, {})".format(self._x, self._y)

    def distance(self, p2: "Point") -> float:
        return math.sqrt(((self.x - p2.x) ** 2) + ((self.y - p2.y) ** 2))

    @property
    def x(self):
        # type: () -> float
        return self._x

    @property
    def y(self):
        # type: () -> float
        return self._y

    @property
    def int_x(self):
        # type: () -> int
        return int(self._x)

    @property
    def int_y(self) -> int:
        return int(self._y)

    def get_inv_point(self):
        return Point(-self._x, -self._y)

    def get_random_normal_change_x(self, sigma=1.0):
        return np.random.normal(self._x, sigma)

    def get_random_normal_change_y(self, sigma=1.0):
        return np.random.normal(self._y, sigma)

    def to_coord(self) -> Tuple[int, int]:
        return (self.int_x, self.int_y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self._x, self._y)

    def to_list(self) -> List[float]:
        return [self._x, self._y]

    def to_list_int(self) -> List[int]:
        return [self.int_x, self.int_y]

    def to_underscore_format(self) -> str:
        return "{}_{}".format(self.int_x, self.int_y)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self._x, "y": self._y}

    def draw(self, img: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), radius: int = 5, thickness: int = 2,
             filled: bool = False) -> np.ndarray:
        if filled:
            thickness = -1
        return cv2.circle(img, self.to_coord(), radius, color, thickness)


class PointUtils(object):

    @staticmethod
    def euclidian_distance(point_1, point_2):
        # type: (Point, Point) -> float
        return math.sqrt(((point_1.x - point_2.x) ** 2) + ((point_1.y - point_2.y) ** 2))

# class PointBuilder():
#     @staticmethod
#     def build(json_data):
#         # type: (Dict[str, float]) -> Point
#         return Point(x=json_data["x"], y=json_data["y"])
