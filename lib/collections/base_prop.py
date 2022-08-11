from typing import Type

try:
    from abc import ABC, abstractmethod
except:
    from abc import ABCMeta as ABC, abstractmethod


class BaseProp(ABC):
    NAME = ""
    KEYWORDS = []
    LABEL_ID = -1
    COLOR = (0, 255, 0)

    def __init__(self, precision=1.0, *args, **kwargs):
        self._precision = precision

    def __repr__(self):
        return "{} ({:.1f}%)".format(self.NAME, self._precision * 100.)

    def __str__(self):
        return "{} ({:.1f}%)".format(self.NAME, self._precision * 100.)

    def __eq__(self, other):
        # type: ("BaseProp") -> bool

        # e.g: test = Car() in [Van(), Pedestrian(), Car()]
        #      test = Car() == Pedestrian()
        return self.LABEL_ID == other.LABEL_ID and self.NAME == other.NAME

    @property
    def precision(self):
        return self._precision

    @abstractmethod
    def serialize(self):
        pass

    def norm_color(self):
        return tuple(map(lambda x: x / 255., self.COLOR))


CT_BaseProp = Type[BaseProp]  # Class type of Baseprop
