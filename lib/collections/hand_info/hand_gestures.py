from lib.collections.base_collection import BaseCollection
from lib.collections.base_prop import BaseProp


class HandGesturesBase(BaseProp):

    def serialize(self):
        return {"name": self.NAME,
                "label_id": self.LABEL_ID}


class OpenHandGesture(HandGesturesBase):
    NAME = "open"
    LABEL_ID = 0
    KEYWORDS = [NAME, "otevrena"]
    COLOR = (0, 0, 255)


class CloseHandGesture(HandGesturesBase):
    NAME = "close"
    LABEL_ID = 1
    KEYWORDS = [NAME, "zavrena"]
    COLOR = (255, 0, 0)


class PointerHandGesture(HandGesturesBase):
    NAME = "pointer"
    LABEL_ID = 2
    KEYWORDS = [NAME]
    COLOR = (0, 255, 0)


class OkHandGesture(HandGesturesBase):
    NAME = "ok_gesture"
    LABEL_ID = 3
    KEYWORDS = [NAME]
    COLOR = (127, 127, 0)


class UnknownHandGesture(HandGesturesBase):
    NAME = "unknown_hand_gesture"
    LABEL_ID = -1
    KEYWORDS = [NAME, "unknown"]
    COLOR = (0, 0, 0)


class HandGestureCollection(BaseCollection):
    NAME = "hand_gesture_collection"
    PROPS = [
        OpenHandGesture,
        CloseHandGesture,
        PointerHandGesture,
        OkHandGesture
    ]
    UNKNOWN_PROP = UnknownHandGesture
