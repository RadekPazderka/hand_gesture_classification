from collections import deque

from lib.collections.base_collection import BaseCollection
from lib.collections.base_prop import BaseProp


class HandControl(object):
    def __init__(self, collection: BaseCollection):
        self._collection = collection
        self._history = deque(maxlen=2)
        self._history.append(0)
        self._last_state = None
        self._changes = []

    def change_state(self, cls_id: int):
        if len(self._changes) == 0 and self._last_state != cls_id:
            self._changes.insert(0, cls_id)
            self._last_state = cls_id
            return

        # if cls_id != self._changes[0]:
        #     self._changes.insert(0, cls_id)
        #     return

    def get_command(self):
        if len(self._changes) == 0:
            return None
        return self._changes.pop()

    def add_classification(self, cls: BaseProp):
        """
        Open: 0
        Close: 1
        Pointer: 2
        OK: 3
        """
        self._history.append(cls.LABEL_ID)
        if len(set(self._history)) == 1 and len(self._history) == self._history.maxlen:
            self.change_state(self._history[0])
