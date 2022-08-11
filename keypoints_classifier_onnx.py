import cv2
import numpy as np

from lib.collections.base_collection import BaseCollection
from lib.collections.base_prop import BaseProp
from lib.coordinates.keypoints.hand_keypoints import HandKeypoints


class KeypointsClassifierOnnx(object):
    def __init__(self, onnx_path: str, collection: BaseCollection):
        self._onnx_net = cv2.dnn.readNetFromONNX(onnx_path)
        self._collection = collection

    def classify_list_keypoints(self, hand_keypoints: HandKeypoints) -> BaseProp:
        keypoints_data = hand_keypoints.to_normalized_list_xy()
        input_data = np.array([keypoints_data], dtype=np.float32)
        self._onnx_net.setInput(input_data)
        pred = self._onnx_net.forward()
        index = int(np.argmax(np.squeeze(pred)))
        return self._collection.get_by_id(index)
