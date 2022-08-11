from queue import Queue
import cv2

from lib.collections.hand_info.hand_gestures import HandGestureCollection
from lib.decorators.async_func import run_async_thread
from lib.video_utils.webcam_reader import WebcamReader
from hand_control import HandControl
from keypoints_classifier_onnx import KeypointsClassifierOnnx
from mediapipe_wrapper import MediapipeWrapper


class HandApp(object):

    @classmethod
    def from_default(cls, q: Queue, onnx_model_path: str):
        webcam_reader = WebcamReader(screen_width=640, screen_height=480, flip_screen=False, swap_rb=True,
                                     debug_text=True)
        mediapipe_wrapper = MediapipeWrapper()
        collection = HandGestureCollection()
        keypoints_classifier = KeypointsClassifierOnnx(onnx_model_path, collection)
        hand_control = HandControl(collection)
        return cls(q, webcam_reader, mediapipe_wrapper, keypoints_classifier, hand_control)

    def __init__(self, queue: Queue,
                 webcam_reader: WebcamReader,
                 mediapipe_wrapper: MediapipeWrapper,
                 keypoints_classifier: KeypointsClassifierOnnx,
                 hand_control: HandControl):
        self._queue = queue
        self._webcam_reader = webcam_reader
        self._mediapipe_wrapper = mediapipe_wrapper
        self._keypoints_classifier = keypoints_classifier
        self._hand_control = hand_control

    @run_async_thread
    def run(self):

        for _, frame_rgb in self._webcam_reader:
            hands = self._mediapipe_wrapper.process(frame_rgb)

            for hand_keypoints in hands:
                hand_keypoints.draw(frame_rgb)

                # Hand sign classification
                hand_sign_cls = self._keypoints_classifier.classify_list_keypoints(hand_keypoints)
                self._hand_control.add_classification(hand_sign_cls)

            command = self._hand_control.get_command()
            if command is not None:
                print("Change state to: {}".format(command))
                self._queue.put(command)

            cv2.imshow("hand control window", frame_rgb[:, :, ::-1])
            cv2.waitKey(1)
