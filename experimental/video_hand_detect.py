from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from data_structure.collections.hand_info.hand_gestures import HandGestureCollection
from data_structure.coordinates.detection_general import DetectionGeneral
from projects_src.hand_detection.keypoints_classifier_onnx import KeypointsClassifierOnnx
from projects_src.hand_detection.mediapipe_wrapper import MediapipeWrapper
from video_utils.video_reader import VideoReader
from video_utils.video_writer import VideoWriter


def cv2_img_add_text(img, text, left_corner: Tuple[int, int],
                     text_rgb_color=(255, 0, 0), text_size=46, font='mingliu.ttc', **option):
    """
    USAGE:
        cv2_img_add_text(img, '中文', (0, 0), text_rgb_color=(0, 255, 0), text_size=12, font='mingliu.ttc')
    """
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype(font="arial.ttf", size=text_size, encoding=option.get('encoding', 'utf-8'))
    draw.text(left_corner, text, text_rgb_color, font=font_text, stroke_width=2, stroke_fill="black")
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img


def main():
    video_path = r"C:\Users\radek\Videos\20220729_204532.mp4"
    video_reader = VideoReader(video_path)
    video_writer = VideoWriter("out_3.mp4", fps=30)

    mediapipe_wrapper = MediapipeWrapper()

    onnx_model_path = "hand_models/keypoint_classifier.onnx"
    collection = HandGestureCollection()

    keypoints_classifier = KeypointsClassifierOnnx(onnx_model_path, collection)

    video_reader.iterate_to_frame(266)
    for frame_id, frame in video_reader.get_iterator(False):
        hands = mediapipe_wrapper.process(frame)

        image = frame
        for hand_keypoints in hands:

            # Hand sign classification
            hand_sign_cls = keypoints_classifier.classify_list_keypoints(hand_keypoints)

            bbox = hand_keypoints.to_bbox()
            bbox = bbox.expand_bbox_size(20, 20)

            det = DetectionGeneral(bbox, hand_sign_cls)
            coords = bbox.min_point.to_coord()

            hand_keypoints.draw(image)
            image = cv2_img_add_text(image, hand_sign_cls.NAME, (coords[0], coords[1] - 46), text_rgb_color=hand_sign_cls.COLOR)
            det.draw(image, overide_color=(0, 255, 0), thickness=3, draw_text_label=False)

        video_writer.add_frame(image)
        cv2.imshow("test", image)
        cv2.waitKey()
    video_writer.release()


if __name__ == '__main__':
    main()
