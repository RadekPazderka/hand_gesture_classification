from queue import Queue

from chrome_t_rex_rush.t_rex_main import introscreen, gameplay
from hand_app import HandApp


def main():
    q = Queue()
    onnx_model_path = "hand_models/keypoint_classifier.onnx"
    HandApp.from_default(q, onnx_model_path).run()

    isGameQuit = introscreen()
    if not isGameQuit:
        gameplay(q)

main()