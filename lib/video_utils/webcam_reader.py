from collections import deque
import cv2


class FpsWrapper(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._diff_times = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._diff_times.append(different_time)

        fps = 1000.0 / (sum(self._diff_times) / len(self._diff_times))
        fps_rounded = round(fps, 2)

        return fps_rounded


class WebcamReader(object):
    def __init__(self, screen_width: int, screen_height: int, webcam_id: int = 0, flip_screen: bool = True,
                 swap_rb: bool = False, debug_text: bool = False):
        self._debug_text = debug_text
        self._swap_rb = swap_rb
        self._flip_screen = flip_screen

        self._cap = cv2.VideoCapture(webcam_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

        self._screen_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._screen_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._processed_frames = 0
        self._fps_wrapper = FpsWrapper(buffer_len=20)
        self._curr_fps = 0

    @property
    def screen_width(self):
        return self._screen_width

    @property
    def screen_height(self):
        return self._screen_height

    @property
    def fps(self):
        return self._curr_fps

    def __iter__(self):
        return self

    def __next__(self):
        self._curr_fps = self._fps_wrapper.get()
        return self.get_next_frame()

    def get_next_frame(self):
        ret, frame = self._cap.read()

        if self._flip_screen:
            frame = cv2.flip(frame, 1)

        if self._swap_rb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._debug_text:
            debug_str = "h: {}, w: {}, FPS: {}".format(self._screen_height, self._screen_width, self._curr_fps)
            cv2.putText(frame, debug_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)

        self._processed_frames += 1
        return ret, frame

# if __name__ == '__main__':
#     webcam_reader = WebcamReader(screen_width=960, screen_height=540, flip_screen=True, swap_rb=True, debug_text=True)
#     h = webcam_reader.screen_height
#     w = webcam_reader.screen_width
#
#     for _, frame in webcam_reader:
#         cv2.imshow("aa", frame)
#         cv2.waitKey(1)
