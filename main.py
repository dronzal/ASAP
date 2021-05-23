from tools import speechToText as STT
from tools.bgmask import bg_mask as bgm
from tools.vision_mood_detection import mood_detection as MD
from tools import gesture_detection
import cv2
import time
import concurrent.futures
from threading import Thread, Lock
from numpy import ndarray
import numpy as np
import pyvirtualcam
import random
import sys
import queue
from collections import Counter, deque

class ASAP:
    def __init__(self, cam_width=640, cam_height=480):
        self.stt = STT.SpeechToText(google_credentials_file="/home/puyar/Documents/Playroom/asap-309508-7398a8c4473f.json")
        self.bgMask = bgm.BackgroundMask()
        self.visionMd = MD.MoodDetection()
        self.gesture = gesture_detection.GestureDetection()

        self.cam_width = cam_width
        self.cam_height = cam_height

        # Bool for continuous runTime
        self.started = False
        self.cap = None
        self.frame = None
        self.counter = 0

        self.stt_result = ""
        self.black_bg = False

        self.debug = False
        self.result_frame = np.zeros(shape=(self.cam_height, self.cam_width, 3))
        self.bgMask_frame = None
        self.gesture_result = None
        self.timings = None

        self.result_queue = queue.Queue()
        self.mood_deque = deque(maxlen=10)

        self.lock = Lock()

        self.last_time_action = 0
        self.actionHandler_delay = 1/20 # 20 frames per second

        self.while_delay = 0.01

        # Actions
        self.black_bg = False
        self.show_gesture_debug = False

    def virtualCam(self):
        """
        VirtualCam function
        :return:
        """
        with pyvirtualcam.Camera(width=self.cam_width, height=self.cam_height, fps=20) as cam:
            while self.started:
                cam.send(self.result_frame)
                cam.sleep_until_next_frame()

    def actionHandler(self):
        """
        ActionHandler function.
        Always run in a thread.
        :return:
        """
        while self.started:
            t = time.time()
            if (t - self.last_time_action) >= self.actionHandler_delay:
                if not self.result_queue.empty():
                    with self.lock:
                        length_queue = len(list(self.result_queue.queue))
                        for x in range(length_queue):
                            result = self.result_queue.get()
                            key = dict(result).keys()
                            # possible actions:
                            # mood, stt, bg, gesture
                            if "mood" in key: # if mood is in result
                                self.mood_deque.append(result['mood'])

                            elif "stt" in key:
                                tmp = result['stt']
                                if "background" in tmp:
                                    self.bgMask.change_bgd(random.randint(0,4))
                                elif "black" in tmp:
                                    self.black_bg = True
                                elif "gesture" in tmp:
                                    self.show_gesture_debug = True
                                elif "stop" in tmp:
                                    self.black_bg = False
                                    self.show_gesture_debug = False

                            elif "bg" in key:
                                self.bgMask_frame = self.flip_frame(result["bg"])

                            elif "gesture" in key:
                                self.gesture_result = result["gesture"]

                        self.build_end_frame()

                self.last_time_action = t
            time.sleep(self.while_delay)

    @staticmethod
    def flip_frame(frame):
        """
        Function that flip the img by verctical axis
        :param frame: ndarray
        :return: flipped img
        """
        return cv2.flip(frame, 1)

    def build_end_frame(self):
        """
        Function that builds the end frame
        :return:
        """
        if self.black_bg:
            self.result_frame = np.zeros(shape=(self.cam_height, self.cam_width, 3))

        elif self.show_gesture_debug:
            if isinstance(self.gesture.debug_frame, ndarray):
                self.result_frame = self.gesture.debug_frame
                self.gesture.debug_frame = None
            else:
                self.result_frame = self.bgMask_frame
        else:
            self.result_frame = self.bgMask_frame

    def start(self, start=True):
        """
        Function that starts the runtime
        :param start: Boolean, std True
        :return:
        """
        if self.started and start:
            print("Already started")
        else:
            self.started = start
            self.runtime()

    def stop(self):
        """
        Functions that stops en close the application
        :return:
        """
        self.started = False
        if self.debug:
            print("Stop")
        sys.exit()

    def videoCap(self):
        """
        Function that store a img frame in self.frame captured by device 0
        Always run this in a thread.
        :return:
        """
        cap = cv2.VideoCapture(0)
        while self.started:
            self.ret, self.frame = cap.read()

    def videoShow(self):
        """
        Function that shows the frame in cv2 GUI.
        Always run this in a thread.
        :return:
        """
        while self.started:
            if isinstance(self.result_frame, ndarray):
                cv2.imshow('frame', self.result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                self.result_frame = None
            time.sleep(self.while_delay)

    @property
    def bg_mask(self):
        """
        Returns the finale result frame
        :return: ndarray
        """
        return self.result_frame

    @property
    def gesture_frame(self):
        """
        Returns the gesture debug frame
        :return: ndarray
        """
        return self.gesture.debug_frame

    @property
    def get_mood(self):
        """
        Returns the detected mood
        :return: string
        """
        return self.mood

    @property
    def get_timings(self):
        """
        Returns the timings that each feature need
        :return: dict
        """
        return self.timings

    @property
    def get_stt(self):
        """
        Returns the stt text
        :return: string
        """
        return self.stt.bucket

    @property
    def get_initial_frame(self):
        """
        Returns the initial frame captured by camera
        :return: ndarray
        """
        return self.frame

    def mood(self, mostCommon=True):
        """
         This functions sort the objects in self.mood_queue by number occurs
         and returns the most occurs element string name if mostCommon = True.
         Otherwise the function return the full list.
        :return: str
        """
        li = list(self.mood_deque)
        if len(li):
            if mostCommon:
                return Counter(li).most_common(1)[0][0]
            else:
                return Counter(li)
        else:
            return None

    def runtime(self):
        """
        ASAP runtime. Runs forever if self.started is True.
        :return:
        """
        # Init a thread for the Speech to Text service, and pass the queue.
        self.stt_thread = Thread(target=self.stt.runTime, args=(self.result_queue, self.lock,), daemon=True)
        self.stt_thread.start()

        # Init a thread for the VideoCapture Service.
        self.videoCap_thread = Thread(target=self.videoCap, daemon=True)
        self.videoCap_thread.start()

        self.action_thread = Thread(target=self.actionHandler, daemon=True)
        self.action_thread.start()

        self.videoShow_thread = Thread(target=self.videoShow, daemon=True)
        self.videoShow_thread.start()

        self.virtualCam_thread = Thread(target=self.virtualCam, daemon=True)
        self.virtualCam_thread.start()

        # Uncomment if you want to use virtualCam
        # virtual_cam = pyvirtualcam.Camera(width=self.cam_width, height=self.cam_height, fps=20)
        if self.debug:
            print('Start ASAP')

        while self.started:

            # Get time, this for calculating the total frames per second.
            startTime = time.time()

            # Init an async threadPool and wait if the childThreads are finished to go further.
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.bgMask.runTime, self.frame)
                executor.submit(self.visionMd.runTime, self.frame)
                executor.submit(self.gesture.runtime, self.frame)

            if not isinstance(self.visionMd.bucket, type(None)):
                with self.lock:
                    self.result_queue.put({"mood": self.visionMd.bucket})
                self.visionMd.bucket = None

            if not isinstance(self.gesture.bucket, type(None)):
                with self.lock:
                    self.result_queue.put({"gesture": self.gesture.bucket})
                self.gesture.bucket = None

            if isinstance(self.bgMask.bucket, ndarray):
                tmp = cv2.resize(self.bgMask.bucket, (self.cam_width, self.cam_height), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.result_queue.put({"bg" : tmp})
                self.bgMask.bucket = None

            stopTime = time.time()

            try:
                fps = round(1/(stopTime - startTime), 1)
            except ZeroDivisionError:
                fps = 0

            self.timings = {"fps" : fps,
                            "bgMask" :self.bgMask.time,
                            "visionMd" :self.visionMd.time,
                            "gesture" : self.gesture.time,}

            # Set timings to zero
            self.bgMask.time = 0
            self.visionMd.time = 0
            self.gesture.time = 0


if __name__ == "__main__":
    asap = ASAP()
    asap_thread = Thread(target=asap.start, daemon=True)
    asap_thread.start()
    while asap.started:
        print(asap.mood())
        time.sleep(0.2)
        pass
