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


class LoadFeatures:
    def __init__(self, features=None, cam_width=640, cam_height=480):

        # self.feature_list = ["bgMask", "stt", "visionMd", "vidGet", "vidShow"]
        # self.features_selected = [x for x in features if x in self.feature_list]
        # Load all features
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
        self.result_frame = None

        self.result_queue = queue.Queue()

        self.lock = Lock()

        self.last_time_action = 0
        self.actionHandler_delay= 1/20 # 20 frames per second

    def actionHandler(self):
        black_bg = False
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
                                print(result["mood"])
                            elif "stt" in key:
                                tmp = result['stt']
                                if "background" in tmp:
                                    self.bgMask.change_bgd(random.randint(0,4))
                                elif "black" in tmp:
                                    black_bg = True
                                elif "stop" in tmp:
                                    black_bg = False

                                print(result["stt"])
                            elif "bg" in key:
                                if not black_bg:
                                    frame = result["bg"]
                                    self.result_frame = frame
                                else:
                                    self.result_frame = np.zeros(shape=(self.cam_height, self.cam_width, 3))

                            elif "gesture" in key:
                                print(result["gesture"])
                self.last_time_action = t
            time.sleep(0.01)

    def start(self, start=True):
        if self.started and start:
            print("Already started")
        else:
            self.started = start
            self.runtime()

    def stop(self):
        self.started = False
        if self.debug:
            print("Stop")
        sys.exit()

    def videoCap(self):
        cap = cv2.VideoCapture(0)
        while self.started:
            self.ret, self.frame = cap.read()

    def runtime(self):
        # Init a thread for the Speech to Text service, and pass the queue.
        self.stt_thread = Thread(target=self.stt.runTime, args=(self.result_queue, self.lock,), daemon=True)
        self.stt_thread.start()

        # Init a thread for the VideoCapture Service.
        self.videoCap_thread = Thread(target=self.videoCap, daemon=True)
        self.videoCap_thread.start()

        self.action_thread = Thread(target=self.actionHandler, daemon=True)
        self.action_thread.start()

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
                    self.result_queue.put({"gesture" : self.gesture.bucket})
                self.gesture.bucket = None                # Resize the frame

            if isinstance(self.bgMask.bucket, ndarray):
                self.tmp = cv2.resize(self.bgMask.bucket, (self.cam_width, self.cam_height), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.result_queue.put({"bg" : self.tmp})
                self.bgMask.bucket = None

            stopTime = time.time()
            fps = round(1/(stopTime - startTime), 1)

            # Print the feature timings,
            # 1, Total frames per second
            # 2, bgMask time
            # 3, vision Mood time
            # 4, Gesture detection time
            if self.debug:
                print(fps, self.bgMask.time, self.visionMd.time, self.gesture.time)

            # Set timings to zero
            self.bgMask.time = 0
            self.visionMd.time = 0
            self.gesture.time = 0

            if isinstance(self.result_frame, ndarray):
                cv2.imshow('frame', self.result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()


if __name__ == "__main__":
    asap = LoadFeatures()
    asap.start()
    while True:
        pass
