from tools import speechToText as STT
from tools.bgmask import bg_mask as bgm
from tools.vision_mood_detection import mood_detection as MD
from tools import gesture_detection
import cv2
import time
import concurrent.futures
import queue
from threading import Thread
from numpy import ndarray
import pyvirtualcam
import random
import sys

class LoadFeatures:

    def __init__(self, features=None, cam_width=640, cam_height=480):

        # self.feature_list = ["bgMask", "stt", "visionMd", "vidGet", "vidShow"]
        # self.features_selected = [x for x in features if x in self.feature_list]
        # Load all features
        self.stt = STT.SpeechToText()
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

        self.debug = True

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
        self.p_stt.terminate()
        sys.exit()

    def videoCap(self):
        cap = cv2.VideoCapture(0)
        while self.started:
            self.ret, self.frame = cap.read()

    def runtime(self):
        # Init a queue for the Speech to text service
        q_stt = queue.Queue()

        # Init a thread for the Speech to Text service, and pass the queue.
        self.stt_thread = Thread(target=self.stt.runTime, args=(q_stt,), daemon=True)
        self.stt_thread.start()

        # Init a thread for the VideoCapture Service.
        self.videoCap_thread = Thread(target=self.videoCap, daemon=True)
        self.videoCap_thread.start()

        # Uncomment if you want to use virtualCam
        # virual_cam = pyvirtualcam.Camera(width=self.cam_width, height=self.cam_height, fps=20)
        if self.debug:
            print('Start ASAP')

        while self.started:

            # Get time, this for calculating the total frames per second.
            startTime = time.time()

            # If self.frame is ndarray, go further.
            if isinstance(self.frame, ndarray):

                # Init an async threadPool and wait if the childThreads are finished to go further.
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    executor.submit(self.bgMask.runTime, self.frame)
                    executor.submit(self.visionMd.runTime, self.frame)
                    executor.submit(self.gesture.runtime, self.frame)

                # Chech if a result is stored in the speech to text queue
                if not q_stt.empty():
                    self.stt_result = str(q_stt.get()).lower()
                    print(self.stt_result)

                # Some actions.
                # In this case, if black is heared, self.black_bg is setted True
                if "black" in self.stt_result:
                    self.black_bg = True
                    self.stt_result = ""
                # If stop is detected, self.black_bg is False
                if "stop" in self.stt_result:
                    self.black_bg = False
                    self.stt_result = ""
                if not isinstance(self.visionMd.bucket, type(None)):
                    print(self.visionMd.bucket)
                    self.visionMd.bucket = None

                if not isinstance(self.gesture.bucket, type(None)):
                    print(self.gesture.bucket)
                    self.gesture.bucket = None                # Resize the frame

                self.result_frame = cv2.resize(self.bgMask.bucket, (self.cam_width, self.cam_height), interpolation=cv2.INTER_AREA)

                # If black_bg is True, fill the background with zero's = black
                if self.black_bg:
                    self.result_frame.fill(0)

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

                # Show frame
                cv2.imshow('frame', self.result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

                # Send frame to virtualCam
                # Uncomment if you want to use virtualCam

                # virual_cam.send(result_frame)

                # Change background
                if self.counter % 10 == 0:
                    self.bgMask.change_bgd(random.randint(0,4))

                # Counter, just for showing the change_bg feature
                self.counter += 1


if __name__ == "__main__":
    asap = LoadFeatures()
    asap.start()
