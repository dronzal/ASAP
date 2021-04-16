import pyvirtualcam as pvc
import threading
import numpy as np
from time import time
import setproctitle
import cv2


class VirtualDevice:
    def __init__(self,frame_width=640, frame_height=480, fps=20):
        self.width = frame_width
        self.height = frame_height
        self.fps = fps
        self.cam = None
        self.frame = None
        self.started = False
        self.time = 0

    def start(self):
        if self.started:
            print("VirtualDevice runTime is already started")
        else:
            self.started = True
            self.thread = threading.Thread(target=self.runTime, name='VirtualCamThread', daemon=True)
            self.thread.start()

    def stop(self):
        if not self.started:
            print("VirtualDevice runTime is already stopped")
        else:
            self.started = False

    def runTime(self):
        setproctitle.setproctitle(threading.currentThread().name)
        with pvc.Camera(self.width, self.height, self.fps) as self.cam:
            while self.started:
                if isinstance(self.frame, np.ndarray):
                    self.frame = cv2.resize(self.frame, (self.width, self.height), interpolation = cv2.INTER_AREA)
                    self.cam.send(self.frame)
                    self.cam.sleep_until_next_frame()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pvc.Camera.close()
