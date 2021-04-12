import cv2
import time
from threading import Thread
import numpy as np
import sys


class VideoGetter:

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

        # Check if cap not already is opened, otherwise raise an error.
        if not self.cap:
            raise Exception(f"Could not open camera.\nGiven: camera {source}")

        self.started = False
        self.time = 0
        self.bucket = None
        self.thread = None

    def start(self):
        if self.started:
            print('Video get frame thread already started')
        else:
            self.started = True
            self.thread = Thread(target=self.runTime, name='VideoGetFrameThread', daemon=True)
            self.thread.start()

    def stop(self):
        if not self.started:
            print('VideoGetFrame thread already stopped')
        else:
            self.started = False
            self.cap.release()

    def runTime(self):
        while self.started:
            startTime = time.time()
            ret, frame = self.cap.read()
            if ret:
                self.bucket = frame
            self.time = round((time.time() - startTime) * 1000)


class VideoShower:

    def __init__(self, exitKey='q'):
        self.frame = None
        self.time = 0
        self.started = False
        self.exitKey = exitKey

    def start(self):
        if self.started:
            print('VideoShowFrame thread already started')
        else:
            self.started = True
            self.thread = Thread(target=self.runTime, name='VideoShowThread', daemon=True)
            self.thread.start()

    def stop(self):
        if not self.started:
            print('VideoGetFrame thread already stopped')
        else:
            self.started = False
            cv2.destroyAllWindows()

    def runTime(self):
        while self.started:
            if isinstance(self.frame, np.ndarray):
                startTime = time.time()
                cv2.imshow("Frame", self.frame)
                self.time = round((time.time() - startTime) * 1000)

                if cv2.waitKey(1) & 0xFF == ord(self.exitKey):
                    self.stop()


if __name__ == "__main__":
    vget = VideoGetter()
    vshow = VideoShower()
    vget.start()
    vshow.start()
    startTime = time.time()
    while True:
        if isinstance(vget.frame, np.ndarray):
            vshow.frame = vget.frame
            vget.frame = None

        if (time.time() - startTime) >= 3:
            print('Time to exit')
            vshow.stop()
            vget.stop()
            sys.exit()
