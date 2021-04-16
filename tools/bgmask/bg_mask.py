import numpy as np
import cv2
import time
from keras.models import load_model
import os
import threading
import setproctitle
import sys

class BackgroundMask:

    def __init__(self, model="models/segmentation/deconv_bnoptimized_munet.h5", bg_im_path="bg_im/",
                 frame_height=640, frame_width=480):
        # load model
        try:
            self.model = load_model(model, compile=False)
        except:
            raise Exception(f"Couldn't load the model\nGiven: {model}")

        # Target size
        self.tgt_size=640
        self.frame_height= frame_height
        self.frame_width= frame_width
        self.bg_path = bg_im_path
        self.bg_list = self.get_bg(self.bg_path)
        self.change_bgd()

        self.time = 0
        self.frame = None
        self.started = None
        self.bucket = None
        self.thread = None

    def get_bg(self, folder):
        """
        Function that list all jpg files

        :param str, folder path
        :return: file_list
        """

        files = os.listdir(folder)
        file_list = [file for file in files if file.endswith('jpg')]
        return file_list

    def change_bgd(self, idx=0):
        """
        Function that choose the background by index of self.file_list

        :param idx: int, valid
        """

        if idx <= len(self.bg_list):
            bg_location = os.path.join(self.bg_path, (self.bg_list[idx]))
            self.bg = cv2.resize(cv2.imread(bg_location), (self.tgt_size,self.tgt_size))
            self.bg = self.cvtColor(self.bg)
        else:
            print(f"Non valid bg_img idx.\nGiven {idx}.\nLen list {len(self.bg_list)} ")

    def cvtColor(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
        return img

    def start(self):
        if self.started:
            print("BG-mask thread already started.")
        else:
            self.started = True
            self.thread = threading.Thread(target=self.runTime, name='BgMaskThread', daemon=True)
            self.thread.start()

    def stop(self):
        if not self.started:
            print("BG-mask thread already stopped.")
        else:
            self.started = False

    def runTime(self):
        setproctitle.setproctitle(threading.currentThread().name)
        while self.started:
            # Check if frame is op type <class 'numpy.ndarray'>
            if isinstance(self.frame, np.ndarray):
                startTime = time.time()

                # Pre-process
                img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                bg = cv2.resize(self.bg, (self.tgt_size, self.tgt_size))

                simg = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
                simg = simg.reshape((1,128,128,3))/255.0

                # Predict
                out= self.model.predict(simg)
                
                orimsk=np.float32((out>0.5)).reshape((128,128,1))

                # Post-process
                msk=cv2.GaussianBlur(orimsk,(5,5),1)
                img=cv2.resize(img, (self.tgt_size,self.tgt_size), interpolation=cv2.INTER_AREA)/255.0
                msk=cv2.resize(msk, (self.tgt_size,self.tgt_size)).reshape((self.tgt_size,self.tgt_size,1))
                # Alpha blending
                frame = (img * msk) + (bg * (1 - msk))
                frame = np.uint8(frame*255.0)
                mask = np.uint8(msk*255.0)

                # Resize
                frame = cv2.resize(frame, (self.frame_height, self.frame_width), interpolation = cv2.INTER_AREA)

                # Drop result in the bucket
                self.bucket = frame[...,::-1]
                self.time = round((time.time() - startTime)*1000)


class Capture:
        def __init__(self):
            self.cap = cv2.VideoCapture(0)
            self.started = False
            self.bucket = None
            self.thread = None

        def start(self):
            if self.started:
                print(f"{ Capture.__qualname__} thread already started.")
            else:
                self.started = True
                self.thread = threading.Thread(target=self.runTime, name='CaptureThread', daemon=True)
                self.thread.start()

        def stop(self):
            if not self.started:
                print("Capture thread already stopped.")
            else:
                self.started = False

        def runTime(self):
            setproctitle.setproctitle(threading.currentThread().name)
            while self.started:
                succes, frame = self.cap.read()
                if succes:
                    self.bucket = frame

if __name__ == "__main__":
    bgMask = BackgroundMask()
    bgMask.start()
    capture = Capture()
    capture.start()
    running = True
    while running:
        if isinstance(capture.bucket, np.ndarray):
            bgMask.frame = capture.bucket

        if isinstance(bgMask.bucket, np.ndarray):
            cv2.imshow('Bgmask', bgMask.bucket)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()
