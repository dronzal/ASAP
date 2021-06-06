"""
Source: https://github.com/anilsathyan7/Portrait-Segmentation
"""

import numpy as np
import cv2
from random import randint
import time
from tensorflow.keras.models import load_model
import os


class BackgroundMask:

    def __init__(self, log, model="models/segmentation/deconv_bnoptimized_munet.h5", bg_im_path="bg_im/"):
        self.log = log
        self.log.debug("init")
        # load model
        try:
            model = os.path.join(os.path.dirname(__file__), model)
            self.model = load_model(model, compile=False)
        except:
            raise Exception(f"Couldn't load the model\nGiven: {model}")

        # Target size
        self.tgt_size = 640

        bg_im_path = os.path.join(os.path.dirname(__file__), bg_im_path)
        self.bg_path = bg_im_path
        self.bg_list = self.get_bg(self.bg_path)
        self.bg_current = 0
        self.change_bgd(idx=randint(0, (len(self.bg_list)-1)))
        self.time = 0
        self.started = None
        self.bucket = None
        self.thread = None

        self.frame = None

    def get_bg(self, folder):
        """
        Function that list all jpg files

        :param folder path
        :return: file_list
        """

        files = os.listdir(folder)
        file_list = [file for file in files if file.endswith('jpg')]
        self.log.debug(f"background file list: {file_list}")
        return file_list

    def change_bgd(self, idx=0):
        """
        Function that choose the background by index of self.file_list

        :param idx: int, valid
        """
        if idx > len(self.bg_list):
            idx = 0
        if idx < 0:
            idx = len(self.bg_list) - 1

        if idx <= len(self.bg_list):
            bg_location = os.path.join(self.bg_path, (self.bg_list[idx]))
            self.log.debug(f"background file: {bg_location}")

            self.bg = cv2.resize(cv2.imread(bg_location), (self.tgt_size, self.tgt_size))
            self.bg = self.cvtColor(self.bg)
            self.bg_current = idx
        else:
            self.log.warning(f"Non valid bg_img idx.\nGiven {idx}.\nLen list {len(self.bg_list)} ")

    @staticmethod
    def cvtColor(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        return img

    def runTime(self, frame):
        try:
            self.bucket = None
            startTime = time.time()

            # Pre-process
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bg = cv2.resize(self.bg, (self.tgt_size, self.tgt_size))

            simg = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            simg = simg.reshape((1, 128, 128, 3)) / 255.0

            # Predict
            out = self.model.predict(simg)

            orimsk = np.float32((out > 0.5)).reshape((128, 128, 1))

            # Post-process
            msk = cv2.GaussianBlur(orimsk, (5, 5), 1)
            img = cv2.resize(img, (self.tgt_size, self.tgt_size), interpolation=cv2.INTER_AREA) / 255.0
            msk = cv2.resize(msk, (self.tgt_size, self.tgt_size)).reshape((self.tgt_size, self.tgt_size, 1))
            # Alpha blending
            frame = (img * msk) + (bg * (1 - msk))
            frame = np.uint8(frame * 255.0)
            mask = np.uint8(msk * 255.0)
            self.time = round((time.time() - startTime) * 1000)
            # Drop result in the bucket
            self.bucket = frame[..., ::-1]
        except Exception as e:
            self.log.warnig(f"{e}")
