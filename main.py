"""
ASAP Application

This file is part of the ASAP Interactive Videoconferencing AI/ML Tools
Copyright 2021, ASAP team, authored by Arne Depuydt
"""

import re
from datetime import datetime

import keyboard

from tools import speechToText as STT
from tools.bgmask import bg_mask as bgm
from tools.vision_mood_detection import mood_detection as MD
from tools.gesture import gesture_detection
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
import websockets
import asyncio
import json


class ASAP:

    def __init__(self, ws_name, cam_width=640, cam_height=480):

        while not ws_name:
            ws_name = input("Please, input a user name: ")
        self.ws_name = ws_name

        self.stt = STT.SpeechToText(google_credentials_file="./google_credentials.json")
        self.bgMask = bgm.BackgroundMask()
        self.visionMd = MD.MoodDetection()
        self.gesture = gesture_detection.GestureDetection()

        self.cam_width = cam_width
        self.cam_height = cam_height

        # Bool for continuous runTime
        self.started = False
        self.cap = None
        self.frame = np.zeros(shape=(self.cam_height, self.cam_width, 3))
        self.counter = 0
        self.show_video = True

        self.stt_result = ""
        self.gesture_result = ""
        self.black_bg = False
        self.set_black_bg = False
        self.unset_black_bg = False
        self.change_bgMask = False
        self.change_up_bgMask = False
        self.change_down_bgMask = False
        self.mute = False
        self.unmute = False
        self.command_mode = False
        self.transcript_mode_on = False
        self.transcript_mode_off = False
        self.transcript_mode = False
        self.transcript_done = False
        self.voting_mode = True
        self.vote_text = ""

        self.debug = False
        self.result_frame = np.zeros(shape=(self.cam_height, self.cam_width, 3))
        self.bgMask_frame = np.zeros(shape=(self.cam_height, self.cam_width, 3))
        self.gesture_result = None
        self.timings = None

        self.result_queue = queue.Queue()
        self.mood_deque = deque(maxlen=10)

        self.lock = Lock()

        self.last_time_action = 0
        self.actionHandler_delay = 1 / 20  # 20 frames per second

        self.while_delay = 0.05

        # Actions
        self.black_bg = False
        self.show_gesture_debug = False

        self.websocket_q = queue.Queue()
        self.frame_q = queue.Queue()

        self.emotions_dict = {0: {"emotion": "ANGRY"},
                              1: {"emotion": "DISGUST"},
                              2: {"emotion": "FEAR"},
                              3: {"emotion": "HAPPY"},
                              4: {"emotion": "SAD"},
                              5: {"emotion": "SURPRISE"},
                              6: {"emotion": "NEUTRAL"}}

    def virtualCam(self, input_q):
        """
        VirtualCam function
        :return:
        """
        with pyvirtualcam.Camera(width=self.cam_width, height=self.cam_height, fps=20) as cam:
            while self.started:
                data = input_q.get()
                data = data[..., ::-1]
                cam.send(data)

    def actionhandler(self):
        """
        ActionHandler function. Always run in a thread.
        :return:
        """
        while self.started:
            t = time.time()
            if (t - self.last_time_action) >= self.actionHandler_delay:
                if not self.result_queue.empty():
                    with self.lock:
                        tmp_queue = self.result_queue
                    length_queue = len(list(tmp_queue.queue))
                    for x in range(length_queue):
                        result = tmp_queue.get()
                        key = dict(result).keys()
                        # possible actions:
                        # mood, stt, bg, gesture
                        if "mood" in key:  # if mood is in result
                            self.mood_deque.append(result['mood'])

                        elif "stt" in key:
                            self.stt_actions(result['stt'])

                        elif "bg" in key:
                            self.bgMask_frame = self.flip_frame(result["bg"])

                        elif "gesture" in key:
                            self.gesture_action(result["gesture"])

                    self.build_end_frame()
                    self.build_actions()

                self.last_time_action = t
            time.sleep(self.while_delay)

    def stt_actions(self, tmp: str):
        """
        This is the action function for the speechToText, stt

        The available actions are
            - Toggle of the command mode
            - Toggle of the transcript mode
            - Random change of the background
            - Set and unset the background to black
            - Toggle mute, mute and unmute
            - Toggle the gesture debug
            - Turn on Voting mode at set text to display

        :param tmp:
        :return:
        """

        self.stt_result = tmp
        # print(self.stt_result)
        self.transcript_done = False
        self.websocket_q.put({
            "action": "log_entry",
            "name": self.ws_name,
            "payload": str(self.stt_result)
        })

        if re.search(r"\b(command mode on)\b", tmp, re.I):
            self.command_mode = True
            self.transcript_mode_on = False
            self.transcript_mode_off = False
            self.change_bgMask = False
            self.change_up_bgMask = False
            self.change_down_bgMask = False
            self.set_black_bg = False
            self.unset_black_bg = False
            self.mute = False
            self.unmute = False
            self.show_gesture_debug = False

        if re.search(r"\b(command mode off)\b", tmp, re.I):
            self.command_mode = False
            self.transcript_mode_on = False
            self.transcript_mode_off = False
            self.change_bgMask = False
            self.change_up_bgMask = False
            self.change_down_bgMask = False
            self.set_black_bg = False
            self.unset_black_bg = False
            self.mute = False
            self.unmute = False
            self.show_gesture_debug = False

        if re.search(r"\b(transcript mode on)\b", tmp, re.I):
            self.transcript_mode_on = True

        if re.search(r"\b(transcript mode off)\b", tmp, re.I):
            self.transcript_mode_off = True

        if re.search(r"\b(change background)\b", tmp, re.I):
            self.change_bgMask = True

        if re.search(r"\b(background left)\b", tmp, re.I):
            self.change_up_bgMask = True

        if re.search(r"\b(background right)\b", tmp, re.I):
            self.change_down_bgMask = True

        if re.search(r"\b(camera off)\b", tmp, re.I):
            self.set_black_bg = True

        if re.search(r"\b(camera on)\b", tmp, re.I):
            self.unset_black_bg = True

        if re.search(r"\b(toggle mute)\b", tmp, re.I):
            self.mute = not self.mute

        if re.search(r"\b(mute)\b", tmp, re.I):
            self.mute = True

        if re.search(r"\b(unmute)\b", tmp, re.I):
            self.unmute = True

        if re.search(r"\b(gesture)\b", tmp, re.I):
            self.show_gesture_debug = True

        if re.search(r"\b(voting on)\b", tmp, re.I):
            self.voting_mode = True
            self.vote_text = "Let's vote"

        if re.search(r"\b(voting off)\b", tmp, re.I):
            self.voting_mode = False
            self.vote_text = ""

        if re.search(r"\b(I vote yes)\b", tmp, re.I):
            self.vote_text = "YES"

        if re.search(r"\b(I vote no)\b", tmp, re.I):
            self.vote_text = "NO"

        if re.search(r"\b(option 1|option A)\b", tmp, re.I):
            self.vote_text = "A"

    def gesture_action(self, tmp: dict):
        """
        This is the action function for the gesture recognition

        The available actions are
            - Command Mode
            Move into command mode: Show two hands to the webcam
            Cancel command mode: Show two hands again
            - Audio
            Mute the microphone: Show flat palm of one hand
            Un-mute the microphone: Make an upward fist
            Increase the volume: Index finger up (and thumb to the side)
            Decrease the volume: Index finger down (and thumb to the side)
            - Video
            Black out the Camera: Point fist at the camera
            Return to Webcam display: Show upwards fist (same as un-mute)
            - Background
            Change the Background one-forward: Fist with thumb to one side
            Change the Background one-backward: Fist with thumb to the other side
            - Voting
            Begin a voting process: Victory sign
            Set the number of options:
                Indicate yes/no question: Thumns-up sign or
                Show number of fingers [1..5]
            Confirm the number of options displayed: OK sign
            --- Explain the options to the group ---
            Start the voting: Victory sign
            Cast your vote:
                In case of yes/no: Show thumbs-up or
                In case of more options: Show number with your fingers
            Confirm vote: OK sign

        :param tmp:
        :return:
        """
        for key in tmp.keys():
            li = tmp[key]['gesture']
            self.gesture_result = ' '.join([str(elem) for elem in li])

        if 'mute' in self.gesture_result:
            pass
            # self.mute = True
            # print("muted")

    def build_actions(self):
        """
        This functions takes care of the actions coming from the gesture control and stt

        Performed actions are:
            - mute/unmute: this is done by interaction with a chat client through keyboard shortcuts
            - The change of the background is done in a random way. 5 Backgrounds are available
            - Set/unset the image to black. This can be seen as turning off or on the camera.
            - Set the background to the next one or the previous one.
            - Enable/disable transcript

        :return:
        """
        if self.command_mode:
            if self.mute:
                print("muted")
                keyboard.press_and_release('ctrl+shift+m')
                self.mute = False

            if self.unmute:
                print("unmuted")
                keyboard.press_and_release('ctrl+shift+m')
                self.unmute = False

            if self.change_bgMask:
                self.bgMask.change_bgd(random.randint(0, len(self.bgMask.bg_list)))
                self.change_bgMask = False

            if self.set_black_bg:
                self.black_bg = True
                self.set_black_bg = False

            if self.unset_black_bg:
                self.black_bg = False
                self.unset_black_bg = False

            if self.change_up_bgMask:
                self.bgMask.change_bgd(self.bgMask.bg_current + 1)
                self.change_up_bgMask = False

            if self.change_down_bgMask:
                self.bgMask.change_bgd(self.bgMask.bg_current - 1)
                self.change_down_bgMask = False

            if self.transcript_mode_on:
                self.transcript_mode = True
                self.transcript_mode_on = False
                self.transcript_done = True

            if self.transcript_mode_off:
                self.transcript_mode = False
                self.transcript_mode_off = False
                self.transcript_done = True

        if self.transcript_mode:
            if not self.transcript_done:
                date_time_obj = datetime.now()
                timestamp_str = date_time_obj.strftime("%d-%b-%Y %H:%M:%S")
                keyboard.write(self.ws_name + " at " + timestamp_str + ": ")
                keyboard.write(self.stt_result + "\n")
                self.transcript_done = True

    @staticmethod
    def flip_frame(frame):
        """
        Function that flip the img by vertical axis
        :param frame: ndarray
        :return: flipped img
        """
        return cv2.flip(frame, 1)

    def build_end_frame(self):
        """
        Function that builds the end frame
        :return:
        """

        def display_vote():
            org = (10, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            color = (255, 0, 0)
            thickness = 5
            cv2.putText(self.result_frame, self.get_vote_text, org, font,
                        font_scale, color, thickness)

        def draw_mood():
            xPositionEmojiLine = 500
            heightLine = 200
            yPositionTop = 250
            height_emoji = heightLine * self.predictions()[0][self.mood()]
            start_point = (xPositionEmojiLine, yPositionTop - heightLine)
            end_point = (xPositionEmojiLine, yPositionTop)
            color = (0, 0, 0)
            thickness = 5
            cv2.putText(self.result_frame, self.mood(getIndex=False).get('emotion'),
                        (int(xPositionEmojiLine), int(yPositionTop - heightLine - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.result_frame = cv2.line(self.result_frame, start_point, end_point, color, thickness)
            self.result_frame = cv2.circle(self.result_frame,
                                           (int(xPositionEmojiLine), int(yPositionTop - height_emoji)), 10, (255, 0, 0),
                                           2)

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

        if self.mood() is not None:
            draw_mood()

        if self.voting_mode:
            display_vote()

        self.frame_q.put(self.result_frame)

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
        try:
            cap = cv2.VideoCapture(0)
        except Exception:
            cap = cv2.VideoCapture(cv2.CAP_DSHOW)

        while self.started:
            self.ret, self.frame = cap.read()

    def videoShow(self):
        """
        Function that shows the frame in cv2 GUI.
        Always run this in a thread.
        :return:
        """
        while self.started:
            if self.show_video:
                if isinstance(self.result_frame, ndarray):
                    cv2.imshow('frame', self.result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop()
                    self.result_frame = None
            time.sleep(self.while_delay)

    def websocket(self, in_q):
        async def start_ws():
            async with websockets.connect("ws://84.196.102.201:6789") as websocket:
                print("connected")
                while self.started:
                    data = in_q.get()
                    msg = json.dumps(data)
                    print("About to send", msg)
                    await websocket.send(msg)
                    res = await websocket.recv()
                    print("Res: ", res)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        asyncio.get_event_loop().run_until_complete(start_ws())

    @property
    def bg_mask_frame(self):
        """
        Returns the finale result frame
        :return: ndarray
        """
        return self.to_jpg(self.bgMask_frame)

    @property
    def gesture_frame(self):
        """
        Returns the gesture debug frame
        :return: ndarray
        """
        return self.to_jpg(self.gesture.debug_frame)

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
        return self.stt_result

    @property
    def get_vote_text(self):
        """
        Returns the vote text
        :return: string
        """
        return self.vote_text

    @property
    def get_gesture(self):
        return self.gesture_result

    @property
    def get_initial_frame(self):
        """
        Returns the initial frame captured by camera
        :return: ndarray
        """
        return self.to_jpg(self.frame)

    def to_jpg(self, frame):
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg

    def mood(self, mostCommon=True, getIndex=True):
        """
         This functions sort the objects in self.mood_queue by number occurs
         and returns the most occurs element string name if mostCommon = True.
         Otherwise the function return the full list.
        :return: str
        """
        li = list(self.mood_deque)
        index = -1
        if len(li):
            if mostCommon:
                index = Counter(map(lambda el: el.get("dominant_index"), li)).most_common(1)[0][0]
            else:
                return self.emotions_dict.get(Counter(map(lambda el: el.get("dominant_index"),
                                                          li)).most_common(1)[0][0])
            if getIndex:
                return index
            else:
                return self.emotions_dict.get(index)
        else:
            return None

    def predictions(self):
        li = list(self.mood_deque)
        if len(li):
            return li[len(li) - 1].get("predictions")

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

        self.action_thread = Thread(target=self.actionhandler, daemon=True)
        self.action_thread.start()

        self.videoShow_thread = Thread(target=self.videoShow, daemon=True)
        self.videoShow_thread.start()

        #self.virtualCam_thread = Thread(target=self.virtualCam, daemon=True, args=(self.frame_q,))
        #self.virtualCam_thread.start()

        self.websocket_thread = Thread(target=self.websocket, args=(self.websocket_q,))
        self.websocket_thread.start()

        while self.started:
            # Get time, this for calculating the total frames per second.
            startTime = time.time()

            # Init an async threadPool and wait if the childThreads are finished to go further.
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.bgMask.runTime, self.frame)
                executor.submit(self.visionMd.runTime, self.frame)
                executor.submit(self.gesture.runTime, self.frame)

            if not isinstance(self.visionMd.bucket, type(None)):
                self.result_queue.put({"mood": self.visionMd.bucket})
                self.websocket_q.put({
                    "action":"mood",
                    "name": self.ws_name,
                    "payload": self.visionMd.bucket.get("predictions")
                })
                self.visionMd.bucket = None

            if not isinstance(self.gesture.bucket, type(None)):
                with self.lock:
                    self.result_queue.put({"gesture": self.gesture.bucket})
                self.gesture.bucket = None

            if not isinstance(self.stt.bucket, type(None)):
                with self.lock:
                    self.result_queue.put({"stt": self.stt.bucket})
                self.stt.bucket = None

            if isinstance(self.bgMask.bucket, ndarray):
                tmp = cv2.resize(self.bgMask.bucket, (self.cam_width, self.cam_height), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.result_queue.put({"bg": tmp})
                self.bgMask.bucket = None

            stopTime = time.time()

            try:
                fps = round(1 / (stopTime - startTime), 1)
            except ZeroDivisionError:
                fps = 0

            self.timings = {"fps": fps,
                            "bgMask": self.bgMask.time,
                            "visionMd": self.visionMd.time,
                            "gesture": self.gesture.time, }

            # Set timings to zero
            self.bgMask.time = 0
            self.visionMd.time = 0
            self.gesture.time = 0


if __name__ == "__main__":
    asap = ASAP(ws_name="Pieter-Jan")
    asap_thread = Thread(target=asap.start, daemon=True)
    asap_thread.start()
    while asap.started:
        print(asap.get_gesture)
        # print("stt", asap.get_stt)
        time.sleep(0.2)
        pass
