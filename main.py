"""
ASAP Application

This file is part of the ASAP Interactive Videoconferencing AI/ML Tools
Copyright 2021, ASAP team, authored by Arne Depuydt
"""

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from collections import Counter, deque
from datetime import datetime
from datetime import datetime as d
from numpy import ndarray
from threading import Thread, Lock
from tools import speechToText as STT
from tools.bgmask import bg_mask as bgm
from tools.gesture import gesture_detection
from tools.vision_mood_detection import mood_detection as MD
import argparse
import asyncio
import concurrent.futures
import cv2
import json
import keyboard
import logging
import numpy as np
import os
import pyvirtualcam
import queue
import random
import re
import sys
import time
import websockets


class ASAP:

    def __init__(self, ws_name, logging, cam_width=640, cam_height=480, flip_frame=False):

        self.ws_name = ws_name
        self.log = logging
        self.flip_end_frame = flip_frame

        self.stt = STT.SpeechToText(google_credentials_file="./google_credentials.json")
        self.bgMask = bgm.BackgroundMask(log=self.log)
        self.visionMd = MD.MoodDetection(log=self.log)
        self.gesture = gesture_detection.GestureDetection(log=self.log)

        self.cam_width = cam_width
        self.cam_height = cam_height

        # Bool for continuous runTime
        self.started = False
        self.cap = None
        self.frame = np.zeros(shape=(self.cam_height, self.cam_width, 3), dtype=np.float32)
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

        self.action = False
        self.action_time = None
        self.action_time_max = 2  # Time between one action and the next gesture action executed

        self.cnt = 0

        self.result_frame = np.zeros(shape=(self.cam_height, self.cam_width, 3), dtype=np.float32)
        self.bgMask_frame = np.zeros(shape=(self.cam_height, self.cam_width, 3), dtype=np.float32)
        self.gesture_result = None
        self.timings = None

        self.result_queue = queue.Queue()
        self.mood_deque = deque(maxlen=10)

        self.lock = Lock()

        self.last_time_action = 0
        self.actionHandler_delay = 1 / 20  # 20 frames per second
        self.frame_capture_fps = 20
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
        try:
            with pyvirtualcam.Camera(width=self.cam_width, height=self.cam_height, fps=20) as cam:
                while self.started:
                    try:
                        data = input_q.get()
                        if isinstance(data, np.ndarray):
                            data = data[..., ::-1]
                            cam.send(data)
                            self.log.debug("virtualCam sendFrame")
                            cam.sleep_until_next_frame()
                        else:
                            self.log.warning(f"Type virtualcam frame: {type(data)}")
                            time.sleep(0.1)
                    except Exception as e:
                        self.log.warning(f"{e}")
        except Exception as e:
            self.log.warning(f"{e}")

    def actionhandler(self):
        """
        ActionHandler function. Always run in a thread.
        :return:
        """
        while self.started:
            try:
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
            except Exception as e:
                self.log.warning(f"{e}")

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
        self.log.debug(f"stt result: {self.stt_result}")

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

        # TODO add volume up and down

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

    def toggle_command_mode(self, activate):
        """
        Changes command_mode between True and False and resets all action variables
        """
        if activate:
            self.command_mode = True
        else:
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
        self.vote_text = ""

    def gesture_action(self, tmp: dict):
        """
        This is the action function for the gesture recognition
        """
        # Translate the results dictionary for gestures into a String
        for key in tmp.keys():
            li = tmp[key]['gesture']
            self.gesture_result = ' | '.join([str(elem) for elem in li])

        # Only one action per cycle and no action for some time after action done
        if self.action:
            if not isinstance(self.action_time, type(None)):
                if (time.time() - self.action_time >= self.action_time_max):
                    self.action = False
                    #print("=> Action timer reset: ", int(time.time() - self.action_time))
                    self.action_time = None

        # Potentially send "Command mode off" directly from Gesture detection
        if re.search(r"\bCommand mode off\b", self.gesture_result, re.I):
            if not self.action:
                self.toggle_command_mode(False)
            self.action = True

        # Use OK sign to switch off Command mode
        if re.search(r"\bOK\b", self.gesture_result, re.I):
            if not self.action:
                self.toggle_command_mode(False)
            self.action = True

        # Switch on Command mode
        if re.search(r"\bCommand mode on\b", self.gesture_result, re.I):
            if not self.action:
                self.toggle_command_mode(True)
            self.action = True

        # Change background one way
        if re.search(r"\b(Background-Left)\b", self.gesture_result, re.I):
            if not self.action:
                self.change_up_bgMask = True
            self.action = True

        # Change background the other way
        if re.search(r"\b(Background-Right)\b", self.gesture_result, re.I):
            if not self.action:
                self.change_down_bgMask = True
            self.action = True

        # Turn camera to black
        if re.search(r"\b(Camera-Off)\b", self.gesture_result, re.I):
            if not self.action:
                self.set_black_bg = True
            self.action = True

        # Reset mute and black camera
        if re.search(r"\b(Unmute/Camera-On)\b", self.gesture_result, re.I):
            if not self.action:
                self.unset_black_bg = True
                self.unmute = True
            self.action = True

        # Mute
        if re.search(r"\b(Mute)\b", self.gesture_result, re.I):
            if not self.action:
                self.mute = True
            self.action = True

        # Start action timer
        if self.action:
            if isinstance(self.action_time, type(None)):
                self.action_time = time.time()
                #print("=> Action timer started: ", self.action_time)

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
                self.mute = False
                keyboard.press_and_release('ctrl+shift+m')

            if self.unmute:
                print("unmuted")
                self.unmute = False
                keyboard.press_and_release('ctrl+shift+m')

            if self.change_bgMask:
                print("change bg random")
                self.bgMask.change_bgd(random.randint(0, len(self.bgMask.bg_list)))
                self.change_bgMask = False

            if self.set_black_bg:
                print("set black")
                self.black_bg = True
                self.set_black_bg = False

            if self.unset_black_bg:
                print("unmute/camera on")
                self.black_bg = False
                self.unset_black_bg = False

            if self.change_up_bgMask:
                print("change bg one way")
                self.bgMask.change_bgd(self.bgMask.bg_current + 1)
                self.change_up_bgMask = False

            if self.change_down_bgMask:
                print("change bg other way")
                self.bgMask.change_bgd(self.bgMask.bg_current - 1)
                self.change_down_bgMask = False

            if self.transcript_mode_on:
                print("transcript mode")
                self.transcript_mode = True
                self.transcript_mode_on = False
                self.transcript_done = True

            if self.transcript_mode_off:
                print("transcript mode off")
                self.transcript_mode = False
                self.transcript_mode_off = False
                self.transcript_done = True

        if self.transcript_mode:
            if not self.transcript_done:
                date_time_obj = datetime.now()
                timestamp_str = date_time_obj.strftime("%d-%b-%Y %H:%M:%S")
                self.transcript_done = True
                keyboard.write(self.ws_name + " at " + timestamp_str + ": ")
                keyboard.write(self.stt_result + "\n")


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
                        (int(xPositionEmojiLine-25), int(yPositionTop - heightLine - 20)),
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

        if self.command_mode:
            self.result_frame = self.print_rect(self.result_frame, "green")

        if self.mood() is not None and not self.black_bg:
            draw_mood()

        if self.voting_mode:
            display_vote()

        if self.flip_end_frame:
            self.result_frame = self.flip_frame(self.result_frame)

        self.frame_q.put(self.result_frame)

    @staticmethod
    def print_rect(image, color):
        c = (0,0,0)
        if color == "green":
            c = (0,255,0)
        elif color == "yellow":
            c = (0,255,255)
        elif color == "red":
            c = (0,0,255)
        image_width, image_height = image.shape[1], image.shape[0]
        image = cv2.rectangle(image, (1, 1), (image_width - 1, image_height - 1), c, 6)
        cv2.putText(image, "Command Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv2.LINE_AA)
        return image

    def start(self, start=True):
        """
        Function that starts the runtime
        :param start: Boolean, std True
        :return:
        """
        if self.started and start:
            self.log.warning("ASAP Already started")
        else:
            self.started = start
            self.runtime()

    def stop(self):
        """
        Functions that stops en close the application
        """
        self.started = False
        self.log.debug("Exit")
        sys.exit()

    def videoCap(self):
        """
        Function that store a img frame in self.frame captured by device 0
        Always run this in a thread.
        :return:
        """
        self.log.debug("started")
        delay = 1 / self.frame_capture_fps
        last = 0
        try:
            cap = cv2.VideoCapture(0)
            self.log.info("videoCapture(0)")
        except Exception as e:
            cap = cv2.VideoCapture(cv2.CAP_DSHOW)
            self.log.warning(f"cv.VideoCapture {e}")
        while self.started:
            try:
                timeNow = time.time()
                if (timeNow - last) >= delay:
                    self.ret, self.frame = cap.read()
                    last = timeNow
                time.sleep(delay / 5)
            except Exception as e:
                self.log.warning(f"while loop {e}")

    def videoShow(self):
        """
        Function that shows the frame in cv2 GUI.
        Always run this in a thread.
        :return:
        """
        while self.started:
            try:
                if self.show_video:
                    if isinstance(self.result_frame, ndarray):
                        cv2.imshow('frame', self.result_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.stop()
                        self.result_frame = None
                time.sleep(self.while_delay)
            except Exception as e:
                self.log.warning(f"ASAP videoShow {e}")

    def websocket(self, in_q: queue.Queue):
        while self.started:
            async def start_ws():
                try:
                    async with websockets.connect("ws://84.196.102.201:6789") as websocket:
                        self.log.debug(f"connected")
                        while True:
                            while in_q.not_empty:
                                data = in_q.get()
                                msg = json.dumps(data)
                                self.log.debug(f"About to send {msg}")
                                await websocket.send(msg)
                                res = await websocket.recv()
                                self.log.debug(f"Res:  {res}")
                                time.sleep(self.while_delay)
                            time.sleep(0.2)
                except Exception as e:
                    self.log.warning(f"{e}")
                    time.sleep(1)

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                asyncio.get_event_loop().run_until_complete(start_ws())
            except Exception as e:
                self.log.warning(f"{e}")
            time.sleep(5)

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
        self.log.debug("Init runtime")
        # Init a thread for the Speech to Text service, and pass the queue.
        self.stt_thread = Thread(target=self.stt.runTime, args=(self.result_queue, self.lock, self.log),
                                 name="ASAP_stt", daemon=True)
        self.stt_thread.start()

        # Init a thread for the VideoCapture Service.
        self.videoCap_thread = Thread(target=self.videoCap, daemon=True,
                                      name="ASAP_videocap")
        self.videoCap_thread.start()

        self.action_thread = Thread(target=self.actionhandler, daemon=True,
                                    name="ASAP_actions")
        self.action_thread.start()

        self.videoShow_thread = Thread(target=self.videoShow, daemon=True,
                                       name="ASAP_vidshow")
        self.videoShow_thread.start()

        self.virtualCam_thread = Thread(target=self.virtualCam, daemon=True, args=(self.frame_q,),
                                        name="ASAP_virtualcam")
        self.virtualCam_thread.start()

        self.websocket_thread = Thread(target=self.websocket, args=(self.websocket_q,),
                                       name='"ASAP_websocket')
        self.websocket_thread.start()

        while self.started:
            # Get time, this for calculating the total frames per second.
            startTime = time.time()

            # Init an async threadPool and wait if the childThreads are finished to go further.
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    executor.submit(self.bgMask.runTime, self.frame)
                    executor.submit(self.visionMd.runTime, self.frame)
                    executor.submit(self.gesture.runTime, self.frame)
            except Exception as e:
                self.log.warning(f"ThreadPoolExcecutor {e}")

            if not isinstance(self.visionMd.bucket, type(None)):
                self.result_queue.put({"mood": self.visionMd.bucket})
                now = datetime.now()
                if now.second % 5 == 0:
                    self.websocket_q.put({
                        "action": "mood",
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
            self.log.debug = f"timings: {self.timings}"

            # Set timings to zero
            self.bgMask.time = 0
            self.visionMd.time = 0
            self.gesture.time = 0


def load_args():
    """
    Loading command line arguments
    Available args:
        --name
        --debug
        --google_cred
        --height
        --width
    :return: dict
    """

    parser = argparse.ArgumentParser(prog="tool", description='ASAP, add AI features to your camera stream',
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=35))
    parser.add_argument(
        '-n', '--name', required=False, type=str, help='User name for websocket')
    parser.add_argument(
        '-d', '--debug', required=False, type=bool, default=True, help='Add debug info to logs/*.log')
    parser.add_argument(
        '-g', '--google_cred', required=False, default="google_credentials.json", type=str,
        help='Google credentials file location')
    parser.add_argument(
        '-H', '--height', required=False, type=int, help='Vritual cam height')
    parser.add_argument(
        '-W', '--width', required=False, type=int, help='Vritual cam weight')
    parser.add_argument(
        '-l', '--level', required=False, choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        default="DEBUG", type=str, help='Debug level')
    parser.add_argument(
        '-f', '--flip_frame', required=False, default=False, type=int, help='Flip frame vertically')
    args = parser.parse_args()

    results = {}

    if args.name is not None:
        results['name'] = args.name
    else:
        results['name'] = input("User name: ")
    results['debug'] = args.debug
    if args.google_cred is not None:
        file = args.google_cred
        if os.path.exists(file):
            results['google'] = file
        else:
            tmp = ""
            while not os.path.exists(tmp):
                tmp = input("Google credentials file not found.\nInput the file location:\n")
            results['google'] = tmp
    if args.height is not None:
        results['cam_height'] = args.height
    if args.width is not None:
        results['cam_width'] = args.width
    if args.level is not None:
        levels = {"CRITICAL": 50,
                  "ERROR": 40,
                  "WARNING": 30,
                  "INFO": 20,
                  "DEBUG": 10,
                  "NOTSET": 0}
        results['level'] = levels.get((str(args.level).upper()))
    if args.flip_frame is not None:
        results['flip_frame'] = args.flip_frame

    return results


if __name__ == "__main__":
    # load args
    args = load_args()

    # init a dateTime string
    date = d.now()
    dt_string = date.strftime("%Y%m%d_%H%M%S")

    if not os.path.exists("logs/"):
        os.makedirs("logs")
    # init a logfile
    logging.basicConfig(filename=f"logs/{dt_string}.log",
                        level=args.get("level"),
                        datefmt='%d-%m-%y %H:%M:%S',
                        format='%(levelname)s %(asctime)s %(module)s %(funcName)s | %(message)s',
                        )

    # init main app
    asap = ASAP(ws_name=args.get('name'), logging=logging, flip_frame= args.get('flip_frame'))

    # start main app in a Thread, main purpose is to run as a containerized app
    asap_thread = Thread(target=asap.start, daemon=True, name="ASAP_MainThread")

    # Start the main thread
    asap_thread.start()
    while asap.started:
        #print(asap.gesture_result)
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            asap.stop()
