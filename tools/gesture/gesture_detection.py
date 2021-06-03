"""
Gesture Detection

This file is part of the ASAP Interactive Videoconferencing AI/ML Tools
Copyright 2021, ASAP team, authored by Alexis Metz

The code makes use of existing libraries and is based in large parts on the following repositories:
- It uses the Mediapipe framework published by Google
https://mediapipe.dev/
- It is based on code published by Kazuhito00 on Github
https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe/blob/main/README_EN.md
published under Apache 2.0 licence
https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe/blob/main/LICENSE
- It uses the hand gestures trained by kinivi, with one addition trained extra
https://github.com/kinivi/tello-gesture-control
published under Apache 2.0 licence
https://github.com/kinivi/tello-gesture-control/blob/main/LICENSE
"""

# Import libraries: ml
import cv2 as cv
import mediapipe as mp
import os
# Import libraries: camera
import pyvirtualcam

# Import libraries: tools
import csv
import numpy as np
import copy
import time
import sys
import itertools

# Import libraries: collections
from collections import Counter
from collections import deque

# Import libraries: threading
import threading
import setproctitle

# Import libraries: local classes
from .utils import cvfpscalc
from .model.keypoint_classifier import keypoint_classifier as kc
from .model.point_history_classifier import point_history_classifier as phc


class GestureDetection:
    """
    Gesture Detection Class

    Called in a Thread from the ASAP aplication

    """

    # Init #############################################################################################################
    def __init__(self):
        """

        """
        # Init MediaPipe parameters
        self.mp_max_num_hands = 2
        self.mp_min_detection_confidence = 0.7
        self.mp_min_tracking_confidence = 0.5
        self.mp_use_brect = 'store_true'

        # Load MediaPipe Hand features
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=self.mp_max_num_hands,
                                         min_detection_confidence=self.mp_min_detection_confidence,
                                         min_tracking_confidence=self.mp_min_tracking_confidence)

        self.time = 0
        self.started = False
        self.thread = None
        self.frame = None
        self.bucket = None
        self.debug_frame = None

        self.cmd_queue = deque(maxlen=10)  # Queue to stabilize by confirming ten similar readings

        # Create the Keypoint classifier object
        self.keypoint_classifier = kc.KeyPointClassifier()
        # Create the Point_history classifier object
        self.point_history_classifier = phc.PointHistoryClassifier()
        path1 = os.path.join(os.path.dirname(__file__), 'model/keypoint_classifier/keypoint_classifier_label.csv')
        # Read the labels stored in the respective csv files
        with open(path1, encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in self.keypoint_classifier_labels]

        path2 = os.path.join(os.path.dirname(__file__), 'model/point_history_classifier/point_history_classifier_label.csv')
        with open(path2, encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [row[0] for row in self.point_history_classifier_labels]

        # Create the point_history deque
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        # Create the finger_gesture_history deque
        self.finger_gesture_history = deque(maxlen=self.history_length)

        # Create the frames per second FPS rate utils object
        self.cvFpsCalc = cvfpscalc.CvFpsCalc(buffer_len=10)

    # Start ############################################################################################################
    def start(self):
        if self.started:
            print("Gesture detection thread already started.")
        else:
            print("Starting gesture detection")
            self.started = True
            self.thread = threading.Thread(target=self.runTime, name='GestureThread', daemon=True)
            self.thread.start()

    # Stop #############################################################################################################
    def stop(self):
        if not self.started:
            print("Gesture detection thread already stopped.")
        else:
            self.started = False

    # Runtime ##########################################################################################################
    def runTime(self, frame):
        """ Runtime execution

        :param self:
        :return:
        """
        # Get the FPS rate
        self.fps = self.cvFpsCalc.get()
        # Set the start time
        startTime = time.time()
        # Image manipulation
        image = cv.flip(frame, 1)  # Flip image
        debug_image = copy.deepcopy(image)  # Copy image
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Change to RGB for Mediapipe

        # Process image with Mediapipe hands model
        image.flags.writeable = False  # Set writeable to False to speed up process
        results = self.hands.process(image)
        image.flags.writeable = True

        # When hand detected:
        if not isinstance(results.multi_hand_landmarks, type(None)):
            tmp = {}
            counter = 0

            """
            self.cmd_queue.append()

            if len(results.multi_hand_landmarks) == 2:
                self.cmd_queue.append(True)
                print("Command Mode")
            else:
                self.cmd_queue.append(False)

            if Deque.count(cmd_queue[0]) == len(cmd_queue):
                print("all equal")

            """

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                # Calculate center of palm
                cx, cy = self.calc_palm_moment(debug_image, hand_landmarks)
                # Calculate bounding rectangle of hand
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Create a list from the detected landmarks
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                # Pre-process landmark list and point history list
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(debug_image, self.point_history)
                # Run keypoint_classifier model and write to point history deque if hand sign id = 2
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Index finger raised
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])
                # Run point_history classifier model (when 32 values in preprocessed list = full 16 history points)
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)

                # Append result to finger_gesture history deque and select most common recognized gesture
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                # Draw landmarks, bounding rectangle and information
                debug_image = self.draw_landmarks(debug_image, cx, cy, landmark_list, handedness)
                debug_image = self.draw_bounding_rect(self.mp_use_brect, debug_image, brect)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                tmp[counter] = {"gesture": [self.keypoint_classifier_labels[hand_sign_id],
                                        self.point_history_classifier_labels[most_common_fg_id[0][0]]]}

                # Test simple number recognition directly with Mediapipe
                thumb_up, recognized_sum = self.recognize_hand_gesture(image, cx, cy, hand_landmarks, handedness)

                counter += 1
            self.bucket = tmp

        else:
            self.point_history.append([0, 0])  # No hand visible on the screen

        # Anyway draw point history and regular information when index finger is shown
        debug_image = self.draw_point_history(debug_image, self.point_history)
        debug_image = self.draw_info(debug_image, self.fps)

        self.debug_frame = debug_image
        self.time = round((time.time() - startTime) * 1000)

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
        image = cv.rectangle(image, (1, 1), (image_width - 1, image_height - 1), c, 6)
        return image

    @staticmethod
    def print_on_image(image, title, text, x_offset, y_offset):
        """
        :param image: The current frame
        :param title: The initial text to print
        :param text: The parametervalue to print
        :param x_offset: The x offset
        :param y_offset: The y offset
        :return: The image with additional line printed
        """
        cv.putText(image, str(title) + " " + str(text), (x_offset, y_offset),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        return image

    @staticmethod
    def calc_palm_moment(image, landmarks):
        """ Returns the center of the palm as X and Y coordinates
        Utilizes the moments calculation to determine the center of the palm surrounding parameters
        that are in the landmarks identified by Mediapipe

        :param image: The current frame
        :param landmarks: The landmarks as detected by Mediapipe
        :return: X and Y of the center of the palm object
        """
        # Get image width and height
        image_width, image_height = image.shape[1], image.shape[0]
        # Create an empty numpy array with integer values
        palm_array = np.empty((0, 2), int)
        # For each landmark in detected landmarks:
        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # Create a landmark point from x and y values
            landmark_point = [np.array((landmark_x, landmark_y))]
            # Append the position of the palm surrounding parameters to the palm array
            if index == 0:  # WRIST
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 1:  # THUMB_CMC
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 5:  # INDEX_FINGER_MCP
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 9:  # MIDDLE_FINGER_MCP
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 13:  # RING_FINGER_MCP
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 17:  # PINKY_MCP
                palm_array = np.append(palm_array, landmark_point, axis=0)
        # Calculate the moments of the palm surrounding points
        M = cv.moments(palm_array)  # Calculate
        cx, cy = 0, 0  # Set x and y to zero
        # Calculate the center x and y of the palm
        if M['m00'] != 0:  # Area of object is not null
            cx = int(M['m10'] / M['m00'])  # Center of object x
            cy = int(M['m01'] / M['m00'])  # Center of object y
        return cx, cy

    @staticmethod
    def calc_bounding_rect(image, landmarks):
        """ Returns the corners of a rectangle that frames the detected hand
        Calculate the corners of the bounding rectangle covering all landmark points
        identified by Mediapipe

        :param image: The current frame
        :param landmarks: The landmarks as detected by Mediapipe
        :return: X and Y coordinates of the corners of a rectangle
        """
        # Get image width and height
        image_width, image_height = image.shape[1], image.shape[0]
        # Create an empty numpy array with integer values
        landmark_array = np.empty((0, 2), int)
        # For each landmark in detected landmarks:
        for _, landmark in enumerate(landmarks.landmark):
            # Get the x and y positions of each landmark (percent * total width/height)
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        # Use OpenCv to calculate the bounding rectangle of all landmarks
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    @staticmethod
    def draw_landmarks(image, cx, cy, landmark_point, handedness):
        """ Drawing the landmarks on the image

        :param image: The current frame
        :param cx: The x coordnate of the center of the palm
        :param cy: The x coordinate of the center of the palm
        :param landmark_point: The hand landmark list (as detected by Mediapipe and transformed into a list)
        :param handedness: The handedness (as detected by Mediapipe)
        :return: The frame with drawings added
        """
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
            # Index
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)
            # Middle
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)
            # Ring
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)
            # Pinky
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)
            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

            # Print handedness indicator (L, R) in center of palm
            # Not used: handedness.classification[0].index, handedness.classification[0].score
            cv.circle(image, (cx, cy), 12, (255, 255, 255), -1)
            cv.circle(image, (cx, cy), 12, (0, 0, 0), 1)
            cv.putText(image,
                       handedness.classification[0].label[0],  # Text
                       (cx - 6, cy + 6),  # Position
                       cv.FONT_HERSHEY_SIMPLEX,  # Font
                       0.6, (0, 0, 255), 2,  # Text Size, Color BGR, Thickness
                       cv.LINE_AA) # Line Type

        # Draw circles for landmark points
        for index, landmark in enumerate(landmark_point):
            if index in (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19):
                size = 5
                color = (255, 255, 255)
            else:
                size = 8
                color = (0, 0, 255)
            cv.circle(image, (landmark[0], landmark[1]), size, color, -1)
            cv.circle(image, (landmark[0], landmark[1]), size, (0, 0, 0), 1)

        return image

    @staticmethod
    def draw_info_text(image, brect, hand_sign_text, finger_gesture_text):
        '''
        Draw information on detected hand sign and finger gesture
        :param image: Handover image to write on
        :param brect: Calculated rectangle that covers the hand area
        :param handedness: Left/right handedness as derived from Mediapipe
        :param hand_sign_text: The detected hand sign
        :param finger_gesture_text: The detected finger gesture
        :return: Adapted image
        '''
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = ""  # handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if hand_sign_text != "":
            cv.putText(image, "Hand Sign: " + hand_sign_text, (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Hand Sign: " + hand_sign_text, (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        if finger_gesture_text != "":
            cv.putText(image, "Finger Gesture: " + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Finger Gesture: " + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        return image

    @staticmethod
    def draw_point_history(image, point_history):
        '''
        Cycle over point history and display circle in 'reducing' size 1 + index / 2
        :param image: Handover image to write on
        :param point_history: handover deque with 16 points
        :return: Adapted image
        '''
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                          (0, 0, 255), 2)

        return image

    @staticmethod
    def draw_info(image, fps):
        ''' Write information on FPS on image
        :param image: Handover image to write on
        :param fps: Handover FPS rate
        :return: Adapted image
        '''
        image_width, image_height = image.shape[1], image.shape[0]
        cv.putText(image, "FPS:" + str(fps), (10, image_height-30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, image_height-30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        return image

    @staticmethod
    def draw_bounding_rect(use_brect, image, brect):
        """ Print the bounding rectangle on the image
        :param use_brect: Take from arguments if rectangle should be printed
        :param image: The current frame
        :param brect: The corners of the bounding rectangle
        :return: The annotated image
        """
        if use_brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)
        return image

    @staticmethod
    def recognize_hand_gesture(image, cx, cy, landmarks, handedness):
        """
        :param landmarks:
        :return:
        """
        # Get image width and height
        image_width, image_height = image.shape[1], image.shape[0]
        # initiate a list
        landmark_point = []
        # For each landmark in detected landmarks:
        for index, landmark in enumerate(landmarks.landmark):
            # Suppress landmarks that are either not present or visible
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            # Get the x and y positions of each landmark (percent * total width/height)
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # Append the landmark coordinate
            landmark_point.append((landmark_x, landmark_y))

        recognized_sum = None
        thumb_up = -1
        thumb_state = -1
        index_state = -1
        middle_state = -1
        ring_state = -1
        pinky_state = -1

        # Thumb up or down
        if landmark_point[2][1] < landmark_point[3][1] and (landmark_point[3][1]) < landmark_point[4][1]:
            thumb_up = 0
        else:
            thumb_up = 1

        # Thumb left or right
        if handedness.classification[0].label[0] == "R":
            lh = 1
        else:
            lh = -1
        if (lh * landmark_point[3][0]) < (lh * landmark_point[4][0]):
            thumb_state = 0
        else:
            thumb_state = 1
        # Index up or down
        if landmark_point[6][1] < landmark_point[7][1] and (landmark_point[7][1]) < landmark_point[8][1]:
            index_state = 0
        else:
            index_state = 1
        # Middle up or down
        if landmark_point[10][1] < landmark_point[11][1] and (landmark_point[11][1]) < landmark_point[12][1]:
            middle_state = 0
        else:
            middle_state = 1
        # Ring up or down
        if landmark_point[14][1] < landmark_point[15][1] and (landmark_point[15][1]) < landmark_point[16][1]:
            ring_state = 0
        else:
            ring_state = 1
        # pinky up or down
        if landmark_point[18][1] < landmark_point[19][1] and (landmark_point[19][1]) < landmark_point[20][1]:
            pinky_state = 0
        else:
            pinky_state = 1

        recognized_sum = thumb_state + index_state + middle_state + ring_state + pinky_state

        return thumb_up, recognized_sum

    @staticmethod
    def calc_landmark_list(image, landmarks):
        # Get image width and height
        image_width, image_height = image.shape[1], image.shape[0]
        # Initiate a list
        landmark_point = []
        # For each landmark in detected landmarks:
        for _, landmark in enumerate(landmarks.landmark):
            # Get the x and y positions of each landmark (percent * total width/height)
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
                
    @staticmethod
    def pre_process_landmark(landmark_list):
        '''
        :param landmark_list:
        :return:
        '''
        temp_landmark_list = copy.deepcopy(landmark_list)
        # ...
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        # ...
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))
        # ...
        max_value = max(list(map(abs, temp_landmark_list)))
        def normalize_(n):
            return n / max_value
        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    @staticmethod
    def pre_process_point_history(image, point_history):
        '''
        :param image: Handover image for image size only
        :param point_history: Handover point history list
        :return: Point history in normalized form
        '''
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)
        # ...
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]
            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
        # ...
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history





class Capture:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.cap.set(3, 1280)  # Setting webcam's image width
        self.cap.set(4, 720)  # Setting webcam's image height
        self.started = False
        self.bucket = None
        self.thread = None

    def start(self):
        if self.started:
            print(f"{Capture.__qualname__} thread already started.")
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
    # cam = pyvirtualcam.Camera(width=1280, height=720, fps=30)
    gesture = GestureDetection()
    gesture.start()
    capture = Capture()
    capture.start()
    running = True
    while running:
        if not isinstance(gesture.bucket_a, type(None)):
            print(gesture.bucket_a)
            gesture.bucket_a = None

        if isinstance(capture.bucket, np.ndarray):
            gesture.frame = capture.bucket

        if isinstance(gesture.bucket, np.ndarray):
            img = gesture.bucket

            cv.imshow('Gesture', img)

            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = cv.flip(img, 1)
            # cam.send(img)
            # cam.sleep_until_next_frame()

            if cv.waitKey(1) & 0xFF == ord('q'):
                sys.exit()
