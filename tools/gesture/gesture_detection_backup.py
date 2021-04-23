import cv2 as cv
import mediapipe as mp
import copy
import threading
import numpy as np
import time
import setproctitle

class GestureDetection:

    def __init__(self):

        # Init MediaPipe parameters
        self.mp_max_num_hands = 1
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

    def start(self):
        if self.started:
            print("Gesture detection thread already started.")
        else:
            print("Starting gesture detection")
            self.started = True
            self.thread = threading.Thread(target=self.runTime, name='GestureThread', daemon=True)
            self.thread.start()

    def stop(self):
        if not self.started:
            print("Gesture detection thread already stopped.")
        else:
            self.started = False

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
        Calculate the corners of teh bounding rectangle covering all landmark points
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
    def draw_landmarks(image, cx, cy, landmarks, handedness):
        """ Drawing the landmarks on the image

        The function takes the image, ...

        :param image: The current frame
        :param cx: The x coordnate of the center of the palm
        :param cy: The x coordnate of the center of the palm
        :param landmarks: The hand landmarks as detected by Mediapipe
        :param handedness: The handedness as detected by Mediapipe
        :return: The image with drawings added
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
            # landmark_z = landmark.z  # Depth of the landmark, not in use
            # Append the landmark coordinate and print the outline
            landmark_point.append((landmark_x, landmark_y))
            if index == 0:  # WRIST
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 1:  # THUMB_CMC
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 2:  # THUMB_MCP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 3:  # THUMB_IP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 4:  # THUMB_TIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 5:  # INDEX_FINGER_MCP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 6:  # INDEX_FINGER_PIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 7:  # INDEX_FINGER_DIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 8:  # INDEX_FINGER_TIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 9:  # MIDDLE_FINGER_MCP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 10:  # MIDDLE_FINGER_PIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 11:  # MIDDLE_FINGER_DIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 12:  # MIDDLE_FINGER_TIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 13:  # RING_FINGER_MCP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 14:  # RING_FINGER_PIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 15:  # RING_FINGER_DIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 16:  # RING_FINGER_TIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 17:  # PINKY_MCP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 18:  # PINKY_PIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 19:  # PINKY_DIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 20:  # PINKY_TIP
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        # Create the connecting lines
        if len(landmark_point) > 0:
            # THUMB
            cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
            cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)
            # INDEX_FINGER
            cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
            cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
            cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)
            # MIDDLE FINGER
            cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
            cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
            cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)
            # RING FINGER
            cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
            cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
            cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)
            # PINKY
            cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
            cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
            cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)
            # PALM
            cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
            cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
            cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
            cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
            cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
            cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
            cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

        # ...
        if len(landmark_point) > 0:
            # handedness.classification[0].index
            # handedness.classification[0].score
            cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
            cv.putText(image, handedness.classification[0].label[0],
                       (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                       2, cv.LINE_AA)  # label[0]: ...
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
                         (0, 255, 0), 2)
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

        recognized_hand_gesture = None
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

        if thumb_state == 1 and index_state == 1 and middle_state == 1 and ring_state == 1 and pinky_state == 1:
            recognized_hand_gesture = "FIVE"
        elif thumb_state == 0 and index_state == 1 and middle_state == 1 and ring_state == 1 and pinky_state == 1:
            recognized_hand_gesture = "FOUR"
        elif thumb_state == 1 and index_state == 1 and middle_state == 1 and ring_state == 0 and pinky_state == 0:
            recognized_hand_gesture = "THREE"
        elif thumb_state == 1 and index_state == 1 and middle_state == 0 and ring_state == 0 and pinky_state == 0:
            recognized_hand_gesture = "TWO"
        elif thumb_state == 0 and index_state == 1 and middle_state == 0 and ring_state == 0 and pinky_state == 0:
            recognized_hand_gesture = "ONE"
        elif thumb_state == 0 and index_state == 0 and middle_state == 0 and ring_state == 0 and pinky_state == 0:
            recognized_hand_gesture = "ZERO"
        else:
            recognized_hand_gesture = None

        return thumb_up, recognized_sum, recognized_hand_gesture

    def runTime(self):
        setproctitle.setproctitle(threading.currentThread().name)
        while self.started:
            if isinstance(self.frame, np.ndarray):
                startTime = time.time()
                image = cv.flip(self.frame, 1)
                debug_image = copy.deepcopy(image)

                # Change to RGB image and detect hands
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                results = self.hands.process(image)

                # When hand detected
                if not isinstance(results.multi_hand_landmarks, type(None)):
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                          results.multi_handedness):

                        # Calculate center of palm
                        cx, cy = self.calc_palm_moment(debug_image, hand_landmarks)

                        # Calculate bounding rectangle of hand
                        brect = self.calc_bounding_rect(debug_image, hand_landmarks)

                        # Draw landmarks and bounding rectangle
                        debug_image = self.draw_landmarks(debug_image, cx, cy,
                                                     hand_landmarks, handedness)
                        debug_image = self.draw_bounding_rect(self.mp_use_brect, debug_image, brect)

                        # Test simple number recognition directly with Mediapipe
                        thumb_up, recognized_sum, recognized_hand_gesture = self.recognize_hand_gesture(image, cx, cy,
                                                                         hand_landmarks, handedness)
                        self.bucket = {'thumb_up': thumb_up,
                                        'recognized_sum': recognized_sum,
                                        'recognized_hand_gesture': recognized_hand_gesture}

                        debug_image = self.print_on_image(debug_image, "Hand Gesture:", recognized_hand_gesture, 10, 60)
                        debug_image = self.print_on_image(debug_image, "Sum:", recognized_sum, 10, 90)
                        debug_image = self.print_on_image(debug_image, "Thumb-up:", thumb_up, 10, 120)
                    self.time = round((time.time() - startTime)*1000)
                    self.debug_frame = debug_image

