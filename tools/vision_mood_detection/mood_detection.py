import time
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from threading import Thread
import os

class MoodDetection:

    def __init__(self, emotions_dict=None, model="models/fer", cascadeClassifier="models/haarcascade_frontalface_default"):

        # Check if given emotions_dict is dict type
        if isinstance(emotions_dict, dict):
            for key in emotions_dict.keys():
                # Uppercase the emotion
                emotions_dict[key]['emotion'] = str(emotions_dict.get(key).get('emotion')).upper()
        else:
            emotions_dict = {0: {"emotion": "ANGRY",
                                 "emoji": "😠"},
                             1: {"emotion": "DISGUST",
                                 "emoji":  "🤢"},
                             2: {"emotion": "FEAR",
                                 "emoji": "😱"},
                             3: {"emotion": "HAPPY",
                                 "emoji": "😃"},
                             4: {"emotion": "SAD",
                                 "emoji": "😞"},
                             5: {"emotion": "SURPRISE",
                                 "emoji": "😲"},
                             6: {"emotion": "NEUTRAL",
                                "emoji": "😐"}}

        self.emotions_dict = emotions_dict

        # Drop extension
        model = os.path.join(os.path.dirname(__file__), model)
        model_name = model
        try:
            # load model
            model = model_from_json(open(f"{model_name}.json", "r").read())
            model.load_weights(f'{model_name}.h5')
            self.model = model
        except:
            raise Exception(f"Error load mood detection models.\nGiven models: {model_name}.json {model_name}.h5")

        cascadeClassifier = os.path.join(os.path.dirname(__file__), cascadeClassifier)
        # Drop extension
        cascadeClassifier = str(cascadeClassifier).split(".")[0]
        try:
            self.face_haar_cascade = cv2.CascadeClassifier(f'{cascadeClassifier}.xml')
        except:
            raise Exception(f"Error load CascadeClassifier.\nGiven: {cascadeClassifier}.xml")

        self.frame = None
        self.bucket = None
        self.started = False
        self.return_emoji = False
        self.load_main = False
        self.time = 0
        self.thread = None


    def start(self):
        if self.started:
            print("mood runTime is already started")
        else:
            self.started = True
            self.thread = Thread(target=self.runTime, name='MoodDetectionThread', daemon=True)
            self.thread.start()

    def stop(self):
        if not self.started:
            print("mood runTime is already stopped")
        else:
            self.started = False

    def runTime(self, frame):
        self.bucket = None
        startTime = time.time()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            predictions = self.model.predict(img_pixels)
            # find max indexed array
            max_index = np.argmax(predictions[0])

            # emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            """
            if self.return_emoji:
                self.bucket = self.emotions_dict.get(max_index).get('emoji')
            else:
                self.bucket = self.emotions_dict.get(max_index).get('emotion')
            """""
            # self.bucket = self.emotions_dict.get(max_index).get('emotion')
            self.bucket = {
                "predictions": predictions.tolist(),
                "dominant_index": max_index
                #"dominant_index": (max_index, predictions[max_index])
            }
            self.time = round((time.time()-startTime)*1000)
