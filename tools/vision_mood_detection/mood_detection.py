import time
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import os


class MoodDetection:

    def __init__(self, log, model="models/fer", cascadeClassifier="models/haarcascade_frontalface_default"):

        self.log = log
        self.log.debug("Init")

        model = os.path.join(os.path.dirname(__file__), model)
        model_name = model

        try:
            # load model
            if os.path.exists(f"{model_name}.json"):
                model = model_from_json(open(f"{model_name}.json", "r").read())
            else:
                raise Exception(f"Error load mood detection model.\nGiven models: {model_name}.json")

            if os.path.exists(f"{model_name}.h5"):
                model.load_weights(f'{model_name}.h5')
            else:
                raise Exception(f"Error load mood detection model.\nGiven models: {model_name}.h5")

            self.model = model
        except:
            raise Exception(f"Error load mood detection models.\nGiven models: {model_name}.json {model_name}.h5")

        cascadeClassifier = os.path.join(os.path.dirname(__file__), cascadeClassifier)

        if os.path.exists(f"{cascadeClassifier}.xml"):
            try:
                self.face_haar_cascade = cv2.CascadeClassifier(f'{cascadeClassifier}.xml')
            except:
                raise Exception(f"Error load CascadeClassifier.\nGiven: {cascadeClassifier}.xml")
        else:
            raise Exception(f"CascadeClassifier not found!\nGiven: {cascadeClassifier}.xml")

        self.frame = None
        self.bucket = None
        self.started = False
        self.return_emoji = False
        self.load_main = False
        self.time = 0
        self.thread = None

    def runTime(self, frame):
        try:
            self.bucket = None
            startTime = time.time()
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_img = np.uint8(gray_img)
            faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
            self.log.debug(f"Faces detected: {len(faces_detected)}")
            print(faces_detected == ())
            if faces_detected == ():
                self.bucket = {
                    "predictions": [],
                    "dominant_index": -1
                }
            for (x, y, w, h) in faces_detected:
                roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                predictions = self.model.predict(img_pixels)
                # find max indexed array
                max_index = np.argmax(predictions[0])

                self.bucket = {
                    "predictions": predictions.tolist(),
                    "dominant_index": max_index
                    #"dominant_index": (max_index, predictions[max_index])
                }
                self.time = round((time.time()-startTime)*1000)
        except Exception as e:
            self.log.warning(f"{e}")
