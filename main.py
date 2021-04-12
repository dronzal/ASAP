from tools import speechToText as STT
from tools.bgmask import bg_mask as bgm
from tools.vision_mood_detection import mood_detection as MD
from tools import videoTools
import numpy as np
from threading import Thread


class LoadFeatures:

    def __init__(self):
        # available features
        self.feature_list = ["bgMask", "stt", "visionMd", "vidGet", "vidShow"]
        self.features_selected = []
        # Load all features
        self.stt = STT.SpeechToText()
        self.bgMask = bgm.BackgroundMask(model='tools/bgmask/models/segmentation/deconv_bnoptimized_munet.h5',
                                         bg_im_path='tools/bgmask/bg_im/')
        self.visionMd = MD.MoodDetection(model='tools/vision_mood_detection/models/fer.h5',
                                         cascadeClassifier='tools/vision_mood_detection/models/haarcascade_frontalface_default.xml')
        self.vidGet = videoTools.VideoGetter()
        self.vidShow = videoTools.VideoShower()

        # Bool for continous runTime
        self.started = False

        # Result buckets
        self.result_mood = None
        self.result_stt = None

        self.vid_frame = None

        self.stt_started = False
        self.bgMask_started = False
        self.visionMd_started = False
        self.vidShow_started = False
        self.vidGet_started = False

        self.thread = None

    def start(self, features=None):
        if isinstance(features, type(None)):
            # No features are selected, so choose all the features that are available
            self.features_selected = self.feature_list
        else:
            # Check if chosen features exists and return them in a list if true.
            self.features_selected = [feature for feature in features if feature in self.feature_list]

        for feature in self.features_selected:
            # Try the code
            try:
                # The exec() method executes the dynamically created program, which is either a string or a code object.
                exec(f"self.{feature}_started = True")
                exec(f"self.{feature}.start()")

            # Catch the fault and raise an Exception Error.
            except:
                raise Exception(f'Feature not available.\nGiven: {feature}.\nAvailable: {self.feature_list}')

        if not self.started:
            self.started = True
            self.thread = Thread(target=self.runTime, name='AsapThread', daemon=True)
            self.thread.start()

    def stop(self, features=None):
        if isinstance(features, type(None)):
            # No features are selected, so choose all the features that are available
            features_stop = self.feature_list
        else:
            # Check if chosen features exists and return them in a list if true.
            features_stop = [feature for feature in features if feature in self.feature_list]

        for feature in features_stop:

            # Try the code
            try:
                # The exec() method executes the dynamically created program, which is either a string or a code object.
                exec(f"self.{feature}_started = False")

            # Catch the fault and raise an Exception Error.
            except:
                raise Exception(f'Feature not available.\nGiven: {feature}.\nAvailable: {self.feature_list}')

    def runTime(self):
        while self.started:
            if self.bgMask_started and isinstance(self.vidGet.bucket, np.ndarray):
                self.bgMask.frame = self.vidGet.bucket

            if self.visionMd_started and isinstance(self.vidGet.bucket, np.ndarray):
                self.visionMd.frame = self.vidGet.bucket
                self.result_mood = self.visionMd.bucket

            if self.vidShow_started and isinstance(self.vidGet.bucket, np.ndarray):
                self.vidShow.frame = self.bgMask.bucket

            if self.stt_started:
                if not isinstance(self.stt.bucket, type(None)):
                    self.result_stt = str(self.stt.bucket)
                    self.stt.bucket = None


if __name__ == "__main__":
    asap = LoadFeatures()
    asap.start(features=["stt", "visionMd", "bgMask", "vidGet", "vidShow"])
    while True:
        if isinstance(asap.result_mood, str) and isinstance(asap.result_stt, str):
            print(f"Results ASAP:\nSpeechToText: {asap.result_stt}\nMood: {asap.result_mood}\n\n")
            asap.result_stt = None
            asap.result_mood = None
