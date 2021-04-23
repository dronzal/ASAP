from tools import speechToText as STT
from tools.bgmask import bg_mask as BGM
from tools.vision_mood_detection import mood_detection as MD
from tools import videoTools
from tools.gesture import gesture_detection as GD
from tools import virtualDevice as VD
import numpy as np
import threading
import setproctitle
import continuous_threading

class LoadFeatures:

    def __init__(self, google_credentials, frame_height=640, frame_width=380):
        # available features
        self.feature_list = ["bgMask", "stt", "visionMd", "gesture", "vidGet", "vidShow", "virtualCam"]
        self.features_selected = []
        # Load all features
        self.stt = STT.SpeechToText(google_credentials=google_credentials)
        self.bgMask = BGM.BackgroundMask(model='tools/bgmask/models/segmentation/deconv_bnoptimized_munet.h5',
                                         bg_im_path='tools/bgmask/bg_im/',
                                         frame_height=frame_height,
                                         frame_width=frame_width)
        self.visionMd = MD.MoodDetection(model='tools/vision_mood_detection/models/fer.h5',
                                         cascadeClassifier='tools/vision_mood_detection/models/haarcascade_frontalface_default.xml')
        self.virtualCam = VD.VirtualDevice(frame_height=frame_height,
                                           frame_width=frame_width)
        self.gesture = GD.GestureDetection()
        self.vidGet = videoTools.VideoGetter()
        self.vidShow = videoTools.VideoShower()

        # Bool for continous runTime
        self.started = False

        self.vid_frame = None
        self.virtualCam_started = False
        self.stt_started = False
        self.bgMask_started = False
        self.visionMd_started = False
        self.vidShow_started = False
        self.vidGet_started = False
        self.gesture_started = False
        self.thread = None
        self.frame_height = frame_height
        self.frame_width = frame_width

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
                print(f'Feature {feature} started')

            # Catch the fault and raise an Exception Error.
            except:
                raise Exception(f'Feature not available.\nGiven: {feature}.\nAvailable: {self.feature_list}')

        if not self.started:
            self.started = True
            self.thread = continuous_threading.ContinuousThread(target=self.runTime, name='AsapThread', daemon=True)
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
        setproctitle.setproctitle(threading.current_thread().name)
        while self.started:
            if isinstance(self.vidGet.bucket, np.ndarray):
                # If bgMask feature is started, give vidget.bucket as inputfeed. (self.vidGet.bucket are captured frames from device)
                if self.bgMask_started:
                    self.bgMask.frame = self.vidGet.bucket

                if self.vidShow_started:
                    self.vidShow.frame = self.bgMask.bucket

                if self.gesture_started:
                    self.gesture.frame = self.vidGet.bucket
                    if not isinstance(self.gesture.bucket, type(None)):
                        print("Gesture", self.gesture.bucket)
                        self.gesture.bucket = None

                if self.visionMd_started:
                    self.visionMd.frame = self.vidGet.bucket
                    if not isinstance(self.visionMd.bucket, type(None)):
                        print("Mood:", self.visionMd.bucket)
                        self.visionMd.bucket = None

                if self.virtualCam_started:
                    self.virtualCam.frame = self.bgMask.bucket

            if self.stt_started:
                if not isinstance(self.stt.bucket, type(None)):
                    print('STT', str(self.stt.bucket))
                    self.stt.bucket = None


if __name__ == "__main__":

    setproctitle.setproctitle("ASAP")
    # Load all the features, specify where the google_credentials are stored
    asap = LoadFeatures(google_credentials="/home/puyar/Documents/Playroom/asap-309508-7398a8c4473f.json")
    # Available features ["bgMask", "stt", "visionMd","gesture", "vidGet", "vidShow", "virtualCam]
    asap.start(features=["stt", "vidGet", "gesture" , "visionMd", "bgMask", "virtualCam"])

