# ASAP

As part of the Postgraduate Artificial Intelligence course offered by the VUB and the Erasmushoogeschool in Brussels, we have selected to develop an interactive and ML-driven addition to videoconferencing. With virtual meetings on the rise, developing and improving the interaction between human participant and the videoconference is becoming an interesting field of study. We are proposing a combination of intelligent interfaces to increase this interaction, using trained Neural Networks and AI online services. Each component of the application is described below and interesting links are added to showcase the ideas and (Github) libraries we have built upon.

## Installation

First: 
```shell
# installing venv 
python3 -m pip install --user virtualenv
# creating virtual env
python3 -m venv env
# activating virtual env
# ------FOR LINUX/MAC---------#
source env/bin/activate
# -------FOR WINDOWS----------#
.\env\Scripts\activate

python3 -m pip install --usere -r requirements.txt
```

Then open main.py in IDE

```python
# Make sure that the google credentials are correct.
asap = LoadFeatures(google_credentials="/folder_to/asap-309508-7398a8c4473f.json")
# Available features ["bgMask", "stt", "visionMd","gesture", "vidGet", "vidShow", "virtualCam]
asap.start(features=["stt", "vidGet", "gesture" , "visionMd", "bgMask", "virtualCam"])
```

## Requirements

## Components

### Gesture Recognition

