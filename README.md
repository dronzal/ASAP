# ASAP

`A:rne-S:imon-A:lexis-P:ieter-Jan`

As part of the Postgraduate Artificial Intelligence course offered by the VUB and the Erasmushoogeschool in Brussels, we have selected to develop an interactive and ML-driven addition to videoconferencing. With virtual meetings on the rise, developing and improving the interaction between human participant and the videoconference is becoming an interesting field of study. We are proposing a combination of intelligent interfaces to increase this interaction, using trained Neural Networks and AI online services. Each component of the application is described below and interesting links are added to showcase the ideas and (Github) libraries we have built upon.

<img src="assets/asap.jpg" width="720">

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
asap = ASAP(google_credentials="/folder_to/asap-309508-7398a8c4473f.json")
# And start the ASAP application
asap.start()
```

## Application

The application needed to be designed to incorporate the results of different features. After initial performance issues, the following design has proven to be operational. A Threadpool executor controls three of the threads that require the webcam frames as input. Speech recognition runs as thread connected to a Google service. Three further threads control video capture, display and the virtual camera. Finally the client side actions are handled in a thread and another websocket thread takes care of the communication between clients / participants.

<img src="assets/asap_uml.jpg" width="1080">

## Components

Please click for further details:

<details>
<summary>Gesture Recognition</summary>
<p><p />
The Gesture Recognition component makes use of the Google-developed Mediapipe framework for hand recognition. The hand landmarks are used as coordinates that can be fed into a neural network to recognize hand gestures (and finger gestures).
 
#### Disclaimer
The code makes use of existing libraries and is based in large parts on the following repositories:
* It uses the Mediapipe framework published by Google: https://mediapipe.dev/
* It is based on code published by Kazuhito00 on Github: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe/blob/main/README_EN.md
published under Apache 2.0 licence: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe/blob/main/LICENSE
* It uses hand gestures trained by kinivi, his neural network design and Jupyter notebook from Github: https://github.com/kinivi/tello-gesture-control published under Apache 2.0 licence: https://github.com/kinivi/tello-gesture-control/blob/main/LICENSE

#### Machine Learning (ML) / Artificial Intelligence (AI)
ML/AI is used in this component to identify hand gestures in webcam images. The Google mediapipe framework allows to identify one or both hand(s) and returns the coordinates of hand, fingers and joints.
<img src="assets/gesturesMediapipe.png" width="720">
<p />
These coordinates are transformed in three steps: from the Mediapipe landmarks to relative coordinates, then the x/y components are separated and the resulting 
variables normalized. 
<img src="assets/gesturesXY.png" width="720">
<p />
The neural network is a simple one with three fully connected RELU layers followed by a Softmax translation to the discrete results (originally 8, for our purposes one added): <br />
<img src="assets/gesturesNN.jpg" width="720">
<p />
The model training is executed in a Jupyter notebook. The neural network is fed with the normalized coordinates and the labels that indicate the hand gesture. The model achieves around 99.96% accuracy in around 100 epochs. The saved model is transformed into a tflite model and used to infer the hand gestures from the webcam images (pre-evaluated through Mediapipe)

#### Further Interesting Links
* Mediapipe Demo
  * <a href="https://mediapipe.dev/demo/holistic_remote/" target="blank">Holistic Mediapipe demo</a>
</p>
</details>

<details>
<summary>Voice Commands / Text-2-Speech</summary>
<p><p />
...
</p>
</details>

<details><summary>Mood Detection</summary>
<p><p />
...
</p>
</details>

<details><summary>Dynamic Background</summary>
<p><p />
...
</p>
</details>

## Features

### Voice / Gesture Commands
 
Both voice commands as well as gestures are used to interact with the video conferencing tool and with the videoconference participants. Currently the following commands are supported:

#### Command Mode
* Move into command mode: Show two hands to the webcam
* Cancel command mode: Show two hands again
 
#### Audio
* Mute the microphone: Show flat palm of one hand
* Un-mute the microphone: Make an upward fist
* Increase the volume: Index finger up (and thumb to the side)
* Decrease the volume: Index finger down (and thumb to the side)
 
#### Video
* Black out the Camera: Point fist at the camera
* Return to Webcam display: Show upwards fist (same as un-mute)
 
#### Background
* Change the Background one-forward: Fist with thumb to one side
* Change the Background one-backward: Fist with thumb to the other side

#### Voting
* Begin a voting process: Victory sign
* Set the number of options: 
   * Indicate yes/no question: Thumns-up sign or
   * Show number of fingers [1..5]
* Confirm the number of options displayed: OK sign

--- Explain the options to the group ---

* Start the voting: Victory sign
* Cast your vote: 
   * In case of yes/no: Show thumbs-up or
   * In case of more options: Show number with your fingers
* Confirm vote: OK sign

[not implemented] 
--- Once all participants have voted, display the result on all screens ---

### Mood Server
...

### The End

<img src="assets/asap_end.png" width="200">
