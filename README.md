# ASAP

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

## Requirements

## Components

<details>
<summary>Gesture Recognition</summary>
<p><p />
The Gesture Recognition component makes use of the Google-developed mediapipe model for hand recognition. The hand landmarks are used as coordinates that can be fed into a neural network to recognize hand gestures and finger gestures

#### Links
* Mediapipe
  * <a href="https://mediapipe.dev/demo/holistic_remote/" target="blank">Holistic Mediapipe demo</a>

</p></details>

<details><summary>Voice Commands / Text-2-Speech</summary>
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

## (Potential) Features

- [x] Voting workflow
- [ ] Play sound when average mood exceeds threshold

