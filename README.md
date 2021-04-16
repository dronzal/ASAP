# ASAP

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
