#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from google.cloud import speech
import pyaudio
import queue
import os
import threading
import setproctitle
from time import time


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate=16000, chunk=int(16000/10)):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
                                                format=pyaudio.paInt16,
                                                # The API currently only supports 1-channel (mono) audio
                                                # https://goo.gl/z757pE
                                                channels=1,
                                                rate=self._rate,
                                                input=True,
                                                frames_per_buffer=self._chunk,
                                                # Run the audio stream asynchronously to fill the buffer object.
                                                # This is necessary so that the input device's buffer doesn't
                                                # overflow while the calling thread makes network requests, etc.
                                                stream_callback=self._fill_buffer,
                                                            )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


class SpeechToText:

    def __init__(self, google_credentials, rate=16000, chunk=int(16000/10), language_code="en-UK"):

        if os.path.isfile(google_credentials):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= google_credentials
        else:
            raise Exception(f"Google Credential file neede. Given {google_credentials}")

        self.bucket = None
        self.rate = rate
        self.chunk = chunk
        self.language_code= language_code
        self.started = False
        self.thread = None
        self.counter = 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread.join()

    def listen_print_loop(self, responses):
        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            overwrite_chars = " " * (num_chars_printed - len(transcript))

            if not result.is_final:
                num_chars_printed = len(transcript)
            else:
                self.bucket = transcript + overwrite_chars
                num_chars_printed = 0

    def runTime(self):
        setproctitle.setproctitle(threading.currentThread().name)
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                          sample_rate_hertz=self.rate,
                                          language_code=self.language_code)

        self.streaming_config = speech.StreamingRecognitionConfig(config=self.config,
                                                             interim_results=True)
        while self.started:
            with MicrophoneStream(self.rate, self.chunk) as stream:
                audio_generator = stream.generator()
                requests = (speech.StreamingRecognizeRequest(audio_content=content)
                            for content in audio_generator)
                responses = self.client.streaming_recognize(self.streaming_config, requests)
                self.listen_print_loop(responses)

    def start(self):
        if self.started:
            print("sst runTime is already started")
        else:
            self.started = True
            self.thread = threading.Thread(target=self.runTime, name='MoodDetectionThread', daemon=True)
            self.thread.start()

    def stop(self):
        if not self.started:
            print("sst runTime is already stopped")
        else:
            self.started = False


if __name__ == "__main__":
    STT = SpeechToText(google_credentials="/home/puyar/Documents/Playroom/asap-309508-7398a8c4473f.json")
    STT.start()
    while True:
        if STT.bucket:
            result = str(STT.bucket).lower()
            STT.bucket = None
            print(result)

