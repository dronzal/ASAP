import asyncio
import json
import logging
import websockets
import datetime
import functools
import numpy as np

logging.basicConfig()

USERS = set()
MEASUREMENTS = {}
LOGENTRIESQUEUE = []


def get_dominant_mood():
    validMeasurements = []
    dominantMood = -1
    for key in MEASUREMENTS:
        if MEASUREMENTS[key].timestamp > datetime.datetime.now() - datetime.timedelta(seconds=30):
            validMeasurements.extend(MEASUREMENTS[key].measurement)

    totalMeasurements = [functools.reduce(lambda a, b: a + b, x) for x in zip(*validMeasurements)]
    totalMeasurements = list(map(lambda x: x / len(validMeasurements), totalMeasurements))
    if len(totalMeasurements) > 0:
        dominantMood = np.argmax(list(totalMeasurements))
    dominantMood = str(dominantMood)
    print(dominantMood)
    return json.dumps({"type": "mood", "measurements": dominantMood})


def get_conversation():
    conversation = list(LOGENTRIESQUEUE)
    if len(conversation) > 0:
        conversation = list(map(lambda le: le.to_json(), conversation))
    else:
        conversation = "Empty log..."
    return json.dumps({"type": "conversation", "conversation": conversation}, default=str)


def users_event():
    return json.dumps({"type": "users", "count": len(USERS)})


async def notify_state(message):
    if USERS:  # asyncio.wait doesn't accept an empty list
        await asyncio.wait([user.send(message) for user in USERS])


async def notify_users():
    if USERS:  # asyncio.wait doesn't accept an empty list
        message = users_event()
        await asyncio.wait([user.send(message) for user in USERS])


async def register(websocket):
    USERS.add(websocket)
    print("New user!")
    await notify_users()


async def unregister(websocket):
    USERS.remove(websocket)
    await notify_users()


def registerMeasurement(name, measurement):
    MEASUREMENTS[name] = measurement


def registerLogEntry(logEntry):
    LOGENTRIESQUEUE.append(logEntry)


def reset():
    global LOGENTRIESQUEUE
    global MEASUREMENTS
    LOGENTRIESQUEUE = []
    MEASUREMENTS = {}


class MoodMeasurement:
    timestamp = None
    measurement = None

    def __init__(self, measurement):
        self.timestamp = datetime.datetime.now()
        self.measurement = measurement


class LogEntry:
    timestamp = None
    message = None
    name = None

    def __init__(self, log_entry, name):
        self.timestamp = datetime.datetime.now()
        self.message = log_entry
        self.name = name

    def to_json(self):
        return {"timestamp": self.timestamp, "message": self.message, "name": self.name}



async def handler(websocket, path):
    await register(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)

            if data["action"] == "mood":
                name = data["name"]
                value = data["payload"]
                measurement = MoodMeasurement(value)
                registerMeasurement(name, measurement)
                await notify_state("ok")
            elif data["action"] == "log_entry":
                name = data["name"]
                value = data["payload"]
                log_entry = LogEntry(value, name)
                registerLogEntry(log_entry)
                await notify_state("ok")
            elif data["action"] == "get_dominant_mood":
                message = get_dominant_mood()
                await notify_state(message)
            elif data["action"] == "get_conversation":
                message = get_conversation()
                await notify_state(message)
            elif data["action"] == "reset":
                reset()
                await notify_state("ok")
            else:
                logging.error("unsupported event: %s", data)
    finally:
        await unregister(websocket)


start_server = websockets.serve(handler, "0.0.0.0", 6789)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()