import os
import socketio
import eventlet
from flask import Flask
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tensorflow as tf

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

sio = socketio.Server()
app = Flask(__name__)

speed_limit = 30


def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (150, 50))
    img = img / 255
    return img


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


@sio.on('telemetry')
def telemetry(sid, data):
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    speed = float(data['speed'])
    throttle = 1.0 - speed/speed_limit
    send_control(steering_angle, throttle)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })


if __name__ == '__main__':
    model = load_model('3model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
