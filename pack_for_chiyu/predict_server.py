from flask import Flask
from flask_socketio import SocketIO
import base64
from io import BytesIO
from PIL import Image
from PIL import ImageOps
import color_detection as cd
import the_ultimate_function
import time
import numpy as np
import pack_for_chiyu.data_organizer as do
import pack_for_chiyu.recorder as rd
import mediapipe as mp
import keras


app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")
user = ""
image_path = ""
section_width = 0
scan_area = 0
center_points = ()


@socketio.on("rotation")
def rotation(image):
    try:
        image = Image.open(BytesIO(base64.b64decode(image)))
        image = ImageOps.mirror(image)  # Flip the image horizontally
        image = np.array(image)
        predictedResult, probabilities = the_ultimate_function.picture_in_result_out(
            image
        )
        print("predictedResult: ", predictedResult, "probabilities: ", probabilities)
        result = {"predictedResult": predictedResult, "probabilities": probabilities}
        socketio.emit("rotation", result)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on("receive_image")
def handle_receive_image():
    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
        socketio.emit("receive_image", encoded_image)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on("save_image")
def handle_save_image(image):
    try:
        global section_width
        global scan_area
        global center_points
        image = Image.open(BytesIO(base64.b64decode(image)))
        image = cd.draw_banner(image)
        image, section_width, scan_area = cd.draw_3x3_grid(image)

        image = Image.fromarray(image)
        image.save(image_path)

    except Exception as e:
        print(f"Error: {e}")


@socketio.on("initialize_cube_color")
def handle_initialize_cube_color():
    try:
        image = Image.open(image_path)
        start_time = time.time()

        records = cd.predict_color(image, section_width, scan_area, user)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Function execution time: {execution_time} seconds")
        socketio.emit("return_cube_color", records)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on("init_color_dataset")
def init_color_dataset(color):
    global section_width
    global scan_area
    try:
        image = Image.open(image_path)
        cd.init_color_dataset(user, color, image, section_width, scan_area)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("join")
def handle_join(user_info):
    global user
    global image_path
    user = user_info
    image_path = f"images/{user}.jpeg"


@socketio.on("clear_color_dataset")
def handle_clear_color_dataset():
    cd.clear_color_dataset(user)


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
