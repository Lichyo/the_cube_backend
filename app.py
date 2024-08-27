from flask import Flask
from flask_socketio import SocketIO
import base64
from io import BytesIO
from PIL import Image
import color_detection as cd
import json
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')
user_define_colors = {
    'white': [0, 0, 0],
    'yellow': [0, 0, 0],
    'red': [0, 0, 0],
    'orange': [0, 0, 0],
    'blue': [0, 0, 0],
    'green': [0, 0, 0]
}
user = ""
image_path = ""
section_width = 0
scan_area = 0
center_points = ()


def print_user_define_colors():
    for key, value in user_define_colors.items():
        print(f"{key}: {value}")
    print()


@socketio.on('receive_image')
def handle_receive_image():
    try:
        with open(image_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
        socketio.emit('receive_image', encoded_image)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on('save_image')
def handle_save_image(image):
    try:
        global section_width
        global scan_area
        global center_points
        image = Image.open(BytesIO(base64.b64decode(image)))
        image = cd.draw_banner(image)
        image, section_width, scan_area, center_points = cd.draw_3x3_grid(image)

        image = Image.fromarray(image)
        image.save(image_path)

    except Exception as e:
        print(f"Error: {e}")


@socketio.on('initialize_cube_color')
def handle_initialize_cube_color():
    try:
        image = Image.open(image_path)
        records = []
        for i in range(9):
            records.append(False)
        for color in cd.color_list:
            image, records = cd.process_image(image=image, color=color, section_width=section_width,
                                              scan_area=scan_area, records=records)
        print(records)
        socketio.emit('return_cube_color', records)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on('initialize_user_define_cube_color')
def handle_initialize_user_define_cube_color(color):
    image = Image.open(image_path)
    points = cd.find_section_range(scan_area, section_width)
    array = []
    for i in range(0, 9):
        x, y = points[i]
        center_color = cd.get_center_color_rgb(image, (x, y))
        array.append(center_color)
    mean = np.mean(array, axis=0).astype(int)
    user_define_colors[color] = mean
    print_user_define_colors()
    flag = True
    for key, value in user_define_colors.items():
        if np.array_equal(value, [0, 0, 0]):
            flag = False
    if flag:
        data = {
            "colors": {k: v.tolist() for k, v in user_define_colors.items()}
        }
        with open(f'user_define_colors/{user}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.flush()
        socketio.emit('initialize_user_define_cube_color', "success")


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('join')
def handle_join(user_info):
    global user
    global image_path
    user = user_info
    image_path = f"images/{user}.jpeg"


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
