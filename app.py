from flask import Flask
from flask_socketio import SocketIO
import base64
from io import BytesIO
from PIL import Image
import color_detection as cd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')
image_path = 'image.jpeg'


@socketio.on('receive_image')
def handle_receive_image(msg):
    try:
        with open(image_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
        socketio.emit('receive_image', encoded_image)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on('save_image')
def handle_save_image(msg):
    try:
        image = Image.open(BytesIO(base64.b64decode(msg)))
        image = cd.draw_banner(image)
        image, section_width, scan_area = cd.draw_3x3_grid(image)
        records = []
        for i in range(9):
            records.append(False)
        counter = 0
        for color in cd.color_list:
            print(f'section {counter}')
            image, records = cd.process_image(image=image, color=color, section_width=section_width,
                                              scan_area=scan_area, records=records)
            counter += 1
        image = Image.fromarray(image)
        image.save(image_path)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
