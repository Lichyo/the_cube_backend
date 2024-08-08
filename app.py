from flask import Flask
from flask_socketio import SocketIO, send
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import color_detection as cd
import cv2
import numpy as np

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
        print('Image sent')
    except Exception as e:
        print(f"Error: {e}")


@socketio.on('save_image')
def handle_save_image(msg):
    try:
        image = Image.open(BytesIO(base64.b64decode(msg)))
        # Add text to the top-right corner
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = "Cy_Cube"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        position = (image.width - text_width - 10, 10)
        draw.text(position, text, font=font, fill="white")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        height, width, _ = image_cv.shape

        color = (0, 0, 0)  # Green color in BGR
        thickness = 2

        section_width = (width - 20) // 3
        section_height = section_width

        start_x = 10
        end_x = width - 10
        start_y = int(height / 2 - section_width * 1.5)
        end_y = int(height / 2 + section_width * 1.5)

        for i in range(0, 4):
            x = i * section_width + start_x
            cv2.line(image_cv, (x, start_y), (x, end_y), color, thickness)

        for i in range(0, 4):
            y = start_y + i * section_height
            cv2.line(image_cv, (start_x, y), (end_x, y), color, thickness)

        image_cv = cd.process_image(img=image_cv, color=cd.white)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_cv)

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
