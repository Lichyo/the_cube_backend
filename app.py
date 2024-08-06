from flask import Flask
from flask_socketio import SocketIO, send
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')
image_path = 'image.jpeg'


@socketio.on('receive_image')
def handle_message(msg):
    try:
        with open(image_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
        socketio.emit('receive_image', encoded_image)
        print('Image sent')

    except Exception as e:
        print(f"Error: {e}")


@socketio.on('save_image')
def handle_message(msg):
    try:
        image = Image.open(BytesIO(base64.b64decode(msg)))

        # Add text to the top-right corner
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=20)
        text = "Cy_Cube"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        # text_height = bbox[3] - bbox[1]
        position = (image.width - text_width - 10, 10)
        draw.text(position, text, font=font, fill="white")

        image.save(image_path)
        socketio.emit('save_image', 'image_saved')

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
