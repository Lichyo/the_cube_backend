from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, send
import base64
from io import BytesIO
from PIL import Image
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')
image_path = 'image.jpeg'


def generate_frame():
    while True:
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/latest_image')
# def latest_image():
    # return send_file(image_path, mimetype='image/jpeg')


@app.route('/video')
def video():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('receive_image')
def handle_message(msg):
    try:
        # print('Receiving image')
        with open(image_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
        send(encoded_image)
        print('Image sent')

    except Exception as e:
        print(f"Error: {e}")


@socketio.on('save_image')
def handle_message(msg):
    try:
        image = Image.open(BytesIO(base64.b64decode(msg)))
        image.save(image_path)
        socketio.emit('image_saved')

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
