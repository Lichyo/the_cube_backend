from flask import Flask
from flask_socketio import SocketIO
import base64
from io import BytesIO
from PIL import Image
from PIL import ImageOps
import color_detection as cd
import time
import numpy as np
import pack_for_chiyu.data_organizer as do
import pack_for_chiyu.recorder as rd
import mediapipe as mp
import keras

recorder = rd.Recorder()
organizer = do.DataOrganizer()
mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法
lstmModel = keras.models.load_model("pack_for_chiyu/lstm_2hand_model.keras")
showResult = "wait"
predictFrequence = 3
predictCount = 0
hands = mpHandsSolution.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
resultsList = [
    "B'",
    "B",
    "D'",
    "D",
    "F ",
    "F'",
    "L'",
    "L",
    "R",
    "R'",
    "U",
    "U'",
    "Stop",
    "wait",
]
currentFeature = []  # 目前畫面的資料
continuousFeature = []  # 目前抓到的前面
missCounter = 0
maxMissCounter = 5
lastResult = 13
acceptableProbability = 0.7


# -----------


def imageHandPosePredict(RGBImage):
    global continuousFeature
    global showResult
    global predictCount
    global hands
    global lastResult
    if not hasattr(imageHandPosePredict, "missCounter"):
        imageHandPosePredict.missCounter = 0

    results = hands.process(RGBImage)  # 偵測手掌
    predictedResult = 13
    probabilities = 0
    if isBothExist(results):
        imageHandPosePredict.missCounter = 0  # miss
        currentFeature = recorder.record2HandPerFrame(results)
        if len(currentFeature) == 84:  # 確認為84個特徵
            predictedResult, probabilities = combineAndPredict(currentFeature)
            if probabilities > 0.7:
                if predictedResult < 13 and predictedResult // 2 == lastResult // 2:
                    predictedResult = lastResult  # block reverse move
                else:
                    lastResult = predictedResult
            else:
                predictedResult = lastResult

    else:
        if missCounter >= maxMissCounter:
            continuousFeature = []
            showResult = "wait"
            predictCount = 0
        else:
            imageHandPosePredict.missCounter = imageHandPosePredict.missCounter + 1
    return predictedResult, probabilities


def isBothExist(results):
    isLeft = False
    isRight = False
    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
                results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                isLeft = True
            elif handed.classification[0].label == "Right":
                isRight = True

    if isLeft and isRight:
        return True
    else:
        return False


def combineAndPredict(currentFeature):
    global continuousFeature
    global predictCount
    global predictFrequence
    featureNumber = 84
    if len(continuousFeature) < 21:
        continuousFeature.append(currentFeature)
    else:
        del continuousFeature[0]

        continuousFeature.append(currentFeature)
        continuousFeature_np = np.array(continuousFeature)
        predictCount = predictCount + 1
        if predictCount == predictFrequence:
            predictCount = 0
            predictedResult, probabilities = predict(continuousFeature_np)
            return predictedResult, probabilities

    return 13, 0


def predict(continuousFeature):
    continuousFeature = np.array(continuousFeature)
    continuousFeature = (continuousFeature - continuousFeature.min()) / (
            continuousFeature.max() - continuousFeature.min()
    )
    predictData = np.expand_dims(continuousFeature, axis=0)  # (1, 21, 84)

    # 進行預測
    predictData = organizer.preprocessingData(predictData)
    prediction = lstmModel.predict(predictData, verbose=0)
    predictedResult = np.argmax(prediction, axis=1)[0]  # 確保predictedResult是一個整數
    probabilities = prediction[0][predictedResult]
    return predictedResult, probabilities


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')
user = ""
image_path = ""
section_width = 0
scan_area = 0
center_points = ()


@socketio.on('rotation')
def rotation(image):
    try:
        image = Image.open(BytesIO(base64.b64decode(image)))
        image = ImageOps.mirror(image)  # Flip the image horizontally
        image = np.array(image)
        predictedResult, probabilities = imageHandPosePredict(image)
        print("predictedResult: ", resultsList[predictedResult], "probabilities: ", probabilities)
        result = {"predictedResult": resultsList[predictedResult], "probabilities": probabilities}
        socketio.emit('rotation', result)
    except Exception as e:
        print(f"Error: {e}")



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
        image, section_width, scan_area = cd.draw_3x3_grid(image)

        image = Image.fromarray(image)
        image.save(image_path)

    except Exception as e:
        print(f"Error: {e}")


@socketio.on('initialize_cube_color')
def handle_initialize_cube_color():
    try:
        image = Image.open(image_path)
        start_time = time.time()

        records = cd.predict_color(image, section_width, scan_area, user)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Function execution time: {execution_time} seconds")
        socketio.emit('return_cube_color', records)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on('init_color_dataset')
def init_color_dataset(color):
    global section_width
    global scan_area
    try:
        image = Image.open(image_path)
        cd.init_color_dataset(user, color, image, section_width, scan_area)
    except Exception as e:
        print(f"Error: {e}")


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('join')
def handle_join(user_info):
    global user
    global image_path
    user = user_info
    image_path = f"images/{user}.jpeg"


@socketio.on('clear_color_dataset')
def handle_clear_color_dataset():
    cd.clear_color_dataset(user)


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
