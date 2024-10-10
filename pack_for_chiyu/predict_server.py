import data_organizer as do
import mediapipe as mp
import keras
import recorder as rd
import numpy as np

# ---------
recorder = rd.Recorder()
organizer = do.DataOrganizer()
mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法
lstmModel = keras.models.load_model("lstm_2hand_model.keras")
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
    "B'(Back Clockwise)",
    "B (Back Counter Clockwise)",
    "D'(Bottom Left)",
    "D (Bottom Right)",
    "F (Front Clockwise)",
    "F' (Front Counter Clockwise)",
    "L'(Left Down)",
    "L (Left Up)",
    "R (Right Down)",
    "R'(Right Up)",
    "U (Top Left)",
    "U'(Top Right)",
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
