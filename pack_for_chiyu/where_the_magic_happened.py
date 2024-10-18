
from . import data_organizer as do
import mediapipe as mp
import keras
from . import recorder as rd
import numpy as np

recorder = rd.Recorder()
organizer = do.DataOrganizer()


recorder = rd.Recorder()
organizer = do.DataOrganizer()
timeSteps = 21
features = 60

mpDrawing = mp.solutions.drawing_utils  # 繪圖方法
mpDrawingStyles = mp.solutions.drawing_styles  # 繪圖樣式
mpHandsSolution = mp.solutions.hands  # 偵測手掌方法

lstmModel = keras.models.load_model(
            "pack_for_chiyu/lstm_2hand_noCTC_60Features.keras",
        )
showResult = "wait"
predictFrequence = 1
predictCount = 0
hands = mpHandsSolution.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
lastResult = 13
resultsList = [
    "B'",
    "B ",
    "D'",
    "D ",
    "F ",
    "F'",
    "L'",
    "L ",
    "R ",
    "R'",
    "U'",
    "U ",
    "Stop",
    "wait",
]
currentFeature = []  # 目前畫面的資料
continuousFeature = []  # 目前抓到的前面
missCounter = 0
maxMissCounter = 10


def predict(continuousFeature):
    continuousFeature = np.array(continuousFeature)
    predictData = np.expand_dims(continuousFeature, axis=0)  # (1, timeSteps, features)
    # 進行預測
    predictData = organizer.preprocessingData(predictData)

    prediction = lstmModel.predict(predictData, verbose=0)  # error
    predictedResult = np.argmax(prediction, axis=1)[0]
    probabilities = prediction[0][predictedResult]
    return predictedResult, probabilities


def blockIllegalResult(probabilities, lastResult, currentResult):
    if probabilities > 0.7:
        if currentResult in [12, 13]:  # stop, wait 不動
            return currentResult

        if currentResult == lastResult:  # block same move
            return 13  # wait

        if lastResult != 12 and (lastResult // 2) == (
            currentResult // 2
        ):  # block reverse move
            return lastResult

        return currentResult
    else:
        return lastResult

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

    if len(continuousFeature) < 21:
        continuousFeature.append(currentFeature)
    else:
        del continuousFeature[0]
        continuousFeature.append(currentFeature)
        continuousFeature_np = np.array(continuousFeature, dtype="float")
        predictCount = predictCount + 1
        if showResult != "stop":
            if predictCount == predictFrequence:
                predictCount = 0
                predictedResult, probabilities = predict(continuousFeature_np)
                continuousFeature = []
                return predictedResult, probabilities

    return 13, 0

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
    if isBothExist(results):  # 有雙手
        imageHandPosePredict.missCounter = 0  # miss
        currentFeature = recorder.record2HandPerFrame(results)
        if len(currentFeature) == 84:  # 確認為fearures個特徵
            predictedResult, probabilities = combineAndPredict(currentFeature)
            predictedResult = blockIllegalResult(
                probabilities, lastResult, predictedResult
            )
            if predictedResult not in [12, 13]:
                print(f"in fucntion{resultsList[predictedResult]}")

    else:
        if imageHandPosePredict.missCounter >= maxMissCounter:
            continuousFeature = []
            showResult = "wait"
            predictCount = 0

        else:
            imageHandPosePredict.missCounter = imageHandPosePredict.missCounter + 1
    resultString = resultsList[predictedResult]
    return resultString,probabilities


