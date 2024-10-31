import tools.data_organizer as do
import mediapipe as mp
import keras
import tools.recorder as rd
import numpy as np
import time

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
    "the_precious_working_model/lstm_2hand_noCTC_60Features.keras",
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
# currentFeature = []  # 目前畫面的資料
continuousFeature = []  # 目前抓到的前面
missCounter = 0
maxMissCounter = 10


def isHandMoving(results, currentFeature):
    global continuousFeature
    if not hasattr(isHandMoving, "lastHandJoint"):
        isHandMoving.lastHandJoint = []
    if not hasattr(isHandMoving, "lastHandJoint2"):
        isHandMoving.lastHandJoint2 = []
    if not hasattr(isHandMoving, "lastFingertips"):
        isHandMoving.lastFingertips = []
    threshold = [0.02, 0.08]
    fingertipsNodes = [8, 12, 16]  # 4,20

    leftFingertips = []
    rightFingertips = []

    if results.multi_hand_landmarks:
        for handLandmarks, handed in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            if handed.classification[0].label == "Left":
                leftWrist = [
                    handLandmarks.landmark[0].x,
                    handLandmarks.landmark[0].y,
                ]

                for i in fingertipsNodes:
                    leftFingertips.append(handLandmarks.landmark[i].x)
                    leftFingertips.append(handLandmarks.landmark[i].y)
            elif handed.classification[0].label == "Right":

                for i in fingertipsNodes:
                    rightFingertips.append(handLandmarks.landmark[i].x)
                    rightFingertips.append(handLandmarks.landmark[i].y)
        currentFingertips = leftFingertips + rightFingertips

        for i in range(len(currentFingertips)):
            currentFingertips[i] = currentFingertips[i] - leftWrist[i % 2]
        currentFingertips = organizer.normalizedOneDimensionList(currentFingertips)

        if isHandMoving.lastFingertips:
            for i in range(len(currentFingertips)):
                fingertipsSAD = abs(
                    currentFingertips[i] - isHandMoving.lastFingertips[i]
                )
                if fingertipsSAD > threshold[i % 2]:  # if moved
                    # print(f"fingertipsSAD:{fingertipsSAD},from:{getIFinger(i)}")
                    isHandMoving.lastFingertips = []
                    isHandMoving.lastHandJoint2 = isHandMoving.lastHandJoint.copy()
                    isHandMoving.lastHandJoint = currentFeature.copy()
                    return True

        isHandMoving.lastFingertips = currentFingertips.copy()
        # 把目前的fingertips 保留
        if not isHandMoving.lastHandJoint2:
            isHandMoving.lastHandJoint2 = currentFeature.copy()
        else:
            isHandMoving.lastHandJoint2 = isHandMoving.lastHandJoint.copy()

        isHandMoving.lastHandJoint = currentFeature.copy()

    return False


# ------------------
def interpolate_number(returnList):
    for i in range(len(returnList)):
        if returnList[i] == None:
            counter = 1
            while (returnList[i + counter]) == None:
                counter = counter + 1
            lastTimeStep = returnList[i + counter]
            nextTimeStep = returnList[i - 1]
            length = counter + 1
            while counter > 0:  # 以time step 為單位
                newTimeStep = []
                for leftValue, rightValue in zip(nextTimeStep, lastTimeStep):
                    interpolated_value = (
                        (rightValue - leftValue) * counter / length
                    ) + leftValue
                    newTimeStep.append(interpolated_value)
                returnList[i + counter - 1] = newTimeStep.copy()
                counter -= 1
    return returnList


def linear_interpolation(targetList):
    global timeSteps
    return_list = [None] * (timeSteps - 1)
    length = len(targetList)
    return_list[0] = targetList[0]  # head and end
    return_list[timeSteps - 2] = targetList[-1]
    for i in range(1, len(targetList) - 1):  # spread the rest of them
        insert_index = ((i * (timeSteps - 2)) // (length - 1)) + 1
        return_list[insert_index] = targetList[i]
    return_list = interpolate_number(return_list)
    return return_list


# -----------------------


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


def imageHandPosePredict(RGBImage):  # 重構判斷式
    global continuousFeature
    global showResult
    global predictCount
    global hands
    global lastResult

    # 初始化靜態變數
    if not hasattr(imageHandPosePredict, "missCounter"):
        imageHandPosePredict.missCounter = 0  # 用於計算沒有雙手的次數
    if not hasattr(imageHandPosePredict, "handMovingPassCount"):
        imageHandPosePredict.handMovingPassCount = 0  # 用於計算免檢查通行次數
    if not hasattr(imageHandPosePredict, "startTime"):
        imageHandPosePredict.startTime = 0  # 用於計算免檢查通行次數

    results = hands.process(RGBImage)  # 偵測手掌
    predictedResult = 13
    probabilities = 0

    if isBothExist(results):  # 如果有雙手
        imageHandPosePredict.missCounter = 0
        currentFeature = recorder.record2HandPerFrame(results)
        currentTime = time.time()
        if imageHandPosePredict.handMovingPassCount == 0:
            if isHandMoving(results, currentFeature):
                imageHandPosePredict.startTime = time.time()
                imageHandPosePredict.handMovingPassCount = timeSteps
                if len(continuousFeature) == 0:
                    continuousFeature.append(
                        isHandMoving.lastHandJoint
                    )  # 此處有空可以重構
                    continuousFeature.append(isHandMoving.lastHandJoint2)
            else:
                continuousFeature = []
                pass
        else:
            imageHandPosePredict.handMovingPassCount -= 1

        if (
            len(currentFeature) == 84 and imageHandPosePredict.handMovingPassCount > 0
        ):  # 確認為特徵的數量
            if (currentTime - imageHandPosePredict.startTime) > 3 and len(
                continuousFeature
            ) > 3:
                continuousFeature = linear_interpolation(
                    continuousFeature
                )  # interpolate to 20 time steps
            predictedResult, probabilities = combineAndPredict(currentFeature)
            predictedResult = blockIllegalResult(
                probabilities, lastResult, predictedResult
            )
            if predictedResult not in [12, 13]:
                resultString = resultsList[predictedResult]
            #     print(resultsList[predictedResult])
            # # if not predictedResult == 13:
            # #     print(resultsList[predictedResult])

    else:
        if imageHandPosePredict.missCounter >= maxMissCounter:
            continuousFeature = []
            showResult = "wait"
            predictCount = 0
            isHandMoving.lastHandJoint = []
            isHandMoving.lastHandJoint2 = []
            imageHandPosePredict.handMovingPassCount = 0
        else:
            imageHandPosePredict.missCounter += 1
    resultString = resultsList[predictedResult]
    return resultString, probabilities
