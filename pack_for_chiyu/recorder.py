def get_dimension(lst):
    # 初始化維度為 1
    dimension = 1
    # 檢查第一個元素是否也是一個列表
    if isinstance(lst[0], list):
        # 遞歸地獲取元素的維度
        dimension = 1 + get_dimension(lst[0])
    return dimension


class Recorder:
    def __init__(self):
        self.isRecording = False
        self.recordedTimes = 0
        self.neededTimes = 21  # 21
        self.isFinish = False
        self.leftFeaturePerData = []
        self.rightFeaturePerData = []
        self.originalData = []
        self.locationData = []
        self.accelerateData = []

    def _fixLocationData(self, featurePerData):
        if not self.isRecording:
            if featurePerData:
                del featurePerData[0]
        return featurePerData

    def recordBothHand(self, results, featurePerData):
        if self.recordedTimes < self.neededTimes:
            featurePerFrame = []
            leftDataPerFrame = []
            rightDataPerFrame = []
            if results.multi_hand_landmarks:
                for handLandmarks, handed in zip(  # 遍歷節點
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # 檢查是否為右手
                    if handed.classification[0].label == "Right":
                        if handLandmarks.landmark:
                            for landmark in handLandmarks.landmark:
                                rightDataPerFrame.append(landmark.x)
                                rightDataPerFrame.append(landmark.y)

                    elif handed.classification[0].label == "Left":
                        if handLandmarks.landmark:
                            for landmark in handLandmarks.landmark:
                                leftDataPerFrame.append(landmark.x)
                                leftDataPerFrame.append(landmark.y)

                featurePerFrame.extend(leftDataPerFrame)
                featurePerFrame.extend(rightDataPerFrame)
                featurePerData.append(featurePerFrame)
                self.recordedTimes = self.recordedTimes + 1

        else:
            self.rightFeaturePerData = []
            self.leftFeaturePerData = []
            self.recordedTimes = 0
            self.isRecording = False
            self.isFinish = True

        return featurePerData

    def record2HandPerFrame(self, results):
        featurePerFrame = []
        leftDataPerFrame = []
        rightDataPerFrame = []
        if results.multi_hand_landmarks:
            for handLandmarks, handed in zip(  # 遍歷節點
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # 檢查是否為右手
                if handed.classification[0].label == "Right":
                    if handLandmarks.landmark:
                        for landmark in handLandmarks.landmark:
                            rightDataPerFrame.append(landmark.x)
                            rightDataPerFrame.append(landmark.y)

                elif handed.classification[0].label == "Left":
                    if handLandmarks.landmark:
                        for landmark in handLandmarks.landmark:
                            leftDataPerFrame.append(landmark.x)
                            leftDataPerFrame.append(landmark.y)

            featurePerFrame.extend(leftDataPerFrame)
            featurePerFrame.extend(rightDataPerFrame)
        return featurePerFrame
