import numpy as np

# from numba import jit


class DataOrganizer:
    def cutFirstTimeStep(self, npArray):
        npArray = npArray[:, 1:, :]
        return npArray

    def removePalmNode(self, inputList):
        palm = [
            0,
            1,
            2,
            3,
            10,
            11,
            18,
            19,
            26,
            27,
            34,
            35,
            42,
            43,
            44,
            45,
            52,
            53,
            60,
            61,
            68,
            69,
            76,
            77,
        ]
        inputList = np.delete(inputList, palm, axis=2)  # 刪除對應的索引(in features)
        return inputList

    # @staticmethod
    # @jit(nopython=True)
    def preprocessingData(self, inputList):

        inputList = np.array(inputList)
        inputList = self.normalizedWithEachTimeSteps(inputList)
        # inputList = self.getRelativeWithFirstTimeStep(inputList)
        inputList = self.getRelativeLocation(inputList)
        inputList = self.removePalmNode(inputList)
        return inputList

    @staticmethod
    # @jit(nopython=True)
    def getRelativeLocation(npArray):  # 輸入:(data number,time step, features)
        for i in range(len(npArray)):
            for j in range(len(npArray[i])):
                originX = npArray[i][j][0]
                originY = npArray[i][j][1]
                for k in range(len(npArray[i][j])):
                    if k % 2 == 0:
                        npArray[i][j][k] = npArray[i][j][k] - originX
                    else:
                        npArray[i][j][k] = npArray[i][j][k] - originY
        return npArray

    @staticmethod
    # @jit(nopython=True)
    def normalizedWithEachTimeSteps(
        inputList,
    ):  # 輸入:(data number,time step, features)

        for i in range(len(inputList)):
            for j in range(inputList.shape[1]):
                inputList[i, j] = (inputList[i, j] - inputList[i, j].min()) / (
                    inputList[i, j].max() - inputList[i, j].min()
                )
        return inputList

    def normalizedOneDimensionList(self, inputList):
        npInputList = np.array(inputList)
        normalizedList = (npInputList - npInputList.min()) / (npInputList.max() - npInputList.min())
        normalizedList= normalizedList.tolist()
        return normalizedList

    def getRelativeWithFirstTimeStep(self, npArray):
        for i in range(len(npArray)):
            originX = npArray[i][0][0]
            originY = npArray[i][0][1]
            for j in range(len(npArray[i])):
                for k in range(len(npArray[i][j])):
                    if k % 2 == 0:
                        npArray[i][j][k] = npArray[i][j][k] - originX
                    else:
                        npArray[i][j][k] = npArray[i][j][k] - originY
        return npArray

    def getDataFromTxt(self, fileName):
        with open(f"{fileName}.txt", "r") as file:
            content = file.read()
        result = eval(content)
        return result

    def getAccelerate(self, npArray):
        for i in range(len(npArray)):
            for j in reversed(range(len(npArray[i]))):
                for k in reversed(range(len(npArray[i][j]))):
                    if not j < 1:
                        npArray[i][j][k] = npArray[i][j][k] - npArray[i][j - 1][k]
        npArray = self.cutFirstTimeStep(npArray)
        return npArray

    def findErrorData(self, fileName):
        targetFile = self.getDataFromTxt(fileName)
        errorList = []
        for i in range(len(targetFile)):
            if not len(targetFile[i]) == 21:
                errorList.append(i)
                continue
            for j in range(len(targetFile[i])):
                if not len(targetFile[i][j]) == 84:
                    errorList.append(i)
                continue
        return errorList

    def reverseTimeData(self, npArray):
        npArray = [sublist[::-1] for sublist in npArray]
        return npArray
