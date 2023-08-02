from tensorflow import keras
import dataPreprocessing
import numpy as np
import os
import xml.etree.ElementTree as ET
import csv
import classification




valueIntegerToString = {0 : 'baroque', 1 : 'classical', 2 : 'romantic'}

folderPath = os.getcwd()
    

# Finds the maximum
def getMaxLengthForPadding():
    maxLength = 0
    with open(f'{folderPath}/tokenizedData.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > maxLength:
                maxLength = len(row)
    return maxLength


# preprocesses the xml file(s) and breaks them into smaller samples
def preprocessInputFile(fileName, folderName):
    with open(f'{folderPath}/{folderName}/{fileName}') as file:
        dataPreprocessing.processFile(fileName, None, folderName, 'testingDataAfterProcessing', False)


# Tokenizes and pads the data
def processPreprocessedFiles(maxLength):
    data = []
    for fileName in os.listdir(f'{folderPath}/testingDataAfterProcessing'):
        with open(f'{folderPath}/testingDataAfterProcessing/{fileName}') as file:
            tree = ET.parse(file)
        root = tree.getroot()
        tokenizedList = dataPreprocessing.tokenizeFile(root)
        paddedList = classification.padSingularList(tokenizedList, maxLength)
        data.append(paddedList)
    return data


# Deletes the files in 'testingDataAfterProcessing', so new files can be added
def deleteUsedDataFiles():
    for filename in os.listdir(f'{folderPath}/testingDataAfterProcessing'):
        filePath = os.path.join(f'{folderPath}/testingDataAfterProcessing', filename)
        if os.path.isfile(filePath):
            os.remove(filePath)


# Runs the data through the model, and tracks the frequency of each output
def passThroughModel(data, modelName):
    model = keras.models.load_model(f'{folderPath}/{modelName}.h5')
    finalData = np.array(data)
    predictions = model.predict(finalData)
    predictedValues = np.argmax(predictions, axis=1)
    occurrences = [0, 0, 0]
    for value in predictedValues:
        occurrences[value] += 1
    return occurrences


# Prints the outputs in readable format
def printOutcomes(occurrences, fileName):
    extensionIndex = fileName.rfind('.')
    nameWithoutExtension = fileName[:extensionIndex]
    total = np.sum(occurrences)
    print(f"\nInput data file \'{nameWithoutExtension}\' is {int((occurrences[0] / total) * 100)}% Baroque, {int((occurrences[1] / total) * 100)}% Classical"
        f" and {int((occurrences[2] / total) * 100)}% Romantic\n")



# Evaluates the input model name on the input training data
def EVALUATE_MODEL(modelName, trainingData, maxLength):
    print("Evaluating the model on the following training data: ", trainingData)
    for fileName in trainingData:
        testSingleFile(fileName, maxLength, 'FILES_BEFORE_PROCESSING', modelName)
    

# Preprocesses a single file and runs it through the model
def testSingleFile(fileName, maxLength, folderName, modelName):
    deleteUsedDataFiles()
    preprocessInputFile(fileName, folderName)
    data = processPreprocessedFiles(maxLength)
    occurrences = passThroughModel(data, modelName)
    printOutcomes(occurrences, fileName)
    


def main():
    MODEL_NAME = 'testModel' # Omit the '.h5' extension in this variable
    maxLength = getMaxLengthForPadding()
    for fileName in os.listdir(f'{folderPath}/testingData'):
        testSingleFile(fileName, maxLength, 'testingData', MODEL_NAME)


if __name__ == '__main__':
    main()