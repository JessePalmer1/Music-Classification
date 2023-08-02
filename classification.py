import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dropout



# Pads the tokenized data so the model receives standardized input
# Helper method for runTrainingProcedures()
def padData():
    print("Padding data...")
    paddedData = []
    with open('tokenizedData.csv', 'r') as file:
        dataReader = csv.reader(file)
        sequences = [list(map(int, row)) for row in dataReader]
        maxLength = max(len(row) for row in sequences)
        for i, row in enumerate(sequences):
            paddedRow = padSingularList(row, maxLength)
            paddedData.append(paddedRow)
        return paddedData, maxLength



# Pads a singular list of data
# Helper method for padData() AND modelTesting.processPreprocessedFiles()
def padSingularList(list, maxLength):
    padding = [0] * (maxLength - len(list))
    paddedRow = list + padding
    return paddedRow


# Trains a classification model on the given inputs
# THIS IS WHERE USER CUSTOMIZATION CAN TAKE PLACE: the type of model, how much dropout, number of layers and nodes, loss algorithm, etc.
# Helper method for RUN_TRAINING_PROCEDURES()
def trainModel(paddedData, targets):
    paddedData = np.array(paddedData)
    targets = np.array(targets)
    
    print("Defining model...")
    model = keras.Sequential([
        keras.layers.Dense(1024, input_shape=(paddedData.shape[1],), activation='relu'),
        Dropout(0.2),
        keras.layers.Dense(420, activation='relu'),
        # Dropout(0.2),
        # keras.layers.Dense(128, activation='relu'),
        # Dropout(0.2),
        # keras.layers.Dense(64, activation='relu'),
        Dropout(0.2),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    print("Compiling model...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit(paddedData, targets, epochs=4, batch_size=32)
    return model



# Reads the data and target values from csv files as lists, pads these lists, and trains a classification model with these data lists as inputs
def RUN_TRAINING_PROCEDURES(modelName):
    paddedData, maxRowLength = padData()
    targets = []
    with open('tokenizedTargets.csv', 'r') as file:
        reader = csv.reader(file)
        targets = [int(value) for value in next(reader)]
    print("Data lists successfully configured.")
    model = trainModel(paddedData, targets)
    model.save(modelName)
    return maxRowLength
        


def main():
    RUN_TRAINING_PROCEDURES('testModel.h5')

if __name__ == '__main__':
    main()