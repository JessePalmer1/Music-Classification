# Music-Classification
This deep learning model classifies piano sheet music into three categories:

Baroque (target value 0)

Classical (target value 1)

Romantic (target value 2)

These are the three most recognized eras of classical music

The data used is not audio but rather sheet music in the format of musicXML (a standard format for representation of a music score in XML format)

The data preprocessing, training, and evaluation can all be run at one time by running dataPreprocessing.py
All you need is raw musicXML files in FILES_BEFORE_PROCESSING with distinct names, and their respective classification values written into initialFilesDatabase.csv. The script will perform the rest of the work.

This model massively overfits likely due to insufficient quantity of data. Nevertheless, it is quite interesting to play around with and see how it classifies certain pieces.
Reference classification.py to customize the actual neural network being used. I am using 3 layers and 4 epochs for training and it already overfits the model. A larger quantity of data might solve this problem.
What is interesting, however, is if testing data is selected after the large files are broken down into files of 4 measures each, the model does much better - this suggests it can pair 4 measure sections to its respective piece fairly accurately even when it has never seen these sections during training.

There are three scripts that perform all the necessary tasks: 'dataPreprocessing', 'classification', and 'modelTesting'
If you run 'dataPreprocessing', this will call all the necessary functions to train/evaluate a new model
If you run 'modelTesting' separately, this will take files in the folder 'testingData' and pass them through the input model for evaluation

Documentation:

dataPreprocessing.py
- Is the main driver script (calls functions from the other two scripts)
- Selects complete files from FILES_BEFORE_PROCESSING to be used as testing data
- Preprocesses the data:
    - Removes extra information such as composer names, titles, formatting information, etc
    - Splits the files into files of 4 measures each so more data can be aqcuired from one singular piece of music. These files are placed in FILES_AFTER_PROCESSING
    - Writes the titles of these new files and there respective classification values to 'processedFilesDatabase.csv'
    - Scrambles this database and rewrites it to 'database.csv'
    - Tokenizes each processed file and writes these arrays to 'tokenizedData.csv'
    - Tokenizes the respective target values and writes this array to 'tokenizedTargets.csv'

classification.py
- Reads the tokenized data from the csv's created from dataPreprocessing.py, and pads it
- Passes this padded two dimensional array as well as the respective target values into a neural network
- Trains the neural network and saves the model to the local directory with the input name

modelTesting.py
- Takes the testing data selected in dataPreprocessing.py and preprocesses it (also using methods from dataPreprocessing.py)
- Passes these files through the model that was input
- Prints the results

Various folders and csv files:
FILES_BEFORE_PROCESSING
- Contains the raw musicXML files of various pieces I chose and downloaded
FILES_AFTER_PROCESSING
- The folder that dataPreprocessing.py will store all the processed musicXML files in
testingData
- The equivalent of FILES_BEFORE_PROCESSING, but for running modelTesting.py separately
testingDataAfterProcessing
- The equivalent of FILES_AFTER_PROCESSING but for when modelTesting.py is run separately
initialFilesDatabase.csv
- The database of the initial raw musicXML files names and their respective classification value. THIS MUST BE ENTERED MANUALLY
processedFilesDatabase.csv
- The list of processed files and their respective classification values
database.csv
- A randomized version of processedFilesDatabase.csv, so the model is trained on data in a random order
tokenizedData.csv
- The tokenized data files to train the model. Each row corresponds to one file. Each number is a tokenized character/word using the        'AutoTokenizer' module from the 'transformers' library
tokenizedTargets.csv
- The tokenized target values of each file. The 'columns' (each value) of this file corresponds to the rows of tokenizedData.csv
