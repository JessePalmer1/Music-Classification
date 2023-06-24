# Music-Classification
This deep learning model classifies piano sheet music into three categories:
Baroque (target value 0)
Classical (target value 1)
Romantic (target value 2)

These are the first three eras of classical music. I might add impressionism at some point, but I wanted to keep it simple.
The data used is not audio but rather sheet music in the format of musicXML (a standard format for representation of a music score in XML format)
These xml files are processed in dataPreprocessing.py to delete any extra information such as composer names, titles, formatting information, etc. dataPreprocessing also splits the files into 4 measures each, writing the processed data to new files. In this way more data can be aquired from one piece of music, and the inputs for the neural network won't be incredibly long. Lastly, this script writes the new titles of each file and the respective target value to a csv.
After preprocessing, dataProcessing.py reads the files from the csv, scrambles them and puts them into a new csv (not sure if this is necessary) and then tokenizes each file to represent it as an array of numerical values. It also tokenizes the target values into 0, 1, and 2. The data is then stored in tokenizedData.csv and tokenizedTargets.csv respectively.
Finally, classification takes the data from the tokenized data csv files, pads it to normalize the length (according to the longest file) and trains a keras sequential model.
