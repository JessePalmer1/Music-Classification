import xml.etree.ElementTree as ET
import csv
import random
import os
import modelTesting
import classification
from transformers import AutoTokenizer



MODEL_NAME = 'testModel'

# Define tokenizer to be used globally in this script (to standardize the tokenizing)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

path = os.getcwd()
MAX_FILES_PER_PIECE = 20


# The following five functions written in snake case delete various elements in the input musicXML file that are not needed for model training
# All five of these functions are helper methods called by processFile()
def remove_unnecessary_elements(root):
    work = root.find('work')
    if work is not None:
        root.remove(work)
    identification = root.find('identification')
    if identification is not None:
        root.remove(identification)
    defaults = root.find('defaults')
    if defaults is not None:
        root.remove(defaults)
    creditElements = root.findall('credit')
    for credit in creditElements:
        root.remove(credit)
    partList = root.find('part-list')
    if partList is not None:
        root.remove(partList)

def remove_width_and_print(root):
    for measure in root.iter('measure'):
        print_element = measure.find('print')
        if print_element is not None:
            measure.remove(print_element)
        if 'width' in measure.attrib:
            del measure.attrib['width']

def remove_words(root):
    for direction_type in root.iter('direction-type'):
        words = direction_type.findall('words')
        for wordElement in words:
            direction_type.remove(wordElement)

def remove_all_defaults_and_relatives(root):
    attribs_to_remove = ['default-x', 'default-y', 'relative-x', 'relative-y']
    for element in root.iter():
        for attr_name in attribs_to_remove:
            if attr_name in element.attrib:
                del element.attrib[attr_name]
def remove_version_and_id(root):
    if 'version' in root.attrib:
        del root.attrib['version']
    parts = root.findall('.//part')
    for part in parts:
        if 'id' in part.attrib:
            del part.attrib['id']


# Iterates through the tree, updating the attributes log and performing splits (both through calls to other methods)
# Helper method for processFile()
def splitFile(tree, name, category, outputFolderName, logAsCSV):
    fileList = []
    root = tree.getroot()
    measuresList = root.findall('.//measure')
    attributesLog = None
    for element in tree.iter():
        if element.tag == 'attributes':
            attributesLog = processAttribute(attributesLog, element)
        elif element.tag == 'measure':
            # Some measure number attributes are non-numerical (to represent alternate endings) and therefore can't be cast to type integer
            try:
                measureNum = int(element.attrib['number'])
            except:
                pass
            if measureNum % 4 == 0 and measureNum != 0:
                fileList.append(performSplit(measuresList, measureNum, attributesLog, name, category, outputFolderName, logAsCSV))
    return fileList


# Updates the attributesLog according to the new attribute element
# Helper method for performSplit() and splitFile()
def processAttribute(attributesLog, attribute):
    if attributesLog is None:
        return attribute
    elementsToUpdate = attribute.findall('*')
    attribs = [element.attrib for element in elementsToUpdate]
    tags = [element.tag for element in elementsToUpdate]
    for i, element in enumerate(attributesLog):
        if element.tag in tags:
            if element.attrib == elementsToUpdate[tags.index(element.tag)].attrib:
                attributesLog.remove(element)
                attributesLog.insert(i, elementsToUpdate[tags.index(element.tag)])
    return attributesLog


# Makes a new tree according to measureNum, adds correct attributes, and writes it to a new file with appropriate category
# Helper method for splitFile()
def performSplit(measuresList, measureNum, attributesLog, name, category, outputFolderName, logAsCSV):
    newTree = ET.ElementTree(ET.Element('score-partwise'))
    newRoot = newTree.getroot()
    newRoot.extend(measuresList[measureNum - 4 : measureNum])
    firstMeasure = newRoot.find('.//measure')
    # When making the first split, there is no need to insert attributes
    if measureNum != 4:
        # If there are new attributes at the beginning of the new first measure, update attributesLog accordingly
        if len(firstMeasure) > 0 and firstMeasure[0].tag == 'attributes':
            attributesLog = processAttribute(attributesLog, firstMeasure[0])
            firstMeasure.remove(firstMeasure[0])
        firstMeasure.insert(0, attributesLog)
        for i, measure in enumerate(newRoot.iter('measure')):
            measure.set('number', f'{i + 1}')
    # Write to a new file, and record the new file in the database
    extensionIndex = name.rfind('.')
    nameWithoutExtension = name[:extensionIndex]
    outputName = f'{nameWithoutExtension}Split{measureNum//4}.xml'
    outputFilePath = f"{path}/{outputFolderName}/{outputName}"
    newTree.write(outputFilePath)
    if logAsCSV:
        with open('tempFilesDatabase.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([outputName, category])
    return outputName
    

# Processes one file
# Helper method for PREPROCESS_DATA()
def processFile(fileName, fileCategory, folderName, outputFolderName, logAsCSV):
    # Load the file using ElementTree
    musicXmlFile = f'{path}/{folderName}/{fileName}'
    tree = ET.parse(musicXmlFile)
    root = tree.getroot()
    # Preprocess the file as a whole
    remove_unnecessary_elements(root)
    remove_width_and_print(root)
    remove_words(root)
    remove_all_defaults_and_relatives(root)
    remove_version_and_id(root)
    # Break into smaller files, 4 bars each
    fileList = splitFile(tree, fileName, fileCategory, outputFolderName, logAsCSV)
    return fileList


# Selects one file from each classification to be ommitted from training and used for testing instead
# Helper method for PREPROCESS_DATA()
def selectFilesForTesting():
    with open(f'{path}/initialFilesDatabase.csv', 'r') as file:
        reader = csv.reader(file)
        print("Splitting files into training/testing...")
        classificationToListOfFiles = dict()
        testingFiles = []
        for row in reader:
            if row[1] in classificationToListOfFiles:
                classificationToListOfFiles[row[1]].append(row[0])
            else:
                classificationToListOfFiles[row[1]] = [row[0]]
        for classification, fileList in classificationToListOfFiles.items():
            testingFiles.append(random.choice(fileList))
            testingFiles.append(random.choice(fileList))
        return testingFiles


# Removes all files in FILES_AFTER_PROCESSING and rows in processedFilesDatabase except ones in fileNameDatabase
# Writes fileNameDatabase to a csv (this will hold the classification value of all processed data)
# Helper method for PREPROCESS_DATA()
def configureProcessedFilesFolderAndDatabase(fileNameDatabase):
    print("Deleting randomly selected files for standardization...")
    with open(f'{path}/initialFilesDatabase.csv', 'r', newline='') as readFile, open('processedFilesDatabase.csv', 'w', newline='') as writeFile:
        reader = csv.reader(readFile)
        writer = csv.writer(writeFile)
        baseFileNameToClassification = dict()
        for row in reader:
            baseFileNameToClassification[row[0]] = row[1]
        for fileName in os.listdir(f'{path}/FILES_AFTER_PROCESSING'):
            if fileName not in fileNameDatabase:
                os.remove(f'{path}/FILES_AFTER_PROCESSING/{fileName}')
            else:
                baseName = f"{fileName[0 : fileName.index('Split')]}.xml"
                writer.writerow([fileName, baseFileNameToClassification[baseName]])


# Takes the data from the processedFilesDatabase.csv, randomizes it, and puts it into database.csv
# Helper method for PREPROCESS_DATA()
def configureRandomDatabase():
    print("Randomizing and writing file information to new database...")
    fileNameToTargetValue = []
    with open('processedFilesDatabase.csv', 'r') as dataBefore:
        reader = csv.reader(dataBefore)
        for row in reader:
            fileNameToTargetValue.append((row[0], row[1]))
    random.shuffle(fileNameToTargetValue)
    with open('database.csv', 'w', newline='') as dataAfter:
        writer = csv.writer(dataAfter)
        for tuple in fileNameToTargetValue:
            writer.writerow(tuple)


# Takes the xml data files from FILES_AFTER_PROCESSING and the target values from processedFilesDatabase.csv, tokenizes them, and puts them into lists for the classification alorithm
# Helper method for PREPROCESS_DATA()
def createDataLists():
    valueStringsToIntegers = {'baroque' : 0, 'classical' : 1, 'romantic' : 2}
    data = []
    targetValues = []
    with open('database.csv', 'r') as database:
        numDataFiles = len(list(csv.reader(database)))
    with open('database.csv', 'r') as database:
        reader = csv.reader(database)
        for i, row in enumerate(reader):
            print(f"Tokenizing data file {i + 1}/{numDataFiles}", end='\r', flush=True)
            fileName = row[0]
            fileTargetValue = row[1]
            with open(f"C:/Users/Jesse Palmer/Documents/Projects/Music Classification/FILES_AFTER_PROCESSING/{fileName}", 'r') as file:
                tree = ET.parse(file)
            root = tree.getroot()
            fileAsTokens = tokenizeFile(root)
            data.append(fileAsTokens)
            targetValues.append(valueStringsToIntegers[fileTargetValue])
    print("\nSuccessfully created data lists for model input")
    return data, targetValues


# Takes an input root of an xml file (already preprocessed) and converts it to an array of tokens
# Helper method for createData()
def tokenizeFile(fileRoot):
    fileAsStr = ET.tostring(fileRoot, encoding='unicode')
    listOfElements = fileAsStr.split('\n')
    fileAsTokens = []
    for element in listOfElements:
        tokens = tokenizer.tokenize(element)
        tokensIDs = tokenizer.convert_tokens_to_ids(tokens)
        for id in tokensIDs:
            fileAsTokens.append(id)
    return fileAsTokens


# Takes the tokenized data and writes it to files (to negate the need to tokenize the data every time you train)
# Helper method for PREPROCESS_DATA()
def storeDataAndTargets(data, targetValues):
    print("Writing data lists to files for storage...")
    with open('tokenizedData.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    with open('tokenizedTargets.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(targetValues)

        
        
# Calls the appropriate methods to preprocess the data (clean it/break it into smaller files/tokenize it)
# Returns the list of testing files (files not processed for training)
def PREPROCESS_DATA():
    testingFiles = selectFilesForTesting()
    # Delete all files in FILES_AFTER_PROCESSING to prepare for new processed data
    for filename in os.listdir(f'{path}/FILES_AFTER_PROCESSING'):
        filePath = os.path.join(f'{path}/FILES_AFTER_PROCESSING', filename)
        if os.path.isfile(filePath):
            os.remove(filePath)
    # Iterate through initialFilesDatabase
    with open(f'{path}/initialFilesDatabase.csv', 'r') as file:
        reader = csv.reader(file)
        print("Preprocessing the training data and breaking them into smaller files...")
        fileNameDatabase = []
        for row in reader:
            fileName = row[0]
            fileCategory = row[1]
            # If the file is one of the randomly selected testing files, don't process it
            if fileName in testingFiles:
                continue
            else:
                filesList = processFile(fileName, fileCategory, 'FILES_BEFORE_PROCESSING', 'FILES_AFTER_PROCESSING', False)
                # If the number of smaller (4 measure) files exceeds the MAX_FILES_PER_PIECE constant, then take a random sample
                if len(filesList) > MAX_FILES_PER_PIECE:
                    fileNameDatabase.extend(random.sample(filesList, MAX_FILES_PER_PIECE))
                else:
                    fileNameDatabase.extend(filesList)
        # Deletes files from FILES_AFTER_PROCESSING which were not randomly selected, and configures the processedFilesDatabase.csv (to store classification value)
        configureProcessedFilesFolderAndDatabase(fileNameDatabase)
        configureRandomDatabase()
        data, targetValues = createDataLists()
        storeDataAndTargets(data, targetValues)
        return testingFiles



def main():
    testingFiles = PREPROCESS_DATA()
    maxRowLengthForPadding = classification.RUN_TRAINING_PROCEDURES(f'{MODEL_NAME}.h5')
    modelTesting.EVALUATE_MODEL(MODEL_NAME, testingFiles, maxRowLengthForPadding)



if __name__ == '__main__':
    main()