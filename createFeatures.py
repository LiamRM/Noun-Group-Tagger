"""
Name        : createFeatures.py
Author      : Liam Richards
Date created: October 29th, 2021
Description : Create a program that takes a file like WSJ_02-21.pos-chunk as input 
              and produces a file consisting of feature value pairs for use with the 
              maxent trainer and classifier. As this step represents the bulk of the assignment, 
              there will be more details below, including the format information, etc. 
              This program should create two output files. From the training corpus (WSJ_02-21.pos-chunk), 
              create a training feature file (training.feature). From the development corpus (WSJ_24.pos), 
              create a test feature file (test.feature). See details below.
"""

import sys      # for command-line args
import nltk
import logging  # for debugging
import string   # for string.punctuation, list of punctuation chars
import traceback    # for debugging traceback stack

from nltk.tokenize import word_tokenize  # for debugging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -  %(levelname)s -  %(message)s')
# logging.disable(logging.CRITICAL) # disabling logging messages (comment this line to show logging messages)

# Variables
porter = nltk.PorterStemmer()       # same amount of correct tags as lancaster, but better F1 score than Lancaster Stemmer by 0.1
wnl = nltk.WordNetLemmatizer()

# Arguments should be structured as follows: [createfeatures.py, training corpus input file name, development corpus file name] --> input: WSJ_02-21.pos-chunk and WSJ_24.pos
try:
    trainingInputFile = sys.argv[1]
    devInputFile = sys.argv[2]

    ### OUTPUT FILE 1: training.feature from WSJ_02-21.pos-chunk (training corpus) ###
    with open(trainingInputFile) as f, \
        open("training.feature", "w+") as trainingFile:

        # Other features
        prev_word_features = ["", ""]       # [word, POS]
        prev_BIO_tag = '@@'
        prev_prev_word_features = ["", ""]

        lines = f.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            line = line.strip()
            
            # Ignore blank lines
            if line == "" or line.isspace():
                trainingFile.write("\n")
            else:
                # Tokenize the line
                lineList = line.split("\t")

                ## FEATURES ##
                word = lineList[0]
                pos_tag = lineList[1]
                BIO_tag = lineList[2]
                # Stem / lemmatize the word
                stemmed = wnl.lemmatize(word)
                # Check if capitalized
                isCapitalized = False
                if(word[0].isupper):
                    isCapitalized = True
                # Check if punctuation
                isPunctuation = False
                if(word in string.punctuation):
                    isPunctuation = True
                # Check if number
                isNumber = False
                if(word.isdigit() or word.isdecimal()):
                    isNumber = True
                finalCharacter = word[-1]

                # Check if at the beginning of a sentence
                beginningOfSent = False
                if(prev_word_features[0] == "" or prev_word_features[0].isspace()):
                    beginningOfSent = True
                
                # Tokenize next line and next next line, as long as within range --> NOT written to file
                if(i < len(lines) - 2):
                    next_line = lines[i+1].strip()
                    next_next_line = lines[i+2].strip()
                    # Ignore blank lines
                    if next_line == "" or next_line.isspace():
                        pass
                    else:
                        nextLineList = next_line.split("\t")
                    # Ignore blank lines
                    if next_next_line == "" or next_next_line.isspace():
                        pass
                    else:
                        nextNextLineList = next_next_line.split("\t")
                elif(i == len(lines) - 1 or i == len(lines) - 2):
                    next_line = lines[i+1].strip()
                    # Ignore blank lines
                    if next_line == "" or next_line.isspace():
                        pass
                    else:
                        nextLineList = next_line.split("\t")
                        nextNextLineList = ["", "", ""]
                else:
                    nextLineList = ["", "", ""]
                    nextNextLineList = ["", "", ""]
                
                next_word = nextLineList[0]
                next_POS = nextLineList[1]
                next_next_word = nextNextLineList[0]
                next_next_POS = nextNextLineList[1]

                # for the training file only, the last field should be the BIO tag (B-NP, I-NP or O)
                trainingFile.write(f"{word}\t{pos_tag}\t{stemmed}\t{prev_word_features[0]}\t{prev_word_features[1]}\t{isCapitalized}\t{isPunctuation}\t{isNumber}\t{finalCharacter}\t{beginningOfSent}\t{BIO_tag}\n")

                # Update previous word tags
                prev_prev_word_features = prev_word_features
                prev_word_features = [word, pos_tag]


    ### OUTPUT FILE 2: test.feature from WSJ_24.pos (development corpus) ###
    with open(devInputFile) as f, \
        open("test.feature", "w+") as testFile:

        # Other features
        prev_word_features = ["", ""]       # [word, POS]
        prev_BIO_tag = '@@'

        lines = f.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            line = line.strip()

            # Ignore blank lines
            if line == "" or line.isspace():
                testFile.write("\n")
            else:
                # Tokenize the line
                lineList = line.split("\t")
                word = lineList[0]
                pos_tag = lineList[1]
                # Stem / Lemmatize the word
                stemmed = wnl.lemmatize(word)
                # Check if capitalized
                isCapitalized = False
                if(word[0].isupper):
                    isCapitalized = True
                # Check if punctuation
                isPunctuation = False
                if(word in string.punctuation):
                    isPunctuation = True
                # Check if number
                isNumber = False
                if(word.isdigit() or word.isdecimal()):
                    isNumber = True
                finalCharacter = word[-1]

                # Check if at the beginning of a sentence
                beginningOfSent = False
                if(prev_word_features[0] == "" or prev_word_features[0].isspace()):
                    beginningOfSent = True

                # Tokenize next line and next next line, as long as within range --> NOT written to file
                if(i < len(lines) - 2):
                    next_line = lines[i+1].strip()
                    next_next_line = lines[i+2].strip()
                    # Ignore blank lines
                    if next_line == "" or next_line.isspace():
                        pass
                    else:
                        nextLineList = next_line.split("\t")
                    # Ignore blank lines
                    if next_next_line == "" or next_next_line.isspace():
                        pass
                    else:
                        nextNextLineList = next_next_line.split("\t")
                elif(i == len(lines) - 1 or i == len(lines) - 2):
                    next_line = lines[i+1].strip()
                    # Ignore blank lines
                    if next_line == "" or next_line.isspace():
                        pass
                    else:
                        nextLineList = next_line.split("\t")
                        nextNextLineList = ["", "", ""]
                else:
                    nextLineList = ["", "", ""]
                    nextNextLineList = ["", "", ""]
                
                next_word = nextLineList[0]
                next_POS = nextLineList[1]
                next_next_word = nextNextLineList[0]
                next_next_POS = nextNextLineList[1]

                # for the test file, there should be no final BIO field (as there is none in the .pos file that you would be training from)
                testFile.write(f"{word}\t{pos_tag}\t{stemmed}\t{prev_word_features[0]}\t{prev_word_features[1]}\t{isCapitalized}\t{isPunctuation}\t{isNumber}\t{finalCharacter}\t{beginningOfSent}\n")
                
                # Update previous word tags
                prev_word_features = [word, pos_tag]

except Exception as e:
    print("\nERROR:", e)
    print("Please run the program as follows: python createFeatures.py [trainingInputFileName] [developmentInputFileName]")
    print("EX:\tpython createFeatures.py WSJ_02-21.pos-chunk WSJ_24.pos\n")
    print(traceback.format_exc())