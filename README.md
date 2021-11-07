# NLP HW 5: Sequence Labelling 
This program will serves as a Noun Group tagger, but uses feature selection and a machine learning algorithm rather than a manually-written algorithm to determine Noun Groups. The machine learning algorithm chosen was Maximum Entropy, and the model is created by running the two java programs ```maxent-3.0.0.jar``` and ```trove.jar```.
My Python program ```createFeatures.py``` takes two files as input: ```WSJ_02-21.pos-chunk``` (the training corpus) and ```WSJ_24.pos``` (the development corpus). It then outputs two feature files, which are fed into the Maximum Entropy java files: ```training.feature``` and ```test.feature```.

### How to run the program
1. Run the Python createFeatures.py:\
```python createFeatures.py [training corpus input file name] [development / test corpus file name]```\
For example, for the development corpus:\
```python createFeatures.py WSJ_02-21.pos-chunk WSJ_24.pos```
2. Compile the Java maximum entropy programs:\
```javac -cp maxent-3.0.0.jar;trove.jar *.java```
3. Creating model of the training data:\
```java -cp .;maxent-3.0.0.jar;trove.jar MEtrain training.feature model.chunk```
4. Creating system output:\
```java -cp .;maxent-3.0.0.jar;trove.jar MEtag test.feature model.chunk response.chunk```
5. Scoring the system results:\
```py -2 score.chunk.py WSJ_24.pos-chunk response.chunk```

The following sections describe the features I tried and their resulting scores for the development corpus: 

### Normalizing the text
I first tried two NLTK-provided stemmers, the first being the Porter Stemmer and the second, the Lancaster Stemmer.
I ran the ML algorithm twice, using teh different stemmers each time. The Porter Stemmer's output had the same amount of correct tags as the Lancaster Stemmer, but the Porter Stemmer had a better F1 score than Lancaster Stemmer by 0.1.

I then decided to try lemmatization, which is slightly slower but more accurate than stemmers, because it removes affixes _only_ if the resulting word is in its dictionary. I chose to use the WordNet Lemmatizer for the task, which ultimately found 40 more correct tags than the Porter Stemmer. 

This evolved to the feature:\
<lemmatizedToken | lemmatized version of present token>\
I was now at an F1 score of 76.39 in the development corpus. 

### Recording the features of previous and following words
This included implementing the following features:\
<prevToken | one word behind>\
<prevPrevToken | two words behind>\
<nextToken | one word  ahead>\
<nextNextToken | two words ahead>\
<prevPOS | POS of one word behind>\
<prevPrevPOS | POS of two words behind>\
<nextPOS | POS of one word ahead>\
<nextNextPOS | POS of two words ahead> 

This section proved to be by far the strangest section as I experimented with combinations of these features. For reasons unknown, the implementation of previousToken and prevPOS alone resulted in the highest F1 score of 80.00 in the development corpus. The addition of prevPrevToken/POS, nextToken/POS, or nextNextToken/POS lowered the overall F1 score, sometimes reaching as low as 56.00.

So,\
<prevToken | one word behind>\
<prevPOS | POS of one word behind>\
were used to raise my F1 score to 80.00.

### Current Word Features     
My final implemented features involved the token itself:\
<isCapital | whether or not the word starts with a capital>\
<isPunctuation | whether or not the word is just a punctuation symbol>\
<isNumber | whether or not the word is just a number>\
<finalLetter | final character of the present word> 

Implementing these descriptive features marginally boosted the F1 score of my overall development corpus output, from an 80.00 to an 81.16. 

### '@@' Symbol and Previous BIO Tag
Also, as recommended, I tried adding the special symbol '@@' to refer to the previous BIO tag, which simulates a (bigram) MEMM:\
<prevBIO_tag | always '@@'>\
However, this resulted in a low precision for correct noun groups, and lowered my overall F1 score to 75.07, so I decided not to implement it.
