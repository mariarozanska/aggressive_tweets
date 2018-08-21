# aggressive_tweets
The final project for Data Science Bootcamp

## Problem:<br/>
Detection of aggressive tweets

The best model: SVM with kernel='rbf' - accuracy 0.96

## Dataset:<br/>
https://www.kaggle.com/dataturks/dataset-for-detection-of-cybertrolls/home<br/>
The dataset has 20001 tweets (in english) which are labeled (by human) as:
- 1 (Cyber-Aggressive; 7822 items)
- 0 (Non Cyber-Aggressive; 12179 items)

## Files:
* _myutils.py_:
  - plotting functions,
  - functions which find patterns in the data and compute statistics of the matching.<br/>
  Requirements:<br/>
    matplotlib, numpy, sklearn, itertools, re

* _PreparingData.ipynb_:
  - reading the original dataset,
  - keeping only relevant information,
  - dividing the data into the training, validation and test part.<br/>
  Requirements:<br/>
    json, os, numpy, pandas, sklearn, matplotlib

* _Baseline.ipynb_:
  - analysis of the baseline model (CountVectorizer, LogisticRegression).<br/>
  Requirements:<br/>
    numpy, pandas, sklearn, nltk, string, myutils

* _TweetAnalysis.ipynb_:
  - analysis of punctuation,
  - analysis of words,
  - stemming vs lemmatization,
  - analysis of sentences.<br/>
  Requirements:<br/>
    numpy, pandas, sklearn, nltk, string, re, scipy, mytextpreprocessing

* _mytextpreprocessing.py_:
  - feature extraction,
  - text preprocessing.<br/>
  Requirements:<br/>
    nltk, numpy, re, sklearn, bs4, scipy, keras

* _ClassificationOfTweets.ipynb_:
  - advanced preprocessing,
  - hyperparameter tuning,
  - classification.<br/>
  Requirements:<br/>
    numpy, pandas, sklearn, nltk, re, scipy, myutils, pprint, pickle, os, mytextpreprocessing

* _Test.ipynb_:
  - testing of models.<br/>
  Requirements:<br/>
    numpy, pandas, pickle, os, mytextpreprocessing

## Data:<br/>
* Original dataset (20001 items): Dataset_for_Detection_of_Cyber-Trolls.json
* Training dataset (12776 items): train.json
* Validation dataset (3194 items): valid.json
* Test dataset (3993 items): test.json
* Stopwords which equally occurs in aggressive and nonaggressive training tweets: irrelevant_stopwords.csv

## Application:<br/>
The application uses the best model.

To run the application, run the app.py file<br/>
and open given url (default: http://127.0.0.1:5000/) in the browser.

Requirements:<br/>
    pickle, os, flask, wtforms, sklearn, nltk, string, re, bs4