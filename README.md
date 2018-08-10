# aggressive_tweets
The final project for Bootcamp Data Science

## Problem:<br/>
Detection of aggressive tweets

## Dataset:<br/>
https://www.kaggle.com/dataturks/dataset-for-detection-of-cybertrolls/home<br/>
The dataset has 20001 tweets (in english) which are labeled (by human) as:
- 1 (Cyber-Aggressive; 7822 items)
- 0 (Non Cyber-Aggressive; 12179 items)

## Files:
* _myutils.py_:
  - plotting functions.<br/>
  Requirements:<br/>
    matplotlib, numpy, sklearn, itertools, re

* _PreparingData.ipynb_:
  - reading the original dataset,
  - keeping only relevant information,
  - dividing the data into the training and test part.<br/>
  Requirements:<br/>
    json, os, numpy, pandas, sklearn

* _Baseline.ipynb_:
  - analysis of the baseline model.<br/>
  Requirements:<br/>
    numpy, pandas, sklearn, nltk, string, myUtils

* _TweetAnalysis.ipynb_:
  - analysis of punctuation,
  - analysis of words.<br/>
  Requirements:<br/>
    numpy, pandas, sklearn, nltk, string, re, scipy

* _ClassificationOfTweets.ipynb_:
  - advanced preprocessing,
  - hyperparameter tuning.<br/>
  Requirements:<br/>
    numpy, pandas, sklearn, nltk, string, re, scipy, myUtils, pprint, pickle, os

## Application:<br/>
To run the application, run the app.py file<br/>
and open given url (default: http://127.0.0.1:5000/) in the browser.