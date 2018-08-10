# aggressive_tweets
The final project for Bootcamp Data Science

Problem: 
Detection of aggressive tweets

Dataset: 
https://www.kaggle.com/dataturks/dataset-for-detection-of-cybertrolls/home
The dataset has 20001 tweets (in english) which are labeled (by human) as:
- 1 (Cyber-Aggressive; 7822 items)
- 0 (Non Cyber-Aggressive; 12179 items)

Files:
* myutils.py:
  - plotting functions.
  Requirements:
    matplotlib, numpy, sklearn, itertools, re
* PreparingData.ipynb:
  - reading the original dataset,
  - keeping only relevant information,
  - dividing the data into the training and test part.
  Requirements:
    json, os, numpy, pandas, sklearn
* Baseline.ipynb:
  - analysis of the baseline model.
  Requirements:
    numpy, pandas, sklearn, nltk, string, myUtils
* TweetAnalysis.ipynb:
  - analysis of punctuation,
  - analysis of words.
  Requirements:
    numpy, pandas, sklearn, nltk, string, re, scipy
* ClassificationOfTweets.ipynb:
  - advanced preprocessing,
  - hyperparameter tuning.
  Requirements:
    numpy, pandas, sklearn, nltk, string, re, scipy, myUtils, pprint, pickle, os