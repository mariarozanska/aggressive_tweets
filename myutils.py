'''
Requirements:
matplotlib, numpy, itertools, re, sklearn, pandas, scipy

Plotting functions:
- plot_confusion_matrix
- plot_accuracy_curve
- plot_learning_curve (accuracy)
- plot_validation_curve (accuracy)
- plot_roc_curve

Functions which find patterns in the data
and compute statistics of the matching:
- compute_binom_pvalue
- print_matching_statistics
- find_all_matches
- compute_matching_words_rate
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools
import re

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve

import pandas as pd
from scipy import stats


########## PLOTTING ##########

def plot_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred) / len(y_true)
    
    fig, ax = plt.subplots(figsize=(3, 3))    
    ax.matshow(conf_mat, cmap=plt.cm.Greens, alpha=0.5)
    
    labels = np.unique(y_true)
    for i, j in itertools.product(labels, repeat=len(labels)):
        ax.annotate(s='%.2f' % conf_mat[i, j], xy=(j, i), va='center', ha='center')
        
    ax.set_xlabel('Predicted label', size=14)
    ax.set_ylabel('Real label', size=14)
    plt.show()

def plot_accuracy_curve(x_range, train_scores, test_scores, x_label='', x_scale='linear'):
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_std = test_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    ax.plot(x_range, train_mean, 
            color='blue', marker='o', label='Accuracy of learning')
    ax.fill_between(x_range, train_mean - train_std, train_mean + train_std, 
                    alpha=0.3, color='blue')
    
    ax.plot(x_range, test_mean, 
            color='red', marker='s', label='Accuracy of validation')
    ax.fill_between(x_range, test_mean - test_std, test_mean + test_std, 
                    alpha=0.3, color='red')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Accuracy')
    ax.set_xscale(x_scale)
    plt.grid()
    plt.legend()
    plt.show()

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            scoring='accuracy',
                                                            cv=5, n_jobs=-1)
    
    plot_accuracy_curve(train_sizes, train_scores, test_scores,
               x_label='Number of samples', x_scale='linear')

def plot_validation_curve(estimator, X, y, param_name, param_range, x_scale='linear'):
    train_scores, test_scores = validation_curve(estimator=estimator, X=X, y=y,
                                                 param_name=param_name, param_range=param_range,
                                                 scoring='accuracy', cv=5, n_jobs=-1)
    
    x_label = 'Parameter ' + re.search('__(.*)', param_name).group(1)
    plot_accuracy_curve(param_range, train_scores, test_scores,
               x_label=x_label, x_scale=x_scale)

def plot_roc_curve(y_labels, y_proba):
    fprs, tprs, thresholds = roc_curve(y_true=y_labels, y_score=y_proba)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fprs, tprs)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    plt.show()


########## MATCHING ##########

def compute_binom_pvalue(data, column_name, p):
    '''Compute p-value for two-sided test of the null hypothesis that
    the probability of occurrence of an aggressive tweet is p.
    Parameters:
    :param data: dataframe
        data with the binary column 'label'
    :param column_name: string
        the name of the column to be checked
    Returns:
    :return pvalue: float
        the p-value
    '''

    matching_labels = data.label[data[column_name].notnull()]
    successes_no = matching_labels[matching_labels == 1].shape[0]
    trials_no = matching_labels.shape[0]
    pvalue = stats.binom_test(x=successes_no, n=trials_no, p=p)
    return pvalue

def print_matching_statistics(data, column_name, p):
    '''Print matching statistics.
    Parameters:
    :param data: dataframe
        data with the binary column 'label'
    :param column_name: string
        the name of the column to be checked
    '''

    matching_labels = data.label[data[column_name].notnull()]

    score = matching_labels.mean()
    print('The mean of labels of matching tweets = %.2f' % score)

    rate = matching_labels.shape[0] / data.shape[0]
    print('The rate of matching tweets = %.3f' % rate)

    pvalue = compute_binom_pvalue(data, column_name, p)
    print('p-value = %.3f' % pvalue)

def find_all_matches(X, y, regex):
    '''Find all matches of regex in a tweet.
    Parameters:
    :param X: array-like
        list of tweets
    :param y: array-like
        list of labels
    :param regex: string
        regular expression
    Returns:
    :return data: dataframe
        it has three columns: content, label, list of matches
    '''

    data = pd.DataFrame(np.c_[X, y], columns=['content', 'label'])
    data[regex] = data.content.apply(lambda doc: re.findall(regex, doc) if re.findall(regex, doc) != [] else None)

    return data

def compute_matching_words_rate(regex, X):
    '''Compute the ratio of matching words to all words in a tweet.
    Parameters:
    :param regex: string
        regular expression
    :param X: array-like
        list of tweets
    Returns:
    :return matching_words_rate: array
        the ratio of matching words to all words in a tweet
    '''

    matching_words_count = np.array([len(re.findall(regex, doc)) for doc in X])

    words_count = np.array([len(list(filter(None, re.split('[^\\w\'*]', doc)))) for doc in X])
    words_count[words_count == 0] = 1

    matching_words_rate = matching_words_count / words_count
    return matching_words_rate