'''
Requirements:
matplotlib, numpy, itertools, re, sklearn

Plotting functions:
- plot_confusion_matrix
- plot_curve (accuracy)
- plot_learning_curve (accuracy)
- plot_validation_curve (accuracy)
'''

import matplotlib.pyplot as plt
import numpy as np
import itertools
import re

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

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

def plot_curve(x_range, train_scores, test_scores, x_label='', x_scale='linear'):
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
                                                            cv=10, n_jobs=1)
    
    plot_curve(train_sizes, train_scores, test_scores, 
               x_label='Number of samples', x_scale='linear')

def plot_validation_curve(estimator, X, y, param_name, param_range, x_scale='linear'):
    train_scores, test_scores = validation_curve(estimator=estimator, X=X, y=y,
                                                 param_name=param_name, param_range=param_range,
                                                 scoring='accuracy', cv=10, n_jobs=1)
    
    x_label = 'Parameter ' + re.search('__(.*)', param_name).group(1)
    plot_curve(param_range, train_scores, test_scores, 
           x_label=x_label, x_scale=x_scale)

