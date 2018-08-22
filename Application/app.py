'''
Detection of aggressive tweets

Requirements:
pickle, os, flask, wtforms, mytextpreprocessing
'''

import pickle
import os
from flask import Flask 
from flask import render_template, request
from wtforms import Form, TextAreaField, validators
from mytextpreprocessing import TextPreprocessor, FrequencyExtractor
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
models_dict = {'svm': 'rbfsvm.p',
               'xgb': 'xgb.p',
               'bag': 'baggingtree.p',
               'lr': 'logisticregression.p'}
rnn_models_list = ['birnn.h5', 'lstmrnn.h5']

with open(os.path.join(cur_dir, 'similarity.p'), 'rb') as file:
    sim = pickle.load(file)


def load_clf(clf_name):
    if clf_name == 'rnn':
        clf = load_model(os.path.join(cur_dir, rnn_models_list[0]))
        global graph
        graph = tf.get_default_graph()
        global wordToIndex
        with open(os.path.join(cur_dir, 'wordToIndex.p'), 'rb') as file:
            wordToIndex = pickle.load(file)
    else:
        with open(os.path.join(cur_dir, models_dict[clf_name]), 'rb') as file:
            clf = pickle.load(file)
    return clf

def classify(tweet, clf_name):
    labels = {0: 'Nonaggressive', 1: 'Aggressive'}
    X = [tweet]
    clf = load_clf(clf_name)
    if clf_name == 'rnn':
        X_rnn = wordToIndex.transform(X)
        with graph.as_default():
            y_pred_proba = clf.predict(X_rnn)[0, 0]
        y_pred = 1 if y_pred_proba >= 0.5 else 0
    else:
        y_pred = clf.predict(X)[0]
        y_pred_proba = clf.predict_proba(X)[0, y_pred]
    return labels[y_pred], y_pred_proba

def compute_similarity(tweet):
    X = [tweet]
    similarity = sim.transform(X)[0]
    return similarity

class TweetForm(Form):
    content = TextAreaField('', [validators.DataRequired(),
                                 validators.length(max=280)])

@app.route('/')
def index():
    form = TweetForm(request.form)
    return render_template('question.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = TweetForm(request.form)

    if request.method == 'POST' and form.validate():
        tweet = request.form['content']
        clf_name = request.form.get('clf')
        pred, proba = classify(tweet, clf_name)
        similarity = compute_similarity(tweet)
        return render_template('answer.html', 
                               content=tweet, 
                               prediction=pred, 
                               probability=round(proba * 100, 2),
                               similarity=round(similarity, 2)
                               )

    return render_template('question.html', form=form)

if __name__ == '__main__':
    app.run()