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
models_list = ['rbfsvm.p', 'xgb.p', 'baggingtree.p', 'logisticregression.p']
rnn_model = 'rnn.h5'
isrnn = True

if isrnn:
    global rnn
    rnn = load_model(os.path.join(cur_dir, rnn_model))
    global graph
    graph = tf.get_default_graph()

    with open(os.path.join(cur_dir, 'wordToIndex.p'), 'rb') as file:
        wordToIndex = pickle.load(file)
else:
    with open(os.path.join(cur_dir, models_list[0]), 'rb') as file:
        clf = pickle.load(file)

with open(os.path.join(cur_dir, 'similarity.p'), 'rb') as file:
    sim = pickle.load(file)


def classify(tweet):
    labels = {0: 'Nonaggressive', 1: 'Aggressive'}
    X = [tweet]
    if isrnn:
        X_rnn = wordToIndex.transform(X)
        with graph.as_default():
            y_pred_proba = rnn.predict(X_rnn)[0, 0]
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
        pred, proba = classify(tweet)
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