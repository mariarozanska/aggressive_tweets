'''
Detection of aggressive tweets

Requirements:
pickle, os, flask, wtforms, mypreprocessor
'''

import pickle
import os
from flask import Flask 
from flask import render_template, request
from wtforms import Form, TextAreaField, validators
from mypreprocessor import AdvancedTextPreprocessor

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir, 'tf_logistic_regression.p'), 'rb') as file:
    clf = pickle.load(file)

def classify(tweet):
    labels = {0: 'Nonaggressive', 1: 'Aggressive'}
    X = [tweet]
    y_pred = clf.predict(X)[0]
    y_pred_proba = clf.predict_proba(X)[0, y_pred]
    return labels[y_pred], y_pred_proba

class TweetForm(Form):
    content = TextAreaField('', [validators.DataRequired()])

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
        return render_template('answer.html', 
                               content=tweet, 
                               prediction=pred, 
                               probability=round(proba * 100, 2)
                               )

    return render_template('question.html', form=form)

if __name__ == '__main__':
    app.run()