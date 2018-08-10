from sklearn.base import TransformerMixin, BaseEstimator
import nltk
import string
import re

class AdvancedTextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, stopwords=[], punctuation='', stemming=True):
        self.stopwords = stopwords
        self.punctuation = punctuation
        self.stemming = stemming
    
    def fit(self, X, y=None):
        self.stopwords_set = set(self.stopwords)
        self.punctuation_set = set(self.punctuation)
        if self.stemming:
            self.stemmer = nltk.PorterStemmer()
        return self
    
    def get_happy_emoticons(self, X, normalize=False):
        # happy face or heart
        regex = '[:;=8x]-?[)D\\]*]|<3'
        emoticons = [' '.join(re.findall(regex, doc)) for doc in X]
        
        # normalize happy faces --> :)
        if normalize:
            emoticons = [re.sub('[;=8x]', ':', doc) for doc in emoticons]
            emoticons = [re.sub('[D\\]*]', ')', doc) for doc in emoticons]
            emoticons = [re.sub('-', '', doc) for doc in emoticons]
            
        return emoticons

    def get_question_marks(self, X):
        # one or more question marks
        regex = '\\?{1,}'
        question_marks = [' '.join(re.findall(regex, doc)) for doc in X]
        return question_marks
    
    def transform(self, X):        
        # keep happy faces and hearts
        emoticons = self.get_happy_emoticons(X)
        # keep one or more question marks
        question_marks = self.get_question_marks(X)
        
        # convert to lowercase
        X_lower = [doc.lower() for doc in X]
        # split texts into words
        X_tokenized = [nltk.word_tokenize(doc) for doc in X_lower]
        # remove punctuation and stopwords
        X_tokenized = [[token for token in doc_tokenized 
                        if token not in self.punctuation_set and token not in self.stopwords_set]
                      for doc_tokenized in X_tokenized]
        
        # leave stems of words
        if self.stemming:
            X_tokenized = [[self.stemmer.stem(token) for token in doc_tokenized]
                          for doc_tokenized in X_tokenized]
            
        # join list of stems/words, emoticons and question marks
        X_preprocessed = [' '.join(doc_tokenized + [emoticons[i], question_marks[i]])
                          for i, doc_tokenized in enumerate(X_tokenized)]
        
        return X_preprocessed