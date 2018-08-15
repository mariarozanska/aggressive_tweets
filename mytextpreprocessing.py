'''
Requirements:
nltk, numpy, re, sklearn

- FrequencyExtractor
- TextPreprocessor
'''

from sklearn.base import TransformerMixin, BaseEstimator
import nltk
import re
import numpy as np

class FrequencyExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def tokenize(self, X):
        '''Split text into words removing punctuation'''
        X_tokenized = [list(filter(None, re.split('[^\\w\'*]', doc))) for doc in X]
        return np.array(X_tokenized)

    def compute_matching_words_rate(self, regex, X):
        '''Compute the ratio of matching words to all words in a tweet.'''
        matching_words_count = np.array([len(re.findall(regex, doc)) for doc in X])

        X_tokenized = self.tokenize(X)
        words_count = np.array(list(map(len, X_tokenized)))
        words_count[words_count == 0] = 1

        matching_words_rate = matching_words_count / words_count
        return matching_words_rate

    def compute_words_with_repeating_letters_rate(self, X):
        regex = '\\b\\w*(([a-zA-Z])\\2{2,})\\w*\\b'
        matching_words_rate = self.compute_matching_words_rate(regex, X)
        return matching_words_rate

    def compute_words_with_uppercase_rate(self, X):
        regex = '\\b[A-Z]{2,}\\b'
        matching_words_rate = self.compute_matching_words_rate(regex, X)
        return matching_words_rate

    def compute_words_with_first_capital_rate(self, X):
        regex = '\\b[A-Z][a-z]+\\b'
        matching_words_rate = self.compute_matching_words_rate(regex, X)
        return matching_words_rate

    def transform(self, X):
        repeating_letters_rate = self.compute_words_with_repeating_letters_rate(X)
        uppercase_rate = self.compute_words_with_uppercase_rate(X)
        first_capital_rate = self.compute_words_with_first_capital_rate(X)

        X_transformed = np.c_[repeating_letters_rate,
                              uppercase_rate,
                              first_capital_rate
                             ]
        return X_transformed

class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, stopwords=[], stemming=False):
        self.stopwords = stopwords
        self.stemming = stemming

    def fit(self, X, y=None):
        self.stopwords_set = set(self.stopwords)
        if self.stemming:
            self.stemmer = nltk.PorterStemmer()
        return self

    def normalize_question_marks(self, X):
        # ? --> onequestionmark
        regex = '(?<!\\?)\\?(?![?!])'
        X_normalized = [re.sub(regex, ' onequestionmark ', doc) for doc in X]

        # # ??... --> manyquestionmarks
        # regex = '\\?{2,}(?!!)'
        # X_normalized = [re.sub(regex, ' manyquestionmarks ', doc) for doc in X_normalized]
        return np.array(X_normalized)
    
    def normalize_exclamation_marks(self, X):
        # ! --> oneexclamationmark
        regex = '(?<![?!])!(?!!)'
        X_normalized = [re.sub(regex, ' oneexclamationmark ', doc) for doc in X]

        # !!... --> manyexclamationmarks
        regex = '(?<!\\?)!{2,}'
        X_normalized = [re.sub(regex, ' manyexclamationmarks ', doc) for doc in X_normalized]
        return np.array(X_normalized)

    def normalize_dots(self, X):
        # . --> onedot
        regex = '(?<!\\.)\\.(?!\\.)'
        X_normalized = [re.sub(regex, ' onedot ', doc) for doc in X]

        # ... --> manydots
        regex = '\\.{2,}'
        X_normalized = [re.sub(regex, ' manydots ', doc) for doc in X_normalized]
        return np.array(X_normalized)

    def normalize_quotation_marks(self, X):
        # " --> quotationmark
        regex = '"'
        X_normalized = [re.sub(regex, ' quotationmark ', doc) for doc in X]
        return np.array(X_normalized)
    
    def normalize_repeating_letters(self, X):
        # letters repeated three or more times --> one letter
        regex = '(([a-z])\\2{2,})'
        X_normalized = [re.sub(regex, '\\2', doc, flags=re.IGNORECASE) for doc in X]
        return np.array(X_normalized)
    
    def normalize_laugh(self, X):
        # e.g. hehehe --> haha
        regex = '(b?w?a?(ha|he)\\2{1,}h?)'
        X_normalized = [re.sub(regex, ' haha ', doc, flags=re.IGNORECASE) for doc in X]
        return np.array(X_normalized)
    
    def normalize_emoticons(self, X):
        # e.g. :) --> emoticonhappyface
        regex = '[:;=8x]-?[)D\\]*]'
        X_normalized = [re.sub(regex, ' emoticonhappyface ', doc) for doc in X]
        
        # # e.g. :( --> emoticonsadface
        # regex = '[:;=8x]\'?-?[/(x#|\\[{]'
        # X_normalized = [re.sub(regex, ' emoticonsadface ', doc) for doc in X_normalized]
        
        # <3... --> emoticonheart
        regex = '<3+'
        X_normalized = [re.sub(regex, ' emoticonheart ', doc) for doc in X_normalized]
        return np.array(X_normalized)

    def translate_html_symbols(self, X):
        html_dictionary = {'&lt;|&#60;': '<',
                           '&gt;|&#62;': '>',
                           '&amp;|&#38;': '&',
                           '&quot;|&#34;': '"',
                           '&apos;|&#39;|&;|&#8217;': '\'',
                           '&#?\\w*?;': ' ',
                           '<br>': ' '
                          }

        X_translated = X
        for k, v in html_dictionary.items():
            X_translated = [re.sub(k, v, doc) for doc in X_translated]
        return np.array(X_translated)

    def normalize(self, X):
        # html symbols
        X_normalized = self.translate_html_symbols(X)

        # words
        X_normalized = self.normalize_repeating_letters(X_normalized)
        X_normalized = self.normalize_laugh(X_normalized)

        # punctuation
        X_normalized = self.normalize_emoticons(X_normalized)
        X_normalized = self.normalize_question_marks(X_normalized)
        X_normalized = self.normalize_exclamation_marks(X_normalized)
        X_normalized = self.normalize_dots(X_normalized)
        X_normalized = self.normalize_quotation_marks(X_normalized)

        # convert to lowercase
        X_normalized = [doc.lower() for doc in X_normalized]
        return np.array(X_normalized)

    def tokenize(self, X):
        '''Split text into words removing punctuation'''
        X_tokenized = [list(filter(None, re.split('[^\\w\'*]', doc))) for doc in X]
        return np.array(X_tokenized)

    def transform(self, X):
        X_normalized = self.normalize(X)

        X_tokenized = self.tokenize(X_normalized)

        # remove stopwords
        if self.stopwords_set:
            X_tokenized = [[token for token in doc_tokenized if token not in self.stopwords_set]
                           for doc_tokenized in X_tokenized]
        
        # leave stems of words
        if self.stemming:
            X_tokenized = [[self.stemmer.stem(token) for token in doc_tokenized]
                          for doc_tokenized in X_tokenized]
            
        # join list of stems/words
        X_preprocessed = [' '.join(doc_tokenized) for doc_tokenized in X_tokenized]
        return np.array(X_preprocessed)
