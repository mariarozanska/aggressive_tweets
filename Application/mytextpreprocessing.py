'''
Requirements:
nltk, numpy, re, sklearn, bs4, scipy, keras

- FrequencyExtractor
- SimilarityToAggressiveTweets
- TextPreprocessor
- WordToIndexTransformer
'''

from sklearn.base import TransformerMixin, BaseEstimator
import nltk
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
import re
import numpy as np
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine

from keras.preprocessing import sequence

class FrequencyExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def words_tokenize(self, X):
        '''Split texts into words removing punctuation'''
        X_tokenized = [list(filter(None, re.split('[^\\w\'*]', doc))) for doc in X]
        return np.array(X_tokenized)

    def sentences_tokenize(self, X):
        '''Split texts into sentences'''
        X_tokenized = [nltk.sent_tokenize(doc) for doc in X]
        return np.array(X_tokenized)

    def fit(self, X, y=None):
        # X_sents_tokenized = self.sentences_tokenize(X)
        # self.max_sentences_no = np.max(list(map(len, X_sents_tokenized)))
        return self

    def compute_matching_words_rate(self, regex, X):
        '''Compute the ratio of matching words to all words in a tweet'''
        matching_words_count = np.array([len(re.findall(regex, doc)) for doc in X])

        X_tokenized = self.words_tokenize(X)
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

    def compute_parts_of_speech_rates(self, X):
        TAG_LIST = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET',
                    'ADP', 'NUM', 'CONJ', 'PRT', '.', 'X']

        X_tokenized = [nltk.word_tokenize(doc) for doc in X]
        X_tagged = nltk.pos_tag_sents(X_tokenized, tagset='universal')
        freqdist_list = [nltk.FreqDist(tag for _, tag in doc_tagged)
                         for doc_tagged in X_tagged]

        parts_of_speech_rates = np.array([[freqdist.get(tag, 0) for tag in TAG_LIST]
                                          for freqdist in freqdist_list])
        words_count = np.tile(parts_of_speech_rates.sum(axis=1), (len(TAG_LIST), 1)).T
        parts_of_speech_rates = parts_of_speech_rates / words_count
        return parts_of_speech_rates

    def compute_number_of_sentences_rate(self, X):
        X_sents_tokenized = self.sentences_tokenize(X)
        sentences_count = list(map(len, X_sents_tokenized))
        no_of_sentences_rate = np.array(sentences_count) / self.max_sentences_no
        return no_of_sentences_rate

    def transform(self, X):
        '''Compute the ratio of:
        - words with repeating letters,
        - words with uppercase,
        - words with first capital,
        - parts of speech: 'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET',
                           'ADP', 'NUM', 'CONJ', 'PRT', '.', 'X'
        to all words in a tweet and
        # - a number of sentences to the maximal number in the corpus
        '''

        repeating_letters_rate = self.compute_words_with_repeating_letters_rate(X)
        uppercase_rate = self.compute_words_with_uppercase_rate(X)
        first_capital_rate = self.compute_words_with_first_capital_rate(X)
        parts_of_speech_rates = self.compute_parts_of_speech_rates(X)
        # no_of_sentences_rate = self.compute_number_of_sentences_rate(X)

        X_transformed = np.c_[repeating_letters_rate,
                              uppercase_rate,
                              first_capital_rate,
                              parts_of_speech_rates,
                              # no_of_sentences_rate
                             ]
        return X_transformed


class SimilarityToAggressiveTweets(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        '''Find the document term matrix
        for aggressive tweets consisting of one sentence'''

        X_one_aggressive_sent = np.array([doc for i, doc in enumerate(X)
                                          if len(nltk.sent_tokenize(doc)) == 1 and y[i] == 1])
        token_pattern = '(?u)\\b[a-z][a-z\'*]+\\b'
        self.vectorizer = CountVectorizer(token_pattern=token_pattern)
        self.vectorizer.fit(X_one_aggressive_sent)
        self.aggressive_sent_dtm = self.vectorizer.transform(X_one_aggressive_sent)
        return self

    def transform(self, X):
        '''Compute the similarity of data to aggressive tweets consisting of one sentence'''

        X_sent_tokenized = [nltk.sent_tokenize(doc) for doc in X]
        sent_dtms = [[self.vectorizer.transform([sent]) for sent in doc_sent_tokenized]
                     for doc_sent_tokenized in X_sent_tokenized]

        def cosine_dist(x1, x2):
            denominator = np.dot(x1, x1.T) * np.dot(x2, x2.T)
            if denominator[0][0]:
                dist = cosine(x1, x2)
            else:
                dist = 1.
            return dist

        similarities = [[[1 - cosine_dist(aggressive_doc, sent_dtm.toarray())
                          for sent_dtm in doc] for doc in sent_dtms]
                        for aggressive_doc in self.aggressive_sent_dtm.toarray()]
        X_transformed = [np.max(np.max(np.array(similarities)[:, i])) for i in range(len(X))]
        return np.array(X_transformed)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, stopwords=[], process=''):
        '''
        :param stopwords: list
            list of stopwords
        :param process: string
            * 'stem' - stemming,
            * 'lem' - lemmatization,
            * '' - nothing
        '''

        self.stopwords = stopwords
        self.process = process
        self.tagger = None

    def fit(self, X, y=None):
        if self.process == 'lem':
            train_tagged_corpus = nltk.corpus.brown.tagged_sents()
            tagger = nltk.DefaultTagger('X')
            for n in range(1, 4):
                tagger = nltk.NgramTagger(n, train_tagged_corpus, backoff=tagger)
            self.tagger = tagger
        return self

    def normalize_question_marks(self, X):
        # ? --> onequestionmark
        regex = '(?<!\\?)\\?(?![?!])'
        X_normalized = [re.sub(regex, ' onequestionmark ', doc) for doc in X]

        # ??... --> manyquestionmarks
        regex = '\\?{2,}(?!!)'
        X_normalized = [re.sub(regex, ' manyquestionmarks ', doc)
                        for doc in X_normalized]
        return np.array(X_normalized)
    
    def normalize_exclamation_marks(self, X):
        # ! --> oneexclamationmark
        regex = '(?<![?!])!(?!!)'
        X_normalized = [re.sub(regex, ' oneexclamationmark ', doc) for doc in X]

        # !!... --> manyexclamationmarks
        regex = '(?<!\\?)!{2,}'
        X_normalized = [re.sub(regex, ' manyexclamationmarks ', doc)
                        for doc in X_normalized]
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

    def normalize_emoticons(self, X):
        # e.g. :) --> emoticonhappyface
        regex = '[:;=8x]-?[)D\\]}>*]'
        X_normalized = [re.sub(regex, ' emoticonhappyface ', doc) for doc in X]

        # e.g. :( --> emoticonsadface
        regex = '[:;=8x]\'?-?[/(x#|\\[{<]'
        X_normalized = [re.sub(regex, ' emoticonsadface ', doc) for doc in X_normalized]

        # <3... --> emoticonheart
        regex = '<3+'
        X_normalized = [re.sub(regex, ' emoticonheart ', doc) for doc in X_normalized]
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

    def join_scattered_letters(self, X):
        # e.g. n e v e r --> never
        regex = '([^\\w](?:[a-z] ){4,}(?:[a-z]\\b)?)'
        X_normalized = [re.sub(regex, ' ' + '\\1'.replace(' ', '') + ' ', doc,
                               flags=re.IGNORECASE) for doc in X]
        return np.array(X_normalized)

    def translate_shortcuts(self, X):
        # e.g. u --> shortyou
        dictionary = {'\\bu\\b': 'you',
                      '\\br\\b': 'are',
                      '\\sm\\s': 'am',
                      '\\bb/c\\b': 'because',
                      '[^\\w:;/]c\\b': 'see'
                      }
        X_translated = X
        for k, v in dictionary.items():
            X_translated = [re.sub(k, ' short' + v + ' ', doc, flags=re.IGNORECASE)
                            for doc in X_translated]
        return np.array(X_translated)

    def normalize_urls(self, X):
        # e.g. http://xxx.pl --> addressurl
        regex = '\\w+://\\S+'
        X_normalized = [re.sub(regex, ' addressurl ', doc) for doc in X]
        return np.array(X_normalized)

    def normalize_emails(self, X):
        # e.g. xxx@xxx.com --> addressemail
        regex = '\\b(\\w[\\w.-]+@[\\w-][\\w.-]*\\w(?:\\.[a-zA-Z]{1,4}))\\b'
        X_normalized = [re.sub(regex, ' addressemail ', doc) for doc in X]
        return np.array(X_normalized)

    def translate_html_symbols(self, X):
        dictionary = {'&;': '\''}
        X_translated = X
        for k, v in dictionary.items():
            X_translated = [re.sub(k, v, doc) for doc in X_translated]

        X_translated = [BeautifulSoup(doc, 'html.parser').get_text()
                        for doc in X_translated]
        return np.array(X_translated)

    def normalize(self, X):
        # addresses
        X_normalized = self.normalize_urls(X)
        X_normalized = self.normalize_emails(X_normalized)

        # html symbols
        X_normalized = self.translate_html_symbols(X_normalized)

        # words
        X_normalized = self.normalize_repeating_letters(X_normalized)
        X_normalized = self.normalize_laugh(X_normalized)
        X_normalized = self.join_scattered_letters(X_normalized)
        X_normalized = self.translate_shortcuts(X_normalized)

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
        '''Split texts into words removing punctuation'''
        X_tokenized = [list(filter(None, re.split('[^\\w\'*]', doc))) for doc in X]
        return np.array(X_tokenized)

    def remove_stopwords(self, X_tokenized):
        stopwords_set = set(self.stopwords)
        X_no_stopwords = [[token for token in doc_tokenized if token not in stopwords_set]
                          for doc_tokenized in X_tokenized]
        return X_no_stopwords

    def stem(self, X_tokenized):
        stemmer = nltk.PorterStemmer()
        X_stemmed = [[stemmer.stem(token) for token in doc_tokenized]
                     for doc_tokenized in X_tokenized]
        return X_stemmed

    def lemmatize(self, X_tokenized):
        pos_dict = {'J': ADJ,
                    'R': ADV,
                    'N': NOUN,
                    'V': VERB
                    }

        X_tagged = [self.tagger.tag(doc_tokenized) for doc_tokenized in X_tokenized]

        lemmatizer = nltk.stem.WordNetLemmatizer()
        X_lemmatized = [[lemmatizer.lemmatize(token, pos=pos_dict[pos[0]])
                         if pos[0] in pos_dict else token for token, pos in doc_tagged]
                        for doc_tagged in X_tagged]
        return X_lemmatized

    def transform(self, X):
        X_normalized = self.normalize(X)

        X_tokenized = self.tokenize(X_normalized)

        if self.stopwords:
            X_tokenized = self.remove_stopwords(X_tokenized)

        if self.process == 'stem':
            X_tokenized = self.stem(X_tokenized)
        elif self.process == 'lem':
            X_tokenized = self.lemmatize(X_tokenized)

        X_preprocessed = [' '.join(doc_tokenized) for doc_tokenized in X_tokenized]
        return np.array(X_preprocessed)


class WordToIndexTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, textpreprocessor):
        self.textpreprocessor = textpreprocessor
        self.maxlen = 280

    def fit(self, X, y=None):
        X_preprocessed = self.textpreprocessor.fit_transform(X)
        X_tokenized = [doc.split(' ') for doc in X_preprocessed]

        all_words = [token for doc_tokenized in X_tokenized for token in doc_tokenized]
        unique_words = np.unique(all_words)

        self.index_to_word = unique_words.tolist()
        self.unique_words_no = len(self.index_to_word)
        self.word_to_index = {w: i for i, w in enumerate(self.index_to_word)}
        return self

    def transform(self, X):
        '''Return the data in the right form for neural networks'''

        X_preprocessed = self.textpreprocessor.transform(X)
        X_tokenized = [doc.split(' ') for doc in X_preprocessed]
        X_transformed = [[self.word_to_index.get(token, self.unique_words_no)
                          for token in doc_tokenized]
                         for doc_tokenized in X_tokenized]
        X_transformed = sequence.pad_sequences(X_transformed, maxlen=self.maxlen)
        return X_transformed
