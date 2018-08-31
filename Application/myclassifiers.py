'''
Requirements:
sklearn, mytextpreprocessing, numpy

- SimilarityClassifier
'''

from sklearn.base import TransformerMixin, BaseEstimator
from mytextpreprocessing import DocumentsSimilarity
import numpy as np


class SimilarityClassifier(BaseEstimator, TransformerMixin):
    '''Similarity classifier
    Compute similarities to aggressive and nonaggressive documents
    Parameters:
    :param class_weight: dictionary
        weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        These weights will multiply similarities
    :param preprocessor: transformer
        the fitted transformer to process texts
    '''

    def __init__(self, class_weight={}, preprocessor=None):
        self.class_weight = class_weight
        self.preprocessor = preprocessor

    def fit(self, X, y):
        self._dsim0 = DocumentsSimilarity(pos_label=0, preprocessor=self.preprocessor)
        self._dsim0.fit(X, y)
        self._dsim1 = DocumentsSimilarity(pos_label=1, preprocessor=self.preprocessor)
        self._dsim1.fit(X, y)
        return self

    def predict_proba(self, X):
        '''Return similarities'''
        nonaggressive_sims = self._dsim0.transform(X) * self.class_weight.get(0, 1)
        aggressive_sims = self._dsim1.transform(X) * self.class_weight.get(1, 1)
        similarities = np.c_[nonaggressive_sims, aggressive_sims]
        return similarities

    def predict(self, X):
        '''Return class labels'''
        similarities = self.predict_proba(X)
        predictions = np.where(similarities[:, 0] > similarities[:, 1], 0, 1)
        return predictions
