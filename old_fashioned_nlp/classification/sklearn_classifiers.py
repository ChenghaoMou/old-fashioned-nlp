#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-09-05 08:19:53
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Sklearn-based text classifiers."""

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

class TfidfLinearSVCClassifier(BaseEstimator):

    def __init__(self, **kwargs):
        """
        TfidfVectorizer + Calibrated Linear SVC. See get_params for details.
        """
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(sublinear_tf=True)),
            ('classifier', CalibratedClassifierCV(LinearSVC(), cv=3)),
        ])
        self.set_params(**kwargs)
    
    def set_params(self, **params):
        self.model.set_params(**params)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)
    
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)