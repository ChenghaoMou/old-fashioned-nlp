#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-09-05 08:19:53
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Sklearn-based text regressors."""

from sklearn.base import BaseEstimator
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR


class TfidfLinearSVR(BaseEstimator):
    def __init__(self, **kwargs):
        """
        TfidfVectorizer + Linear SVM Regressor. See get_params for details.

        Examples
        --------
        >>> model = TfidfLinearSVR(tfidf__sublinear_tf=True)
        """
        self.model = Pipeline(
            [("tfidf", TfidfVectorizer(sublinear_tf=True)), ("regressor", LinearSVR()),]
        )
        self.set_params(**kwargs)

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.model, name)
        return self.__dict__[name]


class TfidfLDALinearSVR(BaseEstimator):
    def __init__(self, **kwargs):
        """
        TfidfVectorizer + LDA + Linear SVM Regressor. See get_params for details.

        Examples
        --------
        >>> model = TfidfLDALinearSVR(tfidf__sublinear_tf=True, lda__n_components=100)
        """
        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(sublinear_tf=True)),
                ("lda", LatentDirichletAllocation(n_components=100)),
                ("regressor", LinearSVR()),
            ]
        )
        self.set_params(**kwargs)

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.model, name)
        return self.__dict__[name]
