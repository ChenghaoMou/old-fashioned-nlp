#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-08-31 18:04:33
# @Author       : Chenghao Mou (chenghao@armorblox.com)

"""Tagging pipeline."""
from typing import Any, Dict

import sklearn_crfsuite
import sklearn_crfsuite.metrics
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer


class CharTfidfTagger(BaseEstimator):
    def __init__(self, **kwargs):
        """
        Character-based tfidf sequence tagger.

        Examples
        --------
        >>> model = CharTfidfTagger()
        >>> model.fit([["token1", "token2"]], [["A", "B"]])
        >>> model.predict([["token1", "token2"]])
        [['A', 'B']]
        """
        self.tagger = sklearn_crfsuite.CRF()
        self.tfidf = TfidfVectorizer(analyzer="char_wb", strip_accents="ascii")

        self.set_params(**kwargs)

    def set_params(self, **params):
        self.tfidf.set_params(
            **{
                k.split("__", 1)[-1]: v
                for k, v in params.items()
                if k.startswith("tfidf__")
            }
        )
        self.tagger.set_params(
            **{
                k.split("__", 1)[-1]: v
                for k, v in params.items()
                if k.startswith("tagger__")
            }
        )

    def get_params(self, deep=True):
        params = {"tfidf__" + k: v for k, v in self.tfidf.get_params(deep).items()}
        params.update(
            {"tagger__" + k: v for k, v in self.tagger.get_params(deep).items()}
        )
        return params

    def fit(self, X, y):
        corpus = [" ".join(example) for example in X]
        self.tfidf.fit(corpus)
        features = [self.featurize(example) for example in X]
        self.tagger.fit(features, y)

    def predict(self, X):
        features = [self.featurize(example) for example in X]
        return self.tagger.predict(features)

    def score(self, X, y):
        predictions = self.predict(X)
        return sklearn_crfsuite.metrics.flat_f1_score(y, predictions, average="macro")

    def featurize(self, tokens):
        return [
            dict(
                zip(
                    self.tfidf.get_feature_names(),
                    self.tfidf.transform([token]).toarray().reshape(-1),
                )
            )
            for token in tokens
        ]
