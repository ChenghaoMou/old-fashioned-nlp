#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-08-31 18:04:33
# @Author       : Chenghao Mou (chenghao@armorblox.com)

"""Classification pipeline."""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def make_classifier(**kargs):
    """
    Make a tfidf classifier.

    Returns
    -------
    Pipeline
        sklearn pipeline for classification

    Examples
    --------
    >>> model = make_classifier(
    ... tfidf__sublinear_tf=True, 
    ... tfidf__stop_words='english', 
    ... svc__cv=3, 
    ... svc__base_estimator__multi_class='ovr'
    ... )
    """
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svc', CalibratedClassifierCV(LinearSVC()))
    ])
    model.set_params(**kargs)
    return model