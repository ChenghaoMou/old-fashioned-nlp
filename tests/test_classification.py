#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-09-05 08:28:03
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Test cases for classification."""

import pytest
from loguru import logger
from sklearn.datasets import fetch_20newsgroups

from old_fashioned_nlp.classification import (
    TfidfCatBoostClassifier,
    TfidfLinearSVCClassifier,
)


@pytest.mark.parametrize(
    "model,args,expected",
    [
        (
            TfidfCatBoostClassifier,
            {"tfidf__max_features": 100, "classifier__iterations": 10},
            0.1,
        ),
        (TfidfLinearSVCClassifier, {}, 0.6),
    ],
)
def test_classification(model, args, expected):
    data_train = fetch_20newsgroups(
        subset="train",
        categories=None,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )

    data_test = fetch_20newsgroups(
        subset="test",
        categories=None,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )
    m = model(**args)
    m.fit(data_train.data, data_train.target)
    logger.info(
        f"{m.__class__.__name__} score in test set: {m.score(data_test.data, data_test.target):.2f}"
    )
    assert (
        m.score(data_test.data, data_test.target) >= expected
    ), "Deteriorated performance"
