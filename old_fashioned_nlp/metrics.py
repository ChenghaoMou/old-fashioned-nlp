#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-09-06 09:09:05
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Useful metrics with confident intervals."""

import math
from typing import Any, Callable, List, Tuple

import numpy as np
from scipy.stats import norm


def metric_with_params(
    metric: Callable,
    truth: List[Any],
    predictions: List[Any],
    confidence: float = 0.95,
    two_sided: bool = True,
) -> Tuple[float, float]:
    """
    Parametric confidence interval calculation with Wilson score interval.

    Parameters
    ----------
    metric : Callable
        Metric to calculate, could be error rate, accuracy etc.
    truth : List[Any]
        List of truth labels
    predictions : List[Any]
        List of predictions
    confidence : float, optional
        Confidence value, by default 0.95
    two_sided : bool, optional
        Two tailed intervals or not, by default True

    Returns
    -------
    Tuple[float, float]
        Confidence interval of the metric

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> metric_with_params(accuracy_score, [1, 2, 1, 1, 1, 1, 3], [1, 2, 1, 1, 1, 1, 1], confidence=.8)
    [0.6876448288176897, 1.0]
    """
    cv = norm.ppf((1 + confidence) / 2.0) if two_sided else norm.ppf(confidence)
    metric_value = metric(truth, predictions)

    return [
        max(
            0.0,
            metric_value
            - cv * math.sqrt((metric_value * (1 - metric_value)) / len(truth)),
        ),
        min(
            1.0,
            metric_value
            + cv * math.sqrt((metric_value * (1 - metric_value)) / len(truth)),
        ),
    ]


def metric_with_bs(
    metric: Callable,
    truth: List[Any],
    predictions: List[Any],
    confidence: float = 0.95,
    random_state: int = 0,
    bootstraps: int = 1000,
) -> Tuple[float, float]:
    """
    Non-parametric confidence interval with bootstrapping.

    Parameters
    ----------
    metric : Callable
        Metric to be evaluated
    truth : List[Any]
        List of truth values
    predictions : List[Any]
        Lis of predictions
    confidence : float, optional
        Confidence value, by default 0.95
    random_state : int, optional
        Random state seed, by default 0
    bootstraps : int, optional
        Number of bootstrapping steps, by default 1000

    Returns
    -------
    Tuple[float, float]
        Confidence interval

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> metric_with_bs(accuracy_score, [1, 2, 1, 1, 1, 1, 3], [1, 2, 1, 1, 1, 1, 1], confidence=.8)
    [0.7142857142857143, 1.0]
    """
    truth = np.asarray(truth)
    predictions = np.asarray(predictions)

    np.random.seed(random_state)
    scores = []
    for _ in range(bootstraps):
        indices = np.random.choice(np.arange(0, len(predictions)), len(predictions))
        scores.append(metric(truth[indices], predictions[indices]))

    alpha = 100 - confidence * 100
    lower_p = alpha / 2.0
    upper_p = (100 - alpha) + (alpha / 2.0)
    return [
        max(0.0, np.percentile(scores, lower_p)),
        min(1.0, np.percentile(scores, upper_p)),
    ]
