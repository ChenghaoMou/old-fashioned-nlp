#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-08-31 18:04:33
# @Author       : Chenghao Mou (chenghao@armorblox.com)

"""Tagging pipeline."""
from typing import Dict, Any

import sklearn_crfsuite

def featurizer(token) -> Dict[str, float]:

    return {
        "prefix": token[:3],
        "suffix": token[-3:]
    }

def make_tagger(**kargs):
    """
    Make a tagger.

    Returns
    -------
    Pipeline
        sklearn pipeline for sequence tagging

    Examples
    --------
    >>> tagger = make_tagger()
    >>> tagger.fit([[{'a': 1}, {'a': 2}]], [['a', 'b']])
    CRF(keep_tempfiles=None)
    >>> tagger.predict([[{'a': 1}, {'a': 2}]])
    [['a', 'b']]
    """
    
    return sklearn_crfsuite.CRF(**kargs)