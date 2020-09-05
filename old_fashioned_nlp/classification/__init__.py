#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-08-31 18:04:33
# @Author       : Chenghao Mou (chenghao@armorblox.com)

from old_fashioned_nlp.classification.sklearn_classifiers import TfidfLinearSVCClassifier
from old_fashioned_nlp.classification.catboost_classifiers import TfidfCatBooostClassifier

__all__ = ['TfidfLinearSVCClassifier', 'TfidfCatBooostClassifier']