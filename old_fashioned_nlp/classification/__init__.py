#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-08-31 18:04:33
# @Author       : Chenghao Mou (chenghao@armorblox.com)

"""Old fashioned text classifiers."""

from old_fashioned_nlp.classification.sklearn_classifiers import (
    TfidfLDALinearSVC,
    TfidfLinearSVC,
)

__all__ = [
    "TfidfLinearSVC",
    "TfidfLDALinearSVC",
]
