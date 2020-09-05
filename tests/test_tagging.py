#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-09-05 08:28:03
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Test cases for tagging."""

import pytest
import nltk
from loguru import logger
from old_fashioned_nlp.tagging import CharTfidfTagger
from sklearn.datasets import fetch_20newsgroups

nltk.download('conll2002')

def test_tagging():

    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    train_tokens, train_pos, train_ner = zip(*[zip(*e) for e in train_sents])
    model = CharTfidfTagger()
    model.fit(train_tokens[:100], train_pos[:100])
    assert model.score(train_tokens[99:100], train_pos[99:100]) >= 0.3, "Deteriorated performance"