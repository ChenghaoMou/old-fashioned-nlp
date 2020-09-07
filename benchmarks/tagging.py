#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-09-06 21:05:58
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Tagging benchmarks."""
import nltk
from loguru import logger

from old_fashioned_nlp.tagging import CharTfidfTagger

nltk.download("conll2002")


def benchmark_tagging():

    train_tokens, train_pos, train_ner = zip(
        *[zip(*e) for e in nltk.corpus.conll2002.iob_sents("esp.train")]
    )
    test_tokens, test_pos, test_ner = zip(
        *[zip(*e) for e in nltk.corpus.conll2002.iob_sents("esp.testb")]
    )

    model = CharTfidfTagger()
    model.fit(train_tokens, train_pos)
    logger.info(f"CONLL POS score: {model.score(test_tokens, test_pos)}")

    model.fit(train_tokens, train_ner)
    logger.info(f"CONLL NER score: {model.score(test_tokens, test_ner)}")


if __name__ == "__main__":

    benchmark_tagging()
