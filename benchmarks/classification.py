#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Update       : 2020-09-06 21:05:58
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Classification benchmarks."""

from loguru import logger
from nlp import load_dataset
from sklearn.metrics import classification_report

from old_fashioned_nlp.classification import TfidfLinearSVCClassifier


def benchmark_classification():

    model = TfidfLinearSVCClassifier()

    sogou = load_dataset("sogou_news")
    model.fit(
        list(map(" ".join, zip(sogou["train"]["title"], sogou["train"]["content"]))),
        sogou["train"]["label"],
    )
    logger.info(
        "SOGOU\n"
        + classification_report(
            sogou["test"]["label"],
            model.predict(
                list(
                    map(" ".join, zip(sogou["test"]["title"], sogou["test"]["content"]))
                )
            ),
        )
    )

    glue_cola = load_dataset("glue", "cola")
    model.fit(glue_cola["train"]["sentence"], glue_cola["train"]["label"])
    logger.info(
        "GLUE/COLA\n"
        + classification_report(
            glue_cola["validation"]["label"],
            model.predict(glue_cola["validation"]["sentence"]),
        )
    )

    glue_sst = load_dataset("glue", "sst2")
    model.fit(glue_sst["train"]["sentence"], glue_sst["train"]["label"])
    logger.info(
        "GLUE/SST2\n"
        + classification_report(
            glue_sst["validation"]["label"],
            model.predict(glue_sst["validation"]["sentence"]),
        )
    )

    ag_news = load_dataset("ag_news")
    model.fit(ag_news["train"]["text"], ag_news["train"]["label"])
    logger.info(
        "AG News\n"
        + classification_report(
            ag_news["test"]["label"], model.predict(ag_news["test"]["text"])
        )
    )

    allocine = load_dataset("allocine")
    model.fit(allocine["train"]["review"], allocine["train"]["label"])
    logger.info(
        "allocine\n"
        + classification_report(
            allocine["test"]["label"], model.predict(allocine["test"]["review"])
        )
    )

    yelp = load_dataset("yelp_polarity")
    model.fit(yelp["train"]["text"], yelp["train"]["label"])
    logger.info(
        "Yelp\n"
        + classification_report(
            yelp["test"]["label"], model.predict(yelp["test"]["text"])
        )
    )


if __name__ == "__main__":

    benchmark_classification()
