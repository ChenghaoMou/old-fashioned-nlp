# Old Fashioned NLP

<?xml version="1.0" encoding="UTF-8"?>

<svg xmlns="http://www.w3.org/2000/svg" width="99" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="99" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h63v20H0z"/>
        <path fill="#97CA00" d="M63 0h36v20H63z"/>
        <path fill="url(#b)" d="M0 0h99v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="31.5" y="14">coverage</text>
        <text x="80" y="15" fill="#010101" fill-opacity=".3">91%</text>
        <text x="80" y="14">91%</text>
    </g>
</svg>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package aims to bring back the old fashioned NLP pipelines into your modeling workflow, providing a baseline reference when you move onto a transformer model.

## Installation

    pip install git+https://github.com/ChenghaoMou/old-fashioned-nlp.git

## Usage

### Classification

Currently, we have `TfidfLinearSVCClassifier` and `TfidfCatBoostClassifier`.

```python
from old_fashioned_nlp.classification import TfidfLinearSVCClassifier, TfidfCatBooostClassifier
from sklearn.datasets import fetch_20newsgroups

data_train = fetch_20newsgroups(subset='train', categories=None,
                                shuffle=True, random_state=42,
                                remove=('headers', 'footers', 'quotes'))

data_test = fetch_20newsgroups(subset='test', categories=None,
                            shuffle=True, random_state=42,
                            remove=('headers', 'footers', 'quotes'))

m = TfidfLinearSVCClassifier()
m.fit(data_train.data, data_train.target)
m.score(data_test.data, data_test.target)
```

### Sequence Tagging

We only have `CharTfidfTagger` right now.

```python
import nltk
from old_fashioned_nlp.tagging import CharTfidfTagger

nltk.download('conll2002')

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
train_tokens, train_pos, train_ner = zip(*[zip(*e) for e in train_sents])

model = CharTfidfTagger()
model.fit(train_tokens, train_pos)
model.score(test_tokens, test_pos)
```

### Text Cleaning

`CleanTextTransformer` can be plugged into any sklearn pipeline.

```python
transformer = CleanTextTransformer(
    replace_dates_with='DATE',
    replace_times_with='TIME',
    replace_emails_with='EMAIL',
    replace_numbers_with='NUMBER',
    replace_percentages_with='PERCENT',
    replace_money_with='MONEY',
    replace_hashtags_with='HASHTAG',
    replace_handles_with='HANDLE',
    expand_contractions=True
)
transformer.transform(["#now @me I'll log 80% entries are due by January 4th, 2017at 8:00pm contact me at chenghao@armorblox.com send me $500.00 now 3,415"])
```
