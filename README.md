# Old Fashioned NLP

<img src="https://raw.githubusercontent.com/ChenghaoMou/old-fashioned-nlp/master/coverage.svg"/> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package aims to bring back the old fashioned NLP pipelines into your modeling workflow, providing a baseline reference before you move onto a transformer model.

## Installation

    pip install git+https://github.com/ChenghaoMou/old-fashioned-nlp.git

## Usage

### Classification

Currently, we have `TfidfLinearSVCClassifier`, `TfidfCatBoostClassifier`, `TfidfLDALinearSVCClassifier` and `TfidfLDACatBoostClassifier`.

```python
from old_fashioned_nlp.classification import TfidfLinearSVCClassifier, TfidfCatBoostClassifier
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

## Benchmarks

### Classification
All scores are test scores using `nlp` datasets from Huggingface. See benchmarks directory for details.

SOGOU
```
              precision    recall  f1-score   support

           0       0.96      0.95      0.95     12000
           1       0.93      0.95      0.94     12000
           2       0.95      0.97      0.96     12000
           3       0.95      0.96      0.96     12000
           4       0.96      0.92      0.94     12000

    accuracy                           0.95     60000
   macro avg       0.95      0.95      0.95     60000
weighted avg       0.95      0.95      0.95     60000
```

GLUE/COLA
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       322
           1       0.69      1.00      0.82       721

    accuracy                           0.69      1043
   macro avg       0.35      0.50      0.41      1043
weighted avg       0.48      0.69      0.57      1043
```

GLUE/SST2
```
              precision    recall  f1-score   support

           0       0.84      0.77      0.80       428
           1       0.79      0.86      0.82       444

    accuracy                           0.81       872
   macro avg       0.82      0.81      0.81       872
weighted avg       0.82      0.81      0.81       872
```

Yelp
```
              precision    recall  f1-score   support

           0       0.94      0.94      0.94     19000
           1       0.94      0.94      0.94     19000

    accuracy                           0.94     38000
   macro avg       0.94      0.94      0.94     38000
weighted avg       0.94      0.94      0.94     38000
```

AG News
```
              precision    recall  f1-score   support

           0       0.94      0.91      0.92      1900
           1       0.96      0.98      0.97      1900
           2       0.90      0.89      0.89      1900
           3       0.89      0.91      0.90      1900

    accuracy                           0.92      7600
   macro avg       0.92      0.92      0.92      7600
weighted avg       0.92      0.92      0.92      7600
```

allocine
```
              precision    recall  f1-score   support

           0       0.93      0.93      0.93     10408
           1       0.92      0.93      0.92      9592

    accuracy                           0.93     20000
   macro avg       0.93      0.93      0.93     20000
weighted avg       0.93      0.93      0.93     20000
```

### Tagging

Default `CharTfidfTagger`

CONLL POS score: 0.5835184323399495
CONLL NER score: 0.15840812513116917
