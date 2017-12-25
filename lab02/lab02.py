import numpy
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report as report

import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

data_set = open(file="SMSSpamCollection.txt", encoding="utf-8").readlines()
filtered = [re.sub(r"[,.;\":()\[\]]", " ", data) for data in data_set]
classified_target, classified_data = zip(*[re.split(r'\s', data, 1) for data in filtered])
classified_target = numpy.array(classified_target)
classified_data = numpy.array(classified_data)

multinomial = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])


class DenseTransformer(TransformerMixin):
    def transform(self, X, *_):
        return X.toarray()

    def fit(self, *_):
        return self


gaussian = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('dense', DenseTransformer()),
    ('clf', GaussianNB())
])

bernoulli = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', BernoulliNB())
])

kfold = KFold(n_splits=10)

print("\n\n############ MULTINOMIAL STARTED ########################")
multinomial_report = [report(classified_target[test], multinomial.fit(classified_data[train], classified_target[train])
                             .predict(classified_data[test])) for train, test in kfold.split(classified_data)]
for rep in multinomial_report:
    print(rep)


print("\n\n############ GAUSSIAN STARTED ########################")
gaussian_report = [report(classified_target[test], gaussian.fit(classified_data[train], classified_target[train])
                          .predict(classified_data[test])) for train, test in kfold.split(classified_data)]
for rep in gaussian_report:
    print(rep)


print("\n\n############ BERNOULLI STARTED ########################")
bernoulli_report = [report(classified_target[test], bernoulli.fit(classified_data[train], classified_target[train])
                           .predict(classified_data[test])) for train, test in kfold.split(classified_data)]
for rep in bernoulli_report:
    print(rep)
