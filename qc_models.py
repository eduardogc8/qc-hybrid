from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np


class my_vectorizer:

    def __init__(self, cv):
        self.cv = cv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.cv.transform(X)


def count_vectorizer(MIN_GRAM=1, MAX_GRAM=1, LOWER=True, max_features=None):
    #return CountVectorizer(analyzer=lambda x: x, strip_accents=None, ngram_range=(MIN_GRAM, MAX_GRAM),
    #                       token_pattern=u'(?u)\\\\b\\\\w+\\\\b', lowercase=LOWER)
    return CountVectorizer(analyzer='word', strip_accents=None, ngram_range=(MIN_GRAM, MAX_GRAM), lowercase=LOWER, max_features=max_features)


def tfidf_vectorizer(MIN_GRAM=1, MAX_GRAM=1, LOWER=True, max_features=None):
    #return TfidfVectorizer(analyzer=lambda x: x, strip_accents=None, ngram_range=(MIN_GRAM, MAX_GRAM),
    #                       token_pattern=u'(?u)\\\\b\\\\w+\\\\b', lowercase=LOWER)
    return TfidfVectorizer(analyzer='word', strip_accents=None, ngram_range=(MIN_GRAM, MAX_GRAM), lowercase=LOWER, max_features=max_features)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec)

    def fit(self, X, y):
        return self

    def transform(self, X):
        ret = np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        return ret


class HybridVectorizer(object):
    def __init__(self, word2vec, cv):
        self.word2vec = word2vec
        self.dim = len(word2vec)
        self.bow = cv

    def fit(self, X, y):
        self.bow.fit(X)
        return self

    def transform(self, X):
        ret = []
        for sentence in X:
            w2v = np.mean([self.word2vec[w] for w in sentence if w in self.word2vec]
                          or [np.zeros(self.dim)], axis=0)
            bow = self.bow.transform([sentence]).toarray()
            ret.append(np.concatenate([w2v, bow[0]]))
        ret = np.array(ret)
        return ret


class TfidfHybridVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec)
        self.tf = tfidf_vectorizer()

    def fit(self, X, y):
        self.tf.fit(X)
        return self

    def transform(self, X):
        ret = []
        for sentence in X:
            w2v = np.mean([self.word2vec[w] for w in sentence if w in self.word2vec]
                           or [np.zeros(self.dim)], axis=0)
            tf = self.tf.transform([sentence]).toarray()
            ret.append(np.concatenate([w2v, tf[0]]))
        ret = np.array(ret)
        return ret


class SequenceHybridVectorizer(object):
    def __init__(self, word2vec, tfidf=False):
        self.word2vec = word2vec
        self.dim = len(word2vec)
        self.word2weight = None
        self.tfidf = tfidf
        self.bow = count_vectorizer()

    def fit(self, X, y):
        self.bow.fit(X)
        tfidf = tfidf_vectorizer()
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        ret = []
        for sentence in X:
            vector = np.array([])
            maxWords = 16
            count = 0
            for word in sentence:
                if count < maxWords:
                    if word in self.word2vec:
                        count += 1
                        if self.tfidf:
                            vector = np.concatenate([vector, self.word2vec[word] * self.word2weight[word]])
                        else:
                            vector = np.concatenate([vector, self.word2vec[word]])
            for i in range(maxWords-count):
                vector = np.concatenate([vector, np.zeros(self.dim)])
            bow = self.bow.transform([sentence]).toarray()
            vector = np.concatenate([vector, bow[0]])
            ret.append(vector)

        ret = np.array(ret)
        return ret
