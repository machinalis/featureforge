# -*- coding: utf-8 -*-
import logging
import numpy

from schema import Schema, SchemaError, Use, Or

from feature_bench.cache import disk_cache, lru_cache

logger = logging.getLogger(__name__)


class SequenceValidator(object):
    def __init__(self, size=None):
        if size is None or isinstance(size, int):
            self.size = size
        else:
            seq = SequenceValidator().validate(size)
            self.size = len(seq)

    def validate(self, x):
        if not (isinstance(x, list) or isinstance(x, tuple) or
                isinstance(x, numpy.ndarray)):
            raise SchemaError("Sequence is not list, tuple or numpy array", [])
        if isinstance(x, numpy.ndarray):
            if x.dtype.kind != "f":
                raise SchemaError("Array dtype must be float, "
                                  "but was {}".format(x.dtype), [])
            x = x.ravel()
        if len(x) == 0:
            raise ValueError("Expecting a non-empty sequence but "
                             "got {}".format(x))
        if self.size is not None and len(x) != self.size:
            raise SchemaError("Expecting sequence length {} but got "
                              "{}".format(self.size, len(x)), [])
        if not isinstance(x, numpy.ndarray):
            for value in x:
                if not isinstance(value, (int, float)):
                    raise SchemaError("Values in sequence are expected to be "
                                      "numeric", [])
            x = numpy.array(x, dtype=float)
        return x

    def __str__(self):
        size = self.size
        if size is None:
            size = ""
        return "SequenceValidator({})".format(size)

    def __repr__(self):
        return str(self)


class VectorizerCacheKey(object):

    def __init__(self, X, vectorizer):
        self.X = X
        self.vectorizer = vectorizer

    def _key(self):
        return id(self.X)

    def __eq__(self, other):
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


class FeatureMappingVectorizer(object):
    """
    This class maps feature dicts into numpy arrays.
    Strictly speaking, maps iterables of feature dicts into bidimensional
    numpy arrays such that if the array shape is (N, M) then there was N
    elements in the iterable and there are M features.

    A feature dict is a python dictionary of the shape:
        {
            "key1": 3,  # Any int
            "key2": u"value",  # Any basestring
            "key3": [1, 5, 9]  # A list of integers
        }
    Keys are meant to be feature names, valid types are str and unicode
    Values are:
        - int/float
        - str/unicode: Are meant to be enumerated types and are one-hot
          encoded.
        - list/tuple/array of integers/floats: A convenience method to pack
          several numbers togheter but otherwise equivalent to giving each
          number in the list a unique key in the dict.

    The vectorizer needs to be _fitted_ to the available feature dictionaries
    before being able to transform feature dicts to numpy arrays. This is
    because during fitting:
        - The dimension of the output array is calculated.
        - A mapping between dict keys and output array indexes is fixed.
        - A schema of the data for validation is inferred.
        - one-hot encoding values are learned.
        - Validation is applied to the data being fitted.

    Validation checks:
        - Types comply with the above description.
        - key/value pairs don't have different types between different dicts.
        - No key/value pairs are missing (from what is learnt during fitting).
        - No extra key/value pair is present.

    After fitting the instance is ready to transform new feature dicts into
    numpy arrays as long as they comply with the schema inferred during
    fitting.
    """
    def __init__(self, data_hash_key=None):
        if data_hash_key is not None:
            self.disk_cache_key_fit = 'vectorization_fit_%s' % data_hash_key
            self.disk_cache_key_transform = 'vectorization_transform_%s' % data_hash_key
        else:
            self.disk_cache_key_fit = None
            self.disk_cache_key_transform = None

    def fit(self, X, y=None):  # `y` is to comply with sklearn estimator
        return self._wrapcall(self.cached_fit, X)

    def transform(self, X, y=None):
        return self._wrapcall(self.cached_transform, X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def _wrapcall(self, method, X):
        if isinstance(X, dict):
            X = [X]
        try:
            return method(X)
        except SchemaError as e:
            raise ValueError(*e.args)

    def _iter_valid(self, X, schema, validator):
        for fdict in X:
            for key in fdict:
                if key not in schema:
                    raise ValueError("Extra key {!r} not seen "
                                     "previously".format(key))
            yield validator.validate(fdict)

    def cached_fit(self, X):
        logger.info("Lookup vectorizer.fit in cache, id=%d", id(X))
        k = VectorizerCacheKey(X, self)
        self.schema, self.validator, self.indexes = self._fit(k, self.disk_cache_key_fit)
        return self

    @staticmethod
    @lru_cache(maxsize=4)
    @disk_cache
    def _fit(k, disk_cache_key):
        X = k.X
        try:
            first = X[0]
        except IndexError:
            raise ValueError("Cannot fit with an empty dataset")
        logger.info("Starting vectorizer.fit id=%d", id(X))
        Schema({Or(str, unicode): Or(int, float, basestring,
                                     SequenceValidator())}).validate(first)
        indexes = {}
        reverse = []
        schema = {}
        for name, data in first.iteritems():
            if isinstance(data, (int, float)):
                indexes[name] = len(indexes)
                reverse.append(name)
                type_ = Use(float)
            elif isinstance(data, basestring):
                type_ = basestring
            else:
                type_ = SequenceValidator(data)
                for i in xrange(type_.size):
                    indexes[(name, i)] = len(indexes)
                    reverse.append((name, i))
            schema[name] = type_
        validator = Schema(schema)

        for fdict in k.vectorizer._iter_valid(X, schema, validator):
            for name, data in fdict.iteritems():
                if isinstance(data, basestring):
                    key = (name, data)
                    if key not in indexes:
                        indexes[key] = len(indexes)
                        reverse.append(key)

        logger.info("Finished vectorizer.fit id=%d", id(X))
        return schema, validator, indexes

    def cached_transform(self, X):
        logger.info("Lookup vectorizer.transform in cache, id=%d", id(X))
        k = VectorizerCacheKey(X, self)
        return self._transform(k, self.disk_cache_key_transform)

    @staticmethod
    @lru_cache(maxsize=4)
    @disk_cache
    def _transform(k, disk_cache_key):
        X = k.X
        self = k.vectorizer
        logger.info("Starting vectorizer.transform, id=%d", id(X))
        matrix = []

        for i, fdict in enumerate(self._iter_valid(X, self.schema, self.validator)):
            vector = numpy.zeros(len(self.indexes), dtype=float)
            for name, data in fdict.iteritems():
                if isinstance(data, float):
                    j = self.indexes[name]
                    vector[j] = data
                elif isinstance(data, basestring):
                    if (name, data) in self.indexes:
                        j = self.indexes[(name, data)]
                        vector[j] = 1.0
                else:
                    j = self.indexes[(name, 0)]
                    assert self.indexes[(name, len(data) - 1)] == \
                           j + len(data) - 1
                    vector[j:j + len(data)] = data
            matrix.append(vector.reshape((1, -1)))

        if not matrix:
            return numpy.zeros((0, len(self.indexes)))
        result = numpy.concatenate(matrix)
        logger.info("Finished vectorizer.transform, id=%d", id(X))
        logger.info("Matrix has size %sx%s" % result.shape)
        return result


class CacheOfSizeOne(object):
    """ Function wrapper that provides caching for the last value evaluated."""
    f = None

    def __init__(self, f):
        self.f = f
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        if args != self.args or kwargs != self.kwargs:
            self.result = self.f(*args, **kwargs)
            self.args = args
            self.kwargs = kwargs
        return self.result

    def __getattr__(self, name):
        return getattr(self.f, name)
