# -*- coding: utf-8 -*-
import collections
import logging
import numpy

from schema import Schema, SchemaError, Use, Or

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


class FeatureMappingFlattener(object):
    """
    This class maps feature dicts into numpy arrays.
    Strictly speaking, maps iterables of feature dicts into bidimensional
    numpy arrays such that if the array shape is (N, M) then there were N
    elements in the iterable and M features.

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

    The flattener needs to be _fitted_ to the available feature dictionaries
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
    def fit(self, X, y=None):  # `y` is to comply with sklearn estimator
        """X must be a list, sequence or iterable of points,
        but not a single data point.
        """
        return self._wrapcall(self._fit, X)

    def transform(self, X, y=None):
        """X must be a list, sequence or iterable points,
        but not a single data point.
        """
        return self._wrapcall(self._transform, X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def _wrapcall(self, method, X):
        if not isinstance(X, collections.Iterable):
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

    def _fit(self, X):
        try:
            first = iter(X).next()
        except (TypeError, StopIteration):
            raise ValueError("Cannot fit with an empty dataset")
        logger.info("Starting flattener.fit id=%d", id(X))
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

        for fdict in self._iter_valid(X, schema, validator):
            for name, data in fdict.iteritems():
                if isinstance(data, basestring):
                    key = (name, data)
                    if key not in indexes:
                        indexes[key] = len(indexes)
                        reverse.append(key)

        logger.info("Finished flattener.fit id=%d", id(X))
        self.schema, self.validator, self.indexes = schema, validator, indexes
        return self

    def _transform(self, X):
        logger.info("Starting flattener.transform, id=%d", id(X))
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
        logger.info("Finished flattener.transform, id=%d", id(X))
        logger.info("Matrix has size %sx%s" % result.shape)
        return result
