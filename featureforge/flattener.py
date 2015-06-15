# -*- coding: utf-8 -*-
import array
from collections import Counter
import logging

from future.builtins import map, range, str
import numpy
from schema import Schema, SchemaError, Use
from scipy.sparse import csr_matrix


logger = logging.getLogger(__name__)


class FeatureMappingFlattener(object):
    """
    This class maps feature tuples into numpy/scipy matrices.

    The main benefits of using it are:
        - String one-hot encoding is handled automatically
        - Input is validated so that each row preserves its "schema"
        - Generates sparse matrices

    A feature tuple is a regular python tuple of the shape:
        (
            ...
            3,                        # Any int (or float)
            u"value",                 # Any string (str on py3, unicode on py2)
            [u"value_1", u"value_2"]  # A set, tuple or list of hashables
            [1, 5, 9]                 # A list of integers (or floats)
            ...
        )

    Tuple values are:
        - int/float
        - str/unicode: Are meant to be enumerated types and are one-hot
          encoded.
        - set, tuple or list of str/unicode or hashables: Are meant to be
          bag-of-words
        - list/tuple/array of integers/floats: A convenience method to pack
          several numbers togheter but otherwise equivalent to inserting each
          value into the feature tuple.

    The flattener needs to be _fitted_ to the available feature tuples
    before being able to transform feature tuples to numpy/scipy matrices.
    This is because during fitting:
        - The dimension of the output matrix' rows are calculated.
        - A mapping between tuple indexes and output row indexes is fixed.
        - A schema of the data for validation is inferred.
        - One-hot encoding and bag-of-words values are learned.
        - Validation is applied to the data being fitted.

    Validation checks:
        - Tuple size is always the same
        - Values' types comply with the above description.
        - The i-th value of the feature tuples doesn't have different types
          between different input tuples.

    After fitting the instance is ready to transform new feature tuples into
    numpy/scipy matrices as long as they comply with the schema inferred during
    fitting.
    """

    def __init__(self, sparse=True):
        """
        If `sparse` is `True` the transform/fit_transform methods generate a
        `scipy.sparse.csr_matrix` matrix.
        Else the transform/fit_transform generate `numpy.array` (dense).
        """
        self.sparse = sparse

    def fit(self, X, y=None):
        """Learns a mapping between feature tuples and matrix row indexes.

        Parameters
        ----------
        X : List, sequence or iterable of tuples but not a single tuple
        y : (ignored)

        Returns
        -------
        self
        """
        return self._wrapcall(self._fit, X)

    def transform(self, X, y=None):
        """Transform feature tuples to a numpy or sparse matrix.

        Parameters
        ----------
        X : List, sequence or iterable of tuples but not a single tuple
        y : (ignored)

        Returns
        -------
        Z : A numpy or sparse matrix
        """
        if self.sparse:
            return self._wrapcall(self._sparse_transform, X)
        else:
            return self._wrapcall(self._transform, X)

    def fit_transform(self, X, y=None):
        """Learns a mapping between feature tuples and matrix row indexes and
        then transforms the feature tuples to a numpy or sparse matrix.

        Parameters
        ----------
        X : List, sequence or iterable of tuples but not a single tuple
        y : (ignored)

        Returns
        -------
        Z : A numpy or sparse matrix
        """
        if self.sparse:
            return self._wrapcall(self._sparse_fit_transform, X)
        else:
            return self._wrapcall(self._fit_transform, X)

    def _wrapcall(self, method, X):
        try:
            return method(X)
        except SchemaError as e:
            raise ValueError(*e.args)

    def _add_column(self, i, value):
        key = (i, value)
        if key not in self.indexes:
            self.indexes[key] = len(self.indexes)
            self.reverse.append(key)

    def _fit_first(self, first):
        # Check for a tuples of numbers, strings or "sequences" or "bags".
        schema = Schema((int, float, str, NumberSequenceValidator(),
                        BagValidator()))
        schema.validate(first)

        if not first:
            raise ValueError("Cannot fit with no empty features")

        # Build validation schema using the first data point
        self.indexes = {}  # Tuple index to matrix column mapping
        self.reverse = []  # Matrix column to tuple index mapping
        self.schema = [None] * len(first)
        self.str_tuple_indexes = []
        self.bag_indexes = []
        for i, data in enumerate(first):
            if isinstance(data, (int, float)):
                type_ = Use(float)  # ints and floats are all mapped to float
                self._add_column(i, None)
            elif isinstance(data, str):
                type_ = str  # One-hot encoded indexes are added last
                self.str_tuple_indexes.append(i)
            else:
                # It's an iterable, maybe of numbers, maybe of hashables.
                # Given that we don't allow empty number-sequences, if empty
                # we'll consider it's a Bag. Otherwise, we'll grab any element
                # and if it's not a number we'll consider it again as a Bag.
                if len(data) != 0:
                    elem = list(data)[0]  # sets are not indexable.
                else:
                    elem = None  # Will evaluate as Not-number, which is fine.
                if type(elem) in (int, float):
                    type_ = NumberSequenceValidator(data)
                    for j in range(type_.size):
                        self._add_column(i, j)
                else:
                    type_ = BagValidator(data)
                    self.bag_indexes.append(i)
            self.schema[i] = type_
        assert None not in self.schema
        self.schema = tuple(self.schema)
        self.validator = TupleValidator(self.schema)

    def _fit_step(self, datapoint):
        for i in self.str_tuple_indexes:
            self._add_column(i, datapoint[i])
        for i in self.bag_indexes:
            # no matter if it's a list, a tuple or a set, we need to
            # register each value only once
            for elem in set(datapoint[i]):
                self._add_column(i, elem)
            # schema fitting
            self.schema[i].fit_step(datapoint[i])

    def _iter_valid(self, X, first=None):
        if first is not None:
            yield self.validator.validate(first)
        for datapoint in X:
            yield self.validator.validate(datapoint)

    def _fit(self, X):
        X = iter(X)
        try:
            first = next(X)
        except (TypeError, StopIteration):
            raise ValueError("Cannot fit with an empty dataset")
        logger.debug("Starting flattener.fit")
        # Build basic schema
        self._fit_first(first)

        if self.str_tuple_indexes or self.bag_indexes:
            # Is there anything to one-hot encode or bag-of-words encode?
            # See all datapoints looking for one-hot encodeable feature values
            for datapoint in self._iter_valid(X, first=first):
                self._fit_step(datapoint)

        logger.debug("Finished flattener.fit")
        logger.debug("Input tuple size %s, output vector size %s" %
                    (len(first), len(self.indexes)))
        return self

    def _transform_step(self, datapoint):
        vector = numpy.zeros(len(self.indexes), dtype=float)
        for i, data in enumerate(datapoint):
            if isinstance(data, float):
                j = self.indexes[(i, None)]
                vector[j] = data
            elif isinstance(data, str):
                if (i, data) in self.indexes:
                    j = self.indexes[(i, data)]
                    vector[j] = 1.0
            else:
                # ok, it's a sequence. Not sure if a Bag or a NumSeq
                if isinstance(self.schema[i], NumberSequenceValidator):
                    j = self.indexes[(i, 0)]
                    assert self.indexes[(i, len(data) - 1)] == \
                        j + len(data) - 1
                    vector[j:j + len(data)] = data
                else:
                    for word in data:
                        # "word" because bag-of-words, but remember that can
                        # be other hashable type
                        if (i, word) in self.indexes:
                            j = self.indexes[(i, word)]
                            vector[j] += 1.0
        return vector

    def _transform(self, X):
        logger.debug("Starting flattener.transform")
        matrix = []

        for datapoint in self._iter_valid(X):
            vector = self._transform_step(datapoint)
            matrix.append(vector.reshape((1, -1)))

        if not matrix:
            result = numpy.zeros((0, len(self.indexes)))
        else:
            result = numpy.concatenate(matrix)

        logger.debug("Finished flattener.transform")
        logger.debug("Matrix has size %sx%s" % result.shape)
        return result

    def _fit_transform(self, X):
        X = iter(X)
        try:
            first = next(X)
        except (TypeError, StopIteration):
            raise ValueError("Cannot fit with an empty dataset")
        logger.debug("Starting flattener.fit_transform")

        self._fit_first(first)

        matrix = []
        for datapoint in self._iter_valid(X, first=first):
            self._fit_step(datapoint)
            vector = self._transform_step(datapoint)
            matrix.append(vector.reshape((1, -1)))

        N = len(self.indexes)
        for i, vector in enumerate(matrix):
            if len(vector) == N:
                break
            # This works because one-hot encoded features go at the end
            vector = numpy.array(vector)
            vector.resize((1, N))
            matrix[i] = vector

        if not matrix:
            result = numpy.zeros((0, N))
        else:
            result = numpy.concatenate(matrix)

        logger.debug("Finished flattener.fit_transform")
        logger.debug("Matrix has size %sx%s" % result.shape)
        return result

    def _sparse_transform_step(self, datapoint):
        """
        Yields pairs (i, value) such that the row that represents `datapoint`
        fulfills `row[i] == value`.
        For valid values of `i` that are not yielded by this function it's true
        that `row[i] == 0.0` (the sparseness condition).
        """
        for i, data in enumerate(datapoint):
            if isinstance(data, float):
                j = self.indexes[(i, None)]
                if data != 0.0:
                    yield j, data
            elif isinstance(data, str):
                if (i, data) in self.indexes:
                    j = self.indexes[(i, data)]
                    yield j, 1.0
            else:
                # ok, it's a sequence. Not sure if a Bag or a NumSeq
                if isinstance(self.schema[i], NumberSequenceValidator):
                    j = self.indexes[(i, 0)]
                    assert self.indexes[(i, len(data) - 1)] == \
                        j + len(data) - 1

                    for k, data_k in enumerate(data):
                        if data_k != 0.0:
                            yield j + k, data_k
                else:
                    counted_data = Counter(data)
                    for word, count in counted_data.items():
                        # "word" because bag-of-words, but remember that can
                        # be other hashable type
                        if (i, word) in self.indexes:
                            j = self.indexes[(i, word)]
                            yield j, count

    def _sparse_transform(self, X):
        logger.debug("Starting flattener.transform")

        data = array.array("d")
        indices = array.array("i")
        indptr = array.array("i", [0])

        for datapoint in self._iter_valid(X):
            for i, value in self._sparse_transform_step(datapoint):
                data.append(value)
                indices.append(i)
            indptr.append(len(data))

        if len(indptr) == 0:
            result = numpy.zeros((0, len(self.indexes)))
        else:
            result = csr_matrix((data, indices, indptr),
                                dtype=float,
                                shape=(len(indptr) - 1, len(self.indexes)))

        logger.debug("Finished flattener.transform")
        logger.debug("Matrix has size %sx%s" % result.shape)
        return result

    def _sparse_fit_transform(self, X):
        X = iter(X)
        try:
            first = next(X)
        except (TypeError, StopIteration):
            raise ValueError("Cannot fit with an empty dataset")
        logger.debug("Starting flattener.fit_transform")

        self._fit_first(first)

        data = array.array("d")
        indices = array.array("i")
        indptr = array.array("i", [0])

        for datapoint in self._iter_valid(X, first=first):
            self._fit_step(datapoint)
            for i, value in self._sparse_transform_step(datapoint):
                data.append(value)
                indices.append(i)
            indptr.append(len(data))

        if len(indptr) == 0:
            result = numpy.zeros((0, len(self.indexes)))
        else:
            result = csr_matrix((data, indices, indptr),
                                dtype=float,
                                shape=(len(indptr) - 1, len(self.indexes)))

        logger.debug("Finished flattener.fit_transform")
        logger.debug("Matrix has size %sx%s" % result.shape)
        return result


class NumberSequenceValidator(object):
    def __init__(self, sample_data_point=None):
        if sample_data_point:
            seq = NumberSequenceValidator().validate(sample_data_point)
            self.size = len(seq)
        else:
            self.size = None

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
        return "NumberSequenceValidator({})".format(size)

    def __repr__(self):
        return str(self)


class BagValidator(object):

    def __init__(self, sample_data=None):
        # Computes elem_type from sample data.
        self.elem_type = None
        if sample_data:
            # We'll infer type from the sample data
            self.elem_type = self.infer_type_from_data(sample_data)

    def fit_step(self, x):
        # BagValidator needs to be fit on every step because since it
        # allows empty bags, you need to walk every data sample until you found
        # the first non-empty and from it infer the elements type.
        if self.elem_type is None and x:
            # We'll infer type from the sample data
            self.elem_type = self.infer_type_from_data(x)

    def infer_type_from_data(self, x):
        if x:
            return type(list(x)[0])

    def validate(self, x):
        if not isinstance(x, (list, set, tuple)):
            raise SchemaError("Sequence is not list, tuple nor set", [])
        if x:
            if self.elem_type:
                elem_type = self.elem_type
            else:
                elem_type = self.infer_type_from_data(x)
            if not all(isinstance(x_i, elem_type) for x_i in x):
                msg = "Expecting all elements to be {}".format(self.elem_type)
                raise SchemaError(msg, [])
        return x

    def __str__(self):
        elem_type = self.elem_type if self.elem_type is not None else ""
        return "BagValidator({})".format(elem_type)

    def __repr__(self):
        return str(self)


class TupleValidator(object):
    def __init__(self, types_tuple):
        self.tt = tuple(map(Schema, types_tuple))
        self.N = len(self.tt)

    def validate(self, x):
        if not isinstance(x, tuple):
            raise SchemaError("Expecting tuple, got {}".format(type(x)), [])
        if len(x) != self.N:
            raise SchemaError("Expecting a tuple of size {}, but got".format(
                              self.N, len(x)), [])
        return tuple(schema.validate(y) for y, schema in zip(x, self.tt))
