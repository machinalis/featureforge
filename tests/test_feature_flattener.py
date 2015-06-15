# -*- coding: utf-8 -*-
from collections import Counter
import random
import unittest

from future.builtins import range, str

import numpy
import scipy

from featureforge.flattener import FeatureMappingFlattener


class Person(object):
    # Hashable object example used on tests
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __lt__(self, other):
        return (self.age, self.name) < (other.age, other.name)

    def __hash__(self):
        return hash((self.age, self.name))

    def __eq__(self, other):
        return (self.age, self.name) == (other.age, other.name)


PEOPLE = [Person('John', 23), Person('Ana', 55), Person('Maria', 3),
          Person('Peter', 11), Person('Rachel', 31)]


class TestFeatureMappingFlattener(unittest.TestCase):
    DRINKS = [u"pepsi", u"coca", u"nafta"]

    def _get_random_tuples(self):
        for _ in range(100):
            t = (random.randint(0, 100),
                 random.randint(-3, 3),
                 random.random() * 10 + 5,
                 random.choice(self.DRINKS),
                 [random.randint(0, 3) or random.random() for _ in range(5)],
                 random.random(),
                 )
            yield t

    def test_fit_empty(self):
        V = FeatureMappingFlattener()
        self.assertRaises(ValueError, V.fit, [])

    def test_fit_ok(self):
        random.seed("sofi needs a ladder")
        X = list(self._get_random_tuples())
        V = FeatureMappingFlattener()
        V.fit(X)
        V = FeatureMappingFlattener()
        # Check that works for one dict
        V.fit([next(self._get_random_tuples())])

    def test_fit_bad_values(self):
        V = FeatureMappingFlattener()
        self.assertRaises(ValueError, V.fit, [tuple()])
        self.assertRaises(ValueError, V.fit, [({},)])
        self.assertRaises(ValueError, V.fit, [([1], u"a"), ([], u"a")])
        self.assertRaises(Exception, V.fit, [(random,)])
        self.assertRaises(ValueError, V.fit, [([1, u"a"],)])
        self.assertRaises(ValueError, V.fit, [(u"a",), (1,)])

    def test_transform_empty(self):
        X = list(self._get_random_tuples())
        for sparse in [True, False]:
            V = FeatureMappingFlattener(sparse=sparse)
            V.fit(X)
            Z = V.transform([])
            self.assertEqual(Z.shape[0], 0)

    def test_transform_ok(self):
        random.seed("i am the program")
        X = list(self._get_random_tuples())
        random.seed("dream on")
        Y = self._get_random_tuples()
        for sparse in [True, False]:
            V = FeatureMappingFlattener(sparse=sparse)
        V.fit(X)
        Z = V.transform(Y)
        n = 100
        m = 4 + 3 + 5  # 3 float, 1 enum, 1 list
        self.assertEqual(Z.shape, (n, m))
        d = next(self._get_random_tuples())
        Z = V.transform([d])  # Test that works for one dict too
        self.assertEqual(Z.shape, (1, m))

    def test_transform_returns_a_matrix(self):
        random.seed("lady smith")
        X = list(self._get_random_tuples())
        random.seed("black mambazo")
        Y = list(self._get_random_tuples())
        for sparse in [True, False]:
            V = FeatureMappingFlattener(sparse=sparse)
            V.fit(X)
            Z = V.transform(Y)
            if sparse:
                self.assertIsInstance(Z, scipy.sparse.csr_matrix)
            else:
                self.assertIsInstance(Z, numpy.ndarray)

    def test_transform_produce_the_expected_values_on_the_result(self):
        random.seed("lady smith")
        X = self._get_random_tuples()
        random.seed("black mambazo")
        Y = list(self._get_random_tuples())
        V = FeatureMappingFlattener(sparse=False)
        V.fit(X)
        Z = V.transform(Y)
        for y, z in zip(Y, Z):
            for i, v in enumerate(y):
                if isinstance(v, (int, float)):
                    vector_idx = V.indexes[(i, None)]
                    self.assertEqual(v, z[vector_idx])
                elif isinstance(v, str):
                    # we know that there's only ENUM type, with DRINKS
                    vector_idx = V.indexes[(i, v)]
                    self.assertEqual(1.0, z[vector_idx])
                    for other_value in self.DRINKS:
                        if other_value != v:
                            vector_idx = V.indexes[(i, other_value)]
                            self.assertEqual(0.0, z[vector_idx])
                else:
                    # It's an array
                    for j, v_j in enumerate(v):
                        vector_idx = V.indexes[(i, j)]
                        self.assertEqual(v_j, z[vector_idx])

    def test_transform_bad_values(self):
        random.seed("king of the streets")
        X = list(self._get_random_tuples())
        d = X.pop()
        for sparse in [True, False]:
            V = FeatureMappingFlattener(sparse=sparse)
        V.fit(X)
        dd = tuple(list(d)[:-1])  # Missing value
        self.assertRaises(ValueError, V.transform, [dd])
        dd = d + (10, )  # Extra value
        self.assertRaises(ValueError, V.transform, [dd])
        dd = tuple([u"a string"] + list(d)[1:])  # Changed type
        self.assertRaises(ValueError, V.transform, [dd])

    def test_fit_transform_empty(self):
        for sparse in [True, False]:
            V = FeatureMappingFlattener(sparse=sparse)
            self.assertRaises(ValueError, V.fit_transform, [])

    def test_fit_transform_ok(self):
        random.seed("a kiss to build a dream on")
        X = list(self._get_random_tuples())
        for sparse in [True, False]:
            V = FeatureMappingFlattener(sparse=sparse)
            Z = V.fit_transform(X)
            n = 100
            m = 4 + 3 + 5  # 4 float, 1 enum, 1 list
            self.assertEqual(Z.shape, (n, m))
            d = next(self._get_random_tuples())
            Z = V.transform([d])  # Test that works for one dict too
            self.assertEqual(Z.shape, (1, m))

    def test_fit_transform_bad_values(self):
        random.seed("king of the streets")
        X = list(self._get_random_tuples())
        d = X.pop()
        for sparse in [True, False]:
            V = FeatureMappingFlattener(sparse=sparse)

            # Typical fit failures
            self.assertRaises(ValueError, V.fit_transform, [tuple()])
            self.assertRaises(ValueError, V.fit_transform, [({},)])
            self.assertRaises(ValueError, V.fit_transform, [([1], u"a"), ([], u"a")])
            self.assertRaises(Exception, V.fit_transform, [(random,)])
            self.assertRaises(ValueError, V.fit_transform, [([1, u"a"],)])
            self.assertRaises(ValueError, V.fit_transform, [("a",), (1,)])

            # Typical transform failures
            bad = X + [tuple(list(d)[:-1])]  # Missing value
            self.assertRaises(ValueError, V.fit_transform, bad)
            bad = X + [d + (10, )]  # Extra value
            self.assertRaises(ValueError, V.fit_transform, bad)
            bad = X + [tuple([u"a string"] + list(d)[1:])]  # Changed type
            self.assertRaises(ValueError, V.fit_transform, bad)

    def test_fit_transform_equivalent(self):
        random.seed("j0hny guitar")
        X = list(self._get_random_tuples())

        for sparse in [True, False]:
            # fit + transform
            A = FeatureMappingFlattener(sparse=sparse)
            A.fit(X)
            YA = A.transform(X)

            # fit_transform
            B = FeatureMappingFlattener(sparse=sparse)
            YB = B.fit_transform(X)

            if sparse:
                self.assertTrue(numpy.array_equal(YA.todense(), YB.todense()))
            else:
                self.assertTrue(numpy.array_equal(YA, YB))
            self.assertEqual(A.indexes, B.indexes)
            self.assertEqual(A.reverse, B.reverse)

    def test_fit_transform_consumes_data_only_once(self):
        random.seed("a kiss to build a dream on")
        X = list(self._get_random_tuples())
        X_consumable = (x for x in X)
        V1 = FeatureMappingFlattener(sparse=False)
        V1.fit(X)
        Z1 = V1.transform(X)
        Z2 = V1.fit_transform(X_consumable)
        self.assertTrue(numpy.array_equal(Z1, Z2))

    def test_sparse_is_equivalent(self):
        random.seed("jingle dingle")
        X = list(self._get_random_tuples())

        # fit + transform
        A = FeatureMappingFlattener(sparse=True)
        YA = A.fit_transform(X).todense()

        # fit_transform
        B = FeatureMappingFlattener(sparse=False)
        YB = B.fit_transform(X)

        self.assertTrue(numpy.array_equal(YA, YB))

    def test_sparse_single_zero(self):
        random.seed("something about us")
        V = FeatureMappingFlattener(sparse=True)
        abc = [chr(i) for i in range(65, 123)]
        X = [
            (set(random.choice(abc) for _ in range(20)), )
            for _ in range(7)
        ]
        element = chr(32)  # Clearly outside what was seen at training
        V.fit(X)
        X = V.transform([(set(element), )])
        self.assertEqual(X.shape[0], 1)


class TestBagOfWordsFit(unittest.TestCase):

    def make_every_list_(self, X, what):
        # "what" must be a type, like set or tuple
        for x in X:
            xt = []
            for xi in x:
                if isinstance(xi, list):
                    xt.append(what(xi))
                else:
                    xt.append(xi)
            yield tuple(xt)

    def check_fit_ok(self, X):
        V = FeatureMappingFlattener()
        V.fit(X)
        V.fit(list(self.make_every_list_(X, set)))
        V.fit(list(self.make_every_list_(X, tuple)))

    def check_fit_fails(self, X):
        V = FeatureMappingFlattener()
        self.assertRaises(ValueError, V.fit, X)
        self.assertRaises(ValueError, V.fit,
                          list(self.make_every_list_(X, set)))
        self.assertRaises(ValueError, V.fit,
                          list(self.make_every_list_(X, tuple)))

    def test_fit_ok_a_bag_with_seq_of_strings(self):
        X = [([u'one', u'two'], ),
             ([u'four', u'two', u'four'], )
             ]
        self.check_fit_ok(X)

    def test_fit_ok_a_bag_with_seq_of_hashables(self):
        X = [(PEOPLE[:2], ),
             (PEOPLE[:], )
             ]
        self.check_fit_ok(X)

    def test_fit_fails_when_bag_elements_are_from_mixed_types(self):
        X = [([u'one', PEOPLE[0]], ),
             ([u'four', PEOPLE[3], u'four'], )
             ]
        self.check_fit_fails(X)

    def test_fit_fails_when_the_successive_bags_are_of_different_type(self):
        # First is for people, later for strings... That's not good.
        X = [(PEOPLE[:2], ),
             ([u'four', u'two', u'four'], )
             ]
        self.check_fit_fails(X)
        # Even if the initial row is empty, when finally discovered the type,
        # is checked
        X.insert(0, ([], ))
        self.check_fit_fails(X)

    def test_fit_fails_a_tuple_elem_with_set_of_numbers(self):
        X = [(set([1, 2]), ),
             (set([4.0, 2.2, 4.0]), )
             ]
        self.check_fit_fails(X)


class TestBagOfWordsTransform(unittest.TestCase):
    COLORS = [u"blue", u"red", u"yellow", u"green"]

    def _get_random_tuples(self):
        bag_len_1 = random.randint(0, 4)
        bag_len_2 = random.randint(0, 4)
        bag_list = [random.choice(PEOPLE) for i in range(bag_len_1)]
        bag_set = set([random.choice(self.COLORS) for i in range(bag_len_2)])
        for _ in range(100):
            t = (bag_list, bag_set)
            yield t
        # Just to be sure that always all people and all colors were returned
        # at least once
        yield (list(PEOPLE), set(self.COLORS))

    def test_transform_produce_expected_values_on_the_result(self):
        random.seed("Lady smith")
        X = list(self._get_random_tuples())
        random.seed("black mambazo")
        Y = list(self._get_random_tuples())
        V = FeatureMappingFlattener(sparse=False)
        V.fit(X)
        Z = V.transform(Y)
        for y, z in zip(Y, Z):
            for i, v_seq in enumerate(y):
                assert isinstance(v_seq, (list, set, tuple))
                # we know that there's only Bag-of-strings type, with COLORS
                # and a Bag of Persons
                counter = Counter(v_seq)
                for v, v_count in (counter.items()):
                    vector_idx = V.indexes[(i, v)]
                    self.assertEqual(v_count, z[vector_idx])

    def test_transforming_non_fitted_word_is_ignored(self):
        X = [(self.COLORS[:-2],),
             (self.COLORS[:-1], )
             ]
        # never fited with self.COLORS[-1]
        known_colors = len(self.COLORS) - 1
        V = FeatureMappingFlattener(sparse=False)
        V.fit(X)
        Y = [(self.COLORS[-1:], ),  # the unknown color only
             (self.COLORS[:], ),
             ]
        Z = V.transform(Y)
        self.assertTrue(numpy.array_equal(Z[0], [0.0] * known_colors))
        self.assertTrue(numpy.array_equal(Z[1], [1.0] * known_colors))

    def test_sparse_is_equivalent(self):
        random.seed("the man who sold the world")
        X = list(self._get_random_tuples())
        # fit + transform
        A = FeatureMappingFlattener(sparse=True)
        YA = A.fit_transform(X).todense()
        # fit_transform
        B = FeatureMappingFlattener(sparse=False)
        YB = B.fit_transform(X)
        self.assertTrue(numpy.array_equal(YA, YB))
