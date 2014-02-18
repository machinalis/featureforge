# -*- coding: utf-8 -*-
import unittest
import random
from featureforge.flattener import FeatureMappingFlattener


class TestFeatureMappingFlattener(unittest.TestCase):
    def test_fit_empty(self):
        V = FeatureMappingFlattener()
        self.assertRaises(ValueError, V.fit, [])

    def _get_random_dicts(self):
        for _ in xrange(100):
            d = {}
            d["some integer"] = random.randint(0, 100)
            d[u"otherinteger"] = random.randint(-3, 3)
            d[u"somefłøæŧ"] = random.random() * 10 + 5
            d[u"j€nµmeæcħeid"] = random.choice([u"pepsi", u"coca", "nafta"])
            d[u"list"] = [random.randint(0, 3) or random.random()
                          for _ in xrange(5)]
            yield d

    def test_fit_ok(self):
        random.seed("sofi needs a ladder")
        X = list(self._get_random_dicts())
        V = FeatureMappingFlattener()
        V.fit(X)
        V = FeatureMappingFlattener()
        V.fit([next(self._get_random_dicts())])  # Test that works for one dict

    def test_fit_bad_values(self):
        random.seed("the alphabet city elite")
        V = FeatureMappingFlattener()
        self.assertRaises(ValueError, V.fit, [{}])  # keys are strings
        self.assertRaises(ValueError, V.fit, [{1: 1}])  # keys are strings
        self.assertRaises(ValueError, V.fit, [{"a": {}}])
        self.assertRaises(ValueError, V.fit, [{"a": []}])
        self.assertRaises(ValueError, V.fit, [{"a": random}])
        self.assertRaises(ValueError, V.fit, [{"a": [1, u"a"]}])
        self.assertRaises(ValueError, V.fit, [{"a": 1}, {"a": "a"}])

    def test_transform_empty(self):
        X = list(self._get_random_dicts())
        V = FeatureMappingFlattener()
        V.fit(X)
        Z = V.transform([])
        self.assertEqual(Z.shape[0], 0)

    def test_transform_ok(self):
        random.seed("i am the program")
        X = list(self._get_random_dicts())
        random.seed("dream on")
        Y = self._get_random_dicts()
        V = FeatureMappingFlattener()
        V.fit(X)
        Z = V.transform(Y)
        n = 100
        m = 3 + 3 + 5  # 3 float, 1 enum, 1 list
        self.assertEqual(Z.shape, (n, m))
        d = next(self._get_random_dicts())
        Z = V.transform([d])  # Test that works for one dict too
        self.assertEqual(Z.shape, (1, m))

    def test_transform_bad_values(self):
        random.seed("king of the streets")
        X = list(self._get_random_dicts())
        V = FeatureMappingFlattener()
        d = X.pop()
        V.fit(X)
        del d["some integer"]  # Missing key
        self.assertRaises(ValueError, V.transform, d)
        d["extra"] = 10  # Extra key
        self.assertRaises(ValueError, V.transform, d)
        d["some integer"] = u"a string"  # Changed type
        self.assertRaises(ValueError, V.transform, d)
        d[u"j€nµmeæcħeid"] = "coca"  # Not unicode
        self.assertRaises(ValueError, V.transform, d)
