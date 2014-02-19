from unittest import TestCase

import mock

from featureforge import vectorizer
from featureforge.feature import Feature

class TestVectorizer(TestCase):

    def test_functions_are_converted(self):
        def sample_feature(x):
            pass
        with mock.patch('featureforge.evaluator.FeatureEvaluator.__init__') as mock_e:
            mock_e.return_value = None
            v = vectorizer.Vectorizer([sample_feature])
        self.assertEqual(len(mock_e.call_args_list), 1)
        arg = mock_e.call_args_list[0][0][0]
        print arg
        self.assertEqual(len(arg), 1)
        self.assertIsInstance(arg[0], Feature)
        self.assertEqual(arg[0].name, "sample_feature")

