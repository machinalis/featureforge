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
            vectorizer.Vectorizer([sample_feature])
        self.assertEqual(len(mock_e.call_args_list), 1)
        arg = mock_e.call_args_list[0][0][0]
        self.assertEqual(len(arg), 1)
        self.assertIsInstance(arg[0], Feature)
        self.assertEqual(arg[0].name, "sample_feature")

    def test_tolerant_argument_is_used(self):
        feature = lambda x: 1
        with mock.patch('featureforge.vectorizer.TolerantFeatureEvaluator') as TFE:
            with mock.patch('featureforge.vectorizer.FeatureEvaluator') as FE:
                vectorizer.Vectorizer([feature], tolerant=False)
                self.assertFalse(TFE.called)
                self.assertTrue(FE.called)
                FE.reset_mock()
                vectorizer.Vectorizer([feature], tolerant=True)
                self.assertFalse(FE.called)
                self.assertTrue(TFE.called)

    def test_sparse_argument_is_used(self):
        feature = lambda x: 1
        with mock.patch('featureforge.vectorizer.FeatureMappingFlattener') as FMF:
            vectorizer.Vectorizer([feature], sparse=False)
            FMF.assert_called_once_with(sparse=False)
            FMF.reset_mock()
            vectorizer.Vectorizer([feature], sparse=True)
            FMF.assert_called_once_with(sparse=True)
