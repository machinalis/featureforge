import mock
import types
from unittest import TestCase

from featureforge import evaluator  # Imported like this to help mocking
from featureforge.feature import make_feature, input_schema, output_schema


@make_feature
@input_schema(dict)
@output_schema(unicode)
def DumbFeatureA(data_point):
    return u'a'


@make_feature
@input_schema(dict)
@output_schema(object)
def EntireSampleFeature(data_point):
    return data_point


@make_feature
@input_schema({'caption': unicode})
@output_schema(unicode)
def CaptionFeature(data_point):
    return data_point['caption']


@make_feature
@input_schema({'caption': unicode})
@output_schema(unicode)
def BrokenFeature(data_point):
    raise RuntimeError()


SAMPLES = [
    {'pk': 1, 'caption': u'nice'},
    {'pk': 2, 'caption': u'awesome moment with friends'},
    {'pk': 3, 'caption': u'', 'other_field': u'something'},
    {'pk': 4, 'caption': u'this is a long the caption'},
    {'pk': 5, 'caption': u'everything is great, but we have no pk'}
]


class SimpleEvaluatorTests(TestCase):

    def test_fit_creates_alive_features_tuple(self):
        ev = evaluator.FeatureEvaluator([DumbFeatureA])
        self.assertFalse(hasattr(ev, 'alive_features'))
        ev.fit([])
        self.assertTrue(hasattr(ev, 'alive_features'))

    def test_returns_generator(self):
        ev = evaluator.FeatureEvaluator([DumbFeatureA])
        ev.fit([])
        Xt = ev.transform([])
        self.assertIsInstance(Xt, types.GeneratorType)

    def test_returns_tuples_of_features_length(self):
        features = [DumbFeatureA, DumbFeatureA]
        ev = evaluator.FeatureEvaluator(features)
        ev.fit(SAMPLES)
        Xt = ev.transform(SAMPLES)
        x = Xt.next()
        self.assertIsInstance(x, tuple)
        self.assertEqual(len(x), len(features))

    def test_returns_as_many_tuples_as_samples(self):
        ev = evaluator.FeatureEvaluator([DumbFeatureA])
        ev.fit(SAMPLES)
        Xt = ev.transform(SAMPLES)
        self.assertEqual(len(list(Xt)), len(SAMPLES))

    def test_fit_transform_does_both_things(self):
        ev = evaluator.FeatureEvaluator([DumbFeatureA])
        Xt_1 = ev.fit_transform(SAMPLES)
        Xt_2 = ev.transform(SAMPLES)
        self.assertListEqual(list(Xt_1), list(Xt_2))


class ActualEvaluatorFailureToleranceTests(TestCase):
    def setUp(self):
        features = [CaptionFeature, DumbFeatureA]
        self.ev = evaluator.ActualEvaluator(features[:])

    def test_feature_is_excluded_if_fails_on_firts_M_samples(self):
        # We'll use 2 as M
        self.ev.FEATURE_STRICT_UNTIL = 2
        # The caption-feature needs "caption" on data-point
        actual_feature = CaptionFeature
        caption_feature = mock.Mock(wraps=actual_feature,
                                    spec=actual_feature)
        self.ev.features = [caption_feature, DumbFeatureA]
        samples = SAMPLES[:]
        samples.insert(0, {'pk': 33})
        self.ev.transform(samples)
        failures = self.ev.get_training_stats()
        # Caption was excluded from features list
        self.assertNotIn(caption_feature, self.ev.features)
        # Caption was included on the exclusion list
        self.assertIn(caption_feature, failures['excluded_features'])
        # Feature was not called anymore after failing, which occurred with the
        # first sample
        self.assertEqual(caption_feature.call_count, 1)

    def test_feature_is_excluded_after_K_fails_no_matter_when(self):
        # We'll make sure strict mode is turned off
        self.ev.FEATURE_STRICT_UNTIL = 0
        # now make sure that a feature can fail up to 2 times (K on test name)
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 2
        actual_feature = BrokenFeature
        broken_feature = mock.Mock(wraps=actual_feature, spec=actual_feature)
        self.ev.features = [broken_feature, DumbFeatureA]
        self.ev.transform(SAMPLES[:])
        failures = self.ev.get_training_stats()
        # Feature was excluded from features list
        self.assertNotIn(broken_feature, self.ev.features)
        # Feature was included on the exclusion list
        self.assertIn(broken_feature, failures['excluded_features'])
        # Feature was not called anymore after failing K+1 times
        self.assertEqual(broken_feature.call_count,
                         self.ev.FEATURE_MAX_ERRORS_ALLOWED + 1)

    def test_if_no_more_features_then_blows_up(self):
        self.ev.FEATURE_STRICT_UNTIL = 2
        self.ev.features = [BrokenFeature]
        with self.assertRaises(self.ev.NoFeaturesLeftError):
            self.ev.transform(SAMPLES[:])

    def test_sample_is_excluded_if_any_feature_fails_when_evaluating_it(self):
        self.ev.FEATURE_STRICT_UNTIL = 0
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = len(SAMPLES) + 1  # dont exclude
        self.ev.features = [CaptionFeature, EntireSampleFeature]
        samples = SAMPLES[:]
        nocaption = {'nocaption': u'this sample has no caption'}
        samples.append(nocaption)
        result = self.ev.transform(samples)
        self.assertTrue(len(list(result)) < len(samples))
        # EntireSampleFeature is the last, so is the last value per tuple
        self.assertNotIn(nocaption, [r[-1] for r in result])

    def test_excluded_samples_are_reported_on_stats(self):
        self.ev.FEATURE_STRICT_UNTIL = 0
        self.ev.features = [CaptionFeature]
        self.ev.transform([{'pk': 123}])
        self.assertIn(123, self.ev.get_training_stats()['discarded_samples'])

    def test_if_a_feature_is_excluded_all_results_doesnt_include_it(self):
        # This means: if a Feature evaluated fine for some samples until it was
        # excluded, once we decided to exclude it, we must make sure that
        # previous samples for which this feature was evaluated, are now
        # striped out of those evaluations
        self.ev.FEATURE_STRICT_UNTIL = 0
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        self.ev.features = [CaptionFeature, DumbFeatureA]
        result = self.ev.transform(SAMPLES + [{'nocaption': u'tada!'}])
        # Check that there are results. Otherwise, next loop is dumb
        self.assertTrue(result)
        for r in result:
            self.assertEqual(len(r), 1)  # only one value per sample
            self.assertEqual(r[0], 'a')  # Remember that DumbFeatureA returns 'a'

    def test_when_a_feature_is_excluded_a_discarded_sample_is_reconsidered(self):
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        self.ev.features = [CaptionFeature, DumbFeatureA, EntireSampleFeature]
        samples = SAMPLES[:]
        nocaption = {'nocaption': u'this sample has no caption'}
        samples.append(nocaption)
        result = list(self.ev.transform(samples))
        self.assertEqual(len(samples), len(result))
        # EntireSampleFeature is the last, so is the last value per tuple
        self.assertIn(nocaption, [r[-1] for r in result])

    def test_if_not_on_train_mode_errors_are_raised(self):
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        self.ev.features = [BrokenFeature, DumbFeatureA]
        self.assertRaises(RuntimeError, self.ev.transform,
                          SAMPLES[:], train_mode=False)


class TolerantFeatureEvaluatorTests(TestCase):

    def test_when_transforming_instantiates_ActualEvaluator_with_same_features(self):
        features = [DumbFeatureA, EntireSampleFeature]
        ev = evaluator.TolerantFeatureEvaluator(features)
        actual_mock = mock.MagicMock()
        actual_mock.transform.return_value = []
        with mock.patch('featureforge.evaluator.ActualEvaluator') as actual_new:
            actual_new.return_value = actual_mock
            ev.transform(SAMPLES[:])
            actual_new.assert_called_once_with(features)

    def test_when_transforming_run_ActualEvaluator(self):
        ev = evaluator.TolerantFeatureEvaluator([])
        with mock.patch.object(evaluator.ActualEvaluator,
                               'transform') as actual_transform_mock:
            actual_transform_mock.return_value = [], {}
            ev.transform(SAMPLES[:])
            actual_transform_mock.assert_called_once_with(SAMPLES, train_mode=True)

    def test_second_and_following_calls_for_same_Evaluator_arent_train(self):
        ev = evaluator.TolerantFeatureEvaluator([])
        with mock.patch.object(evaluator.ActualEvaluator,
                               'transform') as actual_transform_mock:
            actual_transform_mock.return_value = []
            ev.transform(SAMPLES[:])
            ev.transform(SAMPLES[:] + [{}])
            ev.transform(SAMPLES[:] + [{}] + [{}])
            self.assertEqual(actual_transform_mock.call_count, 3)
            for call in actual_transform_mock.call_args_list[1:]:
                self.assertEqual(call[1], {'train_mode': False})

    def test_if_on_training_a_feature_failed_on_prediction_that_feature_is_excluded(self):
        features = [DumbFeatureA, BrokenFeature]
        ev = evaluator.TolerantFeatureEvaluator(features)
        ev.transform(SAMPLES[:])
        # Let's see that when transforming again from same instance of
        # TolerantFeatureEvaluator, the BrokenFeature is not passed
        actual_mock = mock.MagicMock()
        actual_mock.transform.return_value = []
        with mock.patch('featureforge.evaluator.ActualEvaluator') as actual_new:
            actual_new.return_value = actual_mock
            ev.transform(SAMPLES[:])
            actual_new.assert_called_once_with(features[:-1])
