import mock
from unittest import TestCase
from feature_bench import evaluator  # Imported like this to help mocking
from feature_bench import cache
from schema import And

from feature_bench.feature import Feature, soft_schema


class DumbFeatureA(Feature):
    # Never Fails. It accepts any data_point as valid, and always return u'a'
    input_schema = soft_schema()
    output_schema = And(unicode)
    default = u'unknown'

    def _evaluate(self, data_point):
        return u'a'


class EntireSampleFeature(Feature):
    # Artifact for allowing distinguishing is a sample was excluded
    input_schema = soft_schema()
    output_schema = And(object)
    default = {}

    def _evaluate(self, data_point):
        return data_point


class CaptionFeature(Feature):
    input_schema = soft_schema(caption=unicode)
    output_schema = And(unicode)
    default = u'unknown'

    def _evaluate(self, data_point):
        return data_point['caption']


class BrokenFeature(CaptionFeature):
    def _evaluate(self, data_point):
        raise RuntimeError()


SAMPLES = [
    {'pk': 1, 'caption': u'nice'},
    {'pk': 2, 'caption': u'awesome moment with friends'},
    {'pk': 3, 'caption': u'', 'other_field': u'something'},
    {'pk': 4, 'caption': u'this is a long the caption'},
    {'pk': 5, 'caption': u'everything is great, but we have no pk'}
]


class ActualEvaluatorFailuresToleranceTests(TestCase):
    def setUp(self):
        features = [CaptionFeature(), DumbFeatureA()]
        self.ev = evaluator.ActualEvaluator(features[:])

    def test_feature_is_excluded_if_fails_on_firts_M_samples(self):
        # We'll use 2 as M
        self.ev.FEATURE_STRICT_UNTIL = 2
        # The caption-feature needs "caption" on data-point
        actual_feature = CaptionFeature()
        caption_feature = mock.Mock(wraps=actual_feature,
                                    spec=actual_feature)
        self.ev.features = [caption_feature, DumbFeatureA()]
        samples = SAMPLES[:]
        samples.insert(0, {'pk': 33})
        _, failures = self.ev.transform(samples)
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
        # and now make sure that a feature can fail up to 2 times (K on test name)
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 2
        actual_feature = BrokenFeature()
        broken_feature = mock.Mock(wraps=actual_feature, spec=actual_feature)
        self.ev.features = [broken_feature, DumbFeatureA()]
        _, failures = self.ev.transform(SAMPLES[:])
        # Feature was excluded from features list
        self.assertNotIn(broken_feature, self.ev.features)
        # Feature was included on the exclusion list
        self.assertIn(broken_feature, failures['excluded_features'])
        # Feature was not called anymore after failing K+1 times
        self.assertEqual(broken_feature.call_count,
                         self.ev.FEATURE_MAX_ERRORS_ALLOWED + 1)

    def test_if_no_more_features_then_blows_up(self):
        self.ev.FEATURE_STRICT_UNTIL = 2
        self.ev.features = [BrokenFeature()]
        self.assertRaises(self.ev.NoFeaturesLeftError, self.ev.transform, SAMPLES[:])

    def test_sample_is_excluded_if_any_feature_fails_when_evaluating_it(self):
        self.ev.FEATURE_STRICT_UNTIL = 0
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = len(SAMPLES) + 1  # dont exclude
        self.ev.features = [CaptionFeature(), EntireSampleFeature()]
        samples = SAMPLES[:]
        nocaption = {'nocaption': u'this sample has no caption'}
        samples.append(nocaption)
        result, _ = self.ev.transform(samples)
        self.assertTrue(len(result) < len(samples))
        self.assertNotIn(nocaption, [r['EntireSampleFeature'] for r in result])

    def test_excluded_samples_are_reported_on_stats(self):
        self.ev.FEATURE_STRICT_UNTIL = 0
        self.ev.features = [CaptionFeature()]
        result, stats = self.ev.transform([{'pk': 123}])
        self.assertIn(123, stats['discarded_samples'])

    def test_if_a_feature_is_excluded_all_results_doesnt_include_it(self):
        # This means: if a Feature evaluated fine for some samples until it was
        # excluded, once we decided to exclude it, we must make sure that previous samples
        # for which this feature was evaluated, are now striped out of those evaluations
        self.ev.FEATURE_STRICT_UNTIL = 0
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        self.ev.features = [CaptionFeature(), DumbFeatureA()]
        result, _ = self.ev.transform(SAMPLES + [{'nocaption': u'tada!'}])
        self.assertTrue(result)  # Ie, there are results. Otherwise, next assert is dumb
        for r in result:
            self.assertEqual(r.keys(), ['DumbFeatureA'])

    def test_when_a_feature_is_excluded_a_discarded_sample_is_re_considered(self):
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        self.ev.features = [CaptionFeature(), DumbFeatureA(), EntireSampleFeature()]
        samples = SAMPLES[:]
        nocaption = {'nocaption': u'this sample has no caption'}
        samples.append(nocaption)
        result, _ = self.ev.transform(samples)
        self.assertEqual(len(samples), len(result))
        self.assertIn(nocaption, [r['EntireSampleFeature'] for r in result])

    def test_if_not_on_train_mode_errors_are_raised(self):
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        self.ev.features = [BrokenFeature(), DumbFeatureA()]
        self.assertRaises(RuntimeError, self.ev.transform,
                          SAMPLES[:], train_mode=False)


class FeatureEvaluatorTests(TestCase):

    def test_when_transforming_instantiates_ActualEvaluator_with_same_features(self):
        features = [DumbFeatureA(), EntireSampleFeature()]
        ev = evaluator.FeatureEvaluator(features)
        actual_mock = mock.MagicMock()
        actual_mock.transform.return_value = [], {}
        with mock.patch('feature_bench.evaluator.ActualEvaluator') as actual_new:
            actual_new.return_value = actual_mock
            ev.transform(SAMPLES[:])
            actual_new.assert_called_once_with(features)

    def test_when_transforming_run_ActualEvaluator(self):
        ev = evaluator.FeatureEvaluator([])
        with mock.patch.object(evaluator.ActualEvaluator,
                               'transform') as actual_transform_mock:
            actual_transform_mock.return_value = [], {}
            ev.transform(SAMPLES[:])
            actual_transform_mock.assert_called_once_with(SAMPLES, train_mode=True)

    def test_second_and_following_calls_for_same_Evaluator_arent_train(self):
        ev = evaluator.FeatureEvaluator([])
        with mock.patch.object(evaluator.ActualEvaluator,
                               'transform') as actual_transform_mock:
            actual_transform_mock.return_value = [], {'discarded_samples': [],
                                                      'excluded_features': []}
            ev.transform(SAMPLES[:])
            ev.transform(SAMPLES[:] + [{}])  # becareful with avoid lru caching
            ev.transform(SAMPLES[:] + [{}] + [{}])  # becareful with avoid lru caching
            self.assertEqual(actual_transform_mock.call_count, 3)
            for call in actual_transform_mock.call_args_list[1:]:
                self.assertEqual(call[1], {'train_mode': False})

    def test_if_on_training_a_feature_failed_on_prediction_that_feature_is_excluded(self):
        features = [DumbFeatureA(), BrokenFeature()]
        ev = evaluator.FeatureEvaluator(features)
        ev.transform(SAMPLES[:])
        # Let's see that when transforming again from same instance of FeatureEvaluator,
        # the BrokenFeature is not passed
        actual_mock = mock.MagicMock()
        actual_mock.transform.return_value = [], {}
        with mock.patch('feature_bench.evaluator.ActualEvaluator') as actual_new:
            actual_new.return_value = actual_mock
            ev.transform(SAMPLES[:])
            actual_new.assert_called_once_with(features[:-1])


class TestEvaluatorCaches(TestCase):
    # Testing the behavior of lru_cache and on Evaluation transform

    def tearDown(self):
        reload(cache)
        reload(evaluator)

    def test_RAM_cache__same_call_doesnt_execute_ActualEvaluation(self):
        # also known as RAM cache

        from feature_bench import settings as cache_settings
        with mock.patch.object(cache_settings, 'ENABLED_CACHES', ['ram']):
            reload(cache)
            reload(evaluator)
            ev = evaluator.FeatureEvaluator([])
            ev2 = evaluator.FeatureEvaluator([])
            ev3 = evaluator.FeatureEvaluator([])
            with mock.patch.object(evaluator.ActualEvaluator,
                                   'transform') as actual_transform_mock:
                actual_transform_mock.return_value = [], mock.MagicMock()
                samples_list = SAMPLES[:]  # it depents on the python id of the samples
                ev.transform(samples_list)
                self.assertEqual(actual_transform_mock.call_count, 1)
                # Now transforming again with same samples on same evaluator, produces no
                # call
                ev2.transform(samples_list)
                self.assertEqual(actual_transform_mock.call_count, 1)
                # Transforming again with pseudo same samples, do produce another call.
                # And I said pseudo, cause it's a different python list
                ev3.transform(list(samples_list))
                self.assertEqual(actual_transform_mock.call_count, 2)

    def test_RAM_cache_can_be_disabled(self):
        from feature_bench import settings as cache_settings
        enabled_caches = []
        with mock.patch.object(cache_settings, 'ENABLED_CACHES', enabled_caches):
            reload(cache)
            reload(evaluator)
            ev = evaluator.FeatureEvaluator([])
            ev2 = evaluator.FeatureEvaluator([])
            with mock.patch.object(evaluator.ActualEvaluator,
                                   'transform') as actual_transform_mock:
                actual_transform_mock.return_value = [], mock.MagicMock()
                samples_list = SAMPLES[:]  # it depents on the python id of the samples
                ev.transform(samples_list)
                self.assertEqual(actual_transform_mock.call_count, 1)
                ev2.transform(samples_list)
                # Although both calls were made with same arguments, cache was disabled.
                self.assertEqual(actual_transform_mock.call_count, 2)

