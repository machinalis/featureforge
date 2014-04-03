from future.builtins import str
import mock
import types
from unittest import TestCase

from featureforge.evaluator import FeatureEvaluator, TolerantFeatureEvaluator
from featureforge.feature import make_feature, input_schema, output_schema


@make_feature
@input_schema(dict)
@output_schema(str)
def DumbFeatureA(data_point):
    return u'a'


@make_feature
@input_schema(dict)
@output_schema(object)
def EntireSampleFeature(data_point):
    return data_point


@make_feature
@input_schema({'description': str})
@output_schema(str)
def DescriptionFeature(data_point):
    return data_point['description']


@make_feature
@input_schema({'age': int})
@output_schema(str)
def AgeFeature(data_point):
    return data_point['age']


@make_feature
@input_schema({'description': str})
@output_schema(str)
def BrokenFeature(data_point):
    raise RuntimeError()


SAMPLES = [
    {'pk': 1, 'description': u'nice'},
    {'pk': 2, 'description': u'awesome moment with friends'},
    {'pk': 3, 'description': u'', 'other_field': u'something'},
    {'pk': 4, 'description': u'this is a long description text, deal with it'},
    {'pk': 5, 'description': u'everything is great'}
]


class SimpleEvaluatorTests(TestCase):

    def test_fit_creates_alive_features_tuple(self):
        ev = FeatureEvaluator([DumbFeatureA])
        self.assertFalse(hasattr(ev, 'alive_features'))
        ev.fit([])
        self.assertTrue(hasattr(ev, 'alive_features'))

    def test_returns_generator(self):
        ev = FeatureEvaluator([DumbFeatureA])
        ev.fit([])
        Xt = ev.transform([])
        self.assertIsInstance(Xt, types.GeneratorType)

    def test_returns_tuples_of_features_length(self):
        features = [DumbFeatureA, DumbFeatureA]
        ev = FeatureEvaluator(features)
        ev.fit(SAMPLES)
        Xt = ev.transform(SAMPLES)
        x = next(Xt)
        self.assertIsInstance(x, tuple)
        self.assertEqual(len(x), len(features))

    def test_returns_as_many_tuples_as_samples(self):
        ev = FeatureEvaluator([DumbFeatureA])
        ev.fit(SAMPLES)
        Xt = ev.transform(SAMPLES)
        self.assertEqual(len(list(Xt)), len(SAMPLES))

    def test_fit_transform_does_both_things(self):
        ev = FeatureEvaluator([DumbFeatureA])
        Xt_1 = ev.fit_transform(SAMPLES)
        Xt_2 = ev.transform(SAMPLES)
        self.assertListEqual(list(Xt_1), list(Xt_2))


class TolerantFittingCases(object):
    fit_method_name = ''

    def apply_fit(self, *args, **kwargs):
        method = getattr(self.ev, self.fit_method_name)
        return method(*args, **kwargs)

    def test_once_fitted_says_fitted(self):
        self.ev = TolerantFeatureEvaluator([DumbFeatureA])
        self.assertFalse(self.ev.fitted)
        self.apply_fit([])
        self.assertTrue(self.ev.fitted)

    def test_fit_creates_alive_features_tuple(self):
        self.ev = TolerantFeatureEvaluator([DumbFeatureA])
        self.assertFalse(hasattr(self.ev, 'alive_features'))
        self.apply_fit([])
        self.assertTrue(hasattr(self.ev, 'alive_features'))

    def test_feature_is_excluded_if_fails_on_firts_M_samples(self):
        # The description-feature needs "description" on data-point
        actual_feature = DescriptionFeature
        description_feature = mock.Mock(wraps=actual_feature,
                                        spec=actual_feature)
        self.ev = TolerantFeatureEvaluator([description_feature, DumbFeatureA])
        # We'll use 2 as the first M of test title
        self.ev.FEATURE_STRICT_UNTIL = 2
        samples = SAMPLES[:]
        samples.insert(0, {'pk': 33})
        self.apply_fit(samples)
        # Caption was excluded from features list
        self.assertNotIn(description_feature, self.ev.alive_features)
        # Feature was not called anymore after failing, which occurred with the
        # first sample
        self.assertEqual(description_feature.call_count, 1)

    def test_feature_is_excluded_after_K_fails_no_matter_when(self):
        actual_feature = BrokenFeature
        broken_feature = mock.Mock(wraps=actual_feature, spec=actual_feature)
        self.ev = TolerantFeatureEvaluator([broken_feature, DumbFeatureA])

        # We'll make sure strict mode is turned off
        self.ev.FEATURE_STRICT_UNTIL = 0
        # now make sure that a feature can fail up to 2 times (K on test name)
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 2
        self.apply_fit(SAMPLES[:])
        # Feature was excluded from features list
        self.assertNotIn(broken_feature, self.ev.alive_features)
        # Feature was not called anymore after failing K+1 times
        self.assertEqual(broken_feature.call_count,
                         self.ev.FEATURE_MAX_ERRORS_ALLOWED + 1)

    def test_if_no_more_features_then_blows_up(self):
        self.ev = TolerantFeatureEvaluator([BrokenFeature])
        self.ev.FEATURE_STRICT_UNTIL = 2
        with self.assertRaises(self.ev.NoFeaturesLeftError):
            self.apply_fit(SAMPLES[:])

    def test_alive_features_can_hide_bad_samples(self):
        # While a Feature is alive, each time he failed, the sample causing the
        # failure is hidden to the rest of the features, in an attempt to
        # minimize damage of rare bad samples.
        self.ev = TolerantFeatureEvaluator([DescriptionFeature, AgeFeature,
                                            DumbFeatureA])
        self.ev.FEATURE_STRICT_UNTIL = 0  # No strict mode
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 1  # No feature failure tolerated
        no_age = {'description': u'i have'}
        bad_sample = {'nothing': u'this sample has no description, not age'}
        samples = [bad_sample, no_age]
        # Since the 2nd sample will succeed when evaluated on
        # DescriptionFeature but fail when evaluated with AgeFeature, we want
        # to check if at the end the AgeFeature is alive or not. If not, it's
        # because it suffered 2 failures, so the bad_sample wasn't hidden.
        self.apply_fit(samples)
        # Checking the preconditions of our test
        assert DescriptionFeature in self.ev.alive_features
        self.assertIn(AgeFeature, self.ev.alive_features)

    def test_when_feature_is_excluded_discarded_sample_is_reconsidered(self):
        self.ev = TolerantFeatureEvaluator([DescriptionFeature, AgeFeature,
                                            DumbFeatureA])
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        bad_sample = {'nothing': u'this sample has no description, not age'}
        samples = [bad_sample]
        self.apply_fit(samples)
        self.assertEqual(self.ev.alive_features, (DumbFeatureA, ))


class TolerantEvaluatorFitTests(TestCase, TolerantFittingCases):
    fit_method_name = 'fit'

    def test_returns_itself(self):
        self.ev = TolerantFeatureEvaluator([DumbFeatureA])
        self.assertEqual(self.ev.fit([]), self.ev)


class TolerantEvaluatorFitTransformTests(TestCase, TolerantFittingCases):
    # We try to mimic most of the cases seen on Fit tests, and check that
    # they are working equivalently with fit_transform
    fit_method_name = 'fit_transform'

    def test_sample_is_excluded_if_any_feature_fails_when_evaluating_it(self):
        self.ev = TolerantFeatureEvaluator([DescriptionFeature,
                                            EntireSampleFeature])
        self.ev.FEATURE_STRICT_UNTIL = 0
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = len(SAMPLES) + 1  # dont exclude
        samples = SAMPLES[:]
        nodescription = {'nodescription': u'this sample has no description'}
        samples.append(nodescription)
        result = self.ev.fit_transform(samples)
        self.assertTrue(len(list(result)) < len(samples))
        # EntireSampleFeature is the last, so is the last value per tuple
        self.assertNotIn(nodescription, [r[-1] for r in result])

    def test_if_a_feature_is_excluded_all_results_doesnt_include_it(self):
        # This means: if a Feature evaluated fine for some samples until it was
        # excluded, once we decided to exclude it, we must make sure that
        # previous samples for which this feature was evaluated, are now
        # striped out of those evaluations
        self.ev = TolerantFeatureEvaluator([DescriptionFeature, DumbFeatureA])
        self.ev.FEATURE_STRICT_UNTIL = 0
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        result = self.ev.fit_transform(SAMPLES + [{'nodescription': u'tada!'}])
        # Check that there are results. Otherwise, next loop is dumb
        self.assertTrue(result)
        for r in result:
            self.assertEqual(len(r), 1)  # only one value per sample
            self.assertEqual(r[0], 'a')  # Remember DumbFeatureA returns 'a'

    def test_when_feature_is_excluded_discarded_samples_are_reevaluated(self):
        self.ev = TolerantFeatureEvaluator([DescriptionFeature, DumbFeatureA,
                                            EntireSampleFeature])
        self.ev.FEATURE_MAX_ERRORS_ALLOWED = 0  # No feature failure tolerated
        samples = SAMPLES[:]
        nodescription = {'nodescription': u'this sample has no description'}
        samples.append(nodescription)
        result = list(self.ev.fit_transform(samples))
        self.assertEqual(len(samples), len(result))
        # EntireSampleFeature is the last, so is the last value per tuple
        self.assertIn(nodescription, [r[-1] for r in result])

    def test_consumable_is_consumed_only_once(self):
        samples = (s for s in SAMPLES)  # can be consumed once only
        self.ev = TolerantFeatureEvaluator([EntireSampleFeature])
        result = list(self.ev.fit_transform(samples))
        self.assertEqual(len(SAMPLES), len(result))


class TolerantEvaluatorTransformTests(TestCase):

    def test_errors_are_raised(self):
        self.ev = TolerantFeatureEvaluator([BrokenFeature, DumbFeatureA])
        self.ev.fit([])

        def transform():
            list(self.ev.transform(SAMPLES))  # force generation
        self.assertRaises(RuntimeError, transform)
