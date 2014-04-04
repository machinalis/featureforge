from future.builtins import range
import schema

from featureforge import generate
from featureforge.feature import make_feature


EQ = 'EQ'
APPROX = 'APPROX'
IN = 'IN'
RAISES = 'RAISES'

EPSILON = 0.01


def _raise_predicate(spec, data, exception):
    try:
        spec(data)
    except exception:
        return True
    return False

_PREDICATES = {
    EQ: lambda f, d, v: f(d) == v,
    APPROX: lambda f, d, v: abs(f(d) - v) < EPSILON,
    IN: lambda f, d, v: f(d) in v,
    RAISES: _raise_predicate
}

_EXPLAIN_PREDICATE_FAIL = {
    EQ: 'is not equal',
    APPROX: 'is not approx',
    IN: 'is not in',
    RAISES: 'not raised'
}


class FeatureFixtureCheckMixin(object):
    """
    This class is a TestCase mixin that provides some assertions to test
    features.

    In most cases, you shouldn't use this directly but BaseFeatureFixture
    instead
    """

    def assert_feature_passes_fixture(self, feature_spec, fixture):
        """
        Check that the given feature (function or Feature instance) passes
        all the conditions given in the fixture

        `fixture` is a dictionary where each key/value pair describes a simple
        example for the feature. The key should be a string (which will be
        reported in case of failure, so you know which case failed), and the
        value is a tuple (input, predicate, value). The `input` is the value
        that will be passed as argument as a feature. The predicate and the
        value give the condition, and should be one of the following:

         * (input, EQ, value) checks that feature(input) == value
         * (input, APPROX, value) checks that feature(input) == value
           approximately the error allowed is given by the constant EPSILON in
           this module
         * (input, IN, values) checks that feature(input) in values
         * (input, RAISES, eclass) checks that feature(input) raises an
           exception of eclass type. Note that input/output validation always
           raise an exception that subclasses ValueError

        """
        failures = []
        feature_spec = make_feature(feature_spec)
        for label, (data_point, predicate, value) in fixture.items():
            if not _PREDICATES[predicate](feature_spec, data_point, value):
                msg = '%s failed, %s %s %s' % (
                    label, feature_spec(data_point),
                    _EXPLAIN_PREDICATE_FAIL[predicate],
                    value)
                failures.append(msg)
        self.assertFalse(failures, msg='; '.join(failures))

    def assert_passes_fuzz(self, feature_spec, tries=1000):
        """
        Generates tries data points for the feature (which should have an
        input schema which allows generation) randomly, and applies those
        to the feature. It checks that the evaluation proceeds without raising
        exceptions and that it produces valid outputs according to the
        output schema.
        """
        feature_spec = make_feature(feature_spec)
        for i in range(tries):
            data_point = generate.generate(feature_spec.input_schema)
            try:
                feature = feature_spec(data_point)
            except Exception as e:
                self.fail("Error evaluating; input=%r error=%r" %
                          (data_point, e))
            try:
                feature_spec.output_schema.validate(feature)
            except schema.SchemaError:
                self.fail("Invalid output schema; input=%r output=%r" %
                          (data_point, feature))


class BaseFeatureFixture(FeatureFixtureCheckMixin):
    """
    Inheriting this class together with unittest.TestCase allows you to
    quickly build test cases for features. Your subclass should define two
    class attributes:

    `feature` should be a function or a Feature() instance

    `fixture` has a list of cases to test; check the documentation
    of `assert_feature_passes_fixture` for more details.

    The class defined by this will validate all features in the fixture. It
    will also subject the feature to fuzzy testing if the input schema allows
    it. It's also possible to add additional tests to the testcase.

    If you want to have more control about how the fixture is applied or skip
    fuzzy testing, take a look at the FeatureFixtureCheckMixin.
    """

    feature = None  # Needs to be defined on subclasses

    def test_fixtures(self):
        self.assert_feature_passes_fixture(self.feature, self.fixtures)

    def test_fuzz(self):
        self.assert_passes_fuzz(self.feature)


### EXAMPLE ###

if __name__ == "__main__":
    from featureforge.feature import input_schema, output_schema
    import unittest

    @input_schema(str)
    @output_schema(int, lambda n: n >= 0)
    def length(data_point):
        return len(data_point)

    # This is an example on how to use assertions directly
    class TestLength(unittest.TestCase, FeatureFixtureCheckMixin):

        def test_f(self):
            fixture = dict(
                test_eq=('hello', EQ, 5),
                test_approx=('world!', APPROX, 6.00001),
                test_in=('hello', IN, (5, 6, 1)),
                test_raise=(None, RAISES, ValueError),
            )
            self.assert_feature_passes_fixture(length, fixture)

        def test_fuzz(self):
            self.assert_passes_fuzz(length)

    class TestLength2(unittest.TestCase, BaseFeatureFixture):
        feature = length
        fixtures = dict(
                test_eq=('hello', EQ, 5),
                test_approx=('world!', APPROX, 6.00001),
                test_in=('hello', IN, (5, 6, 1)),
                test_raise=(None, RAISES, ValueError),
            )

    unittest.main()
