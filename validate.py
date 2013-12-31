from feature_bench import generate
import schema

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

    def assert_feature_passes_fixture(self, feature_spec, fixture):
        failures = []
        for label, (data_point, predicate, value) in fixture.items():
            if not _PREDICATES[predicate](feature_spec, data_point, value):
                msg = '%s failed, %s %s %s' % (
                    label, feature_spec(data_point),
                    _EXPLAIN_PREDICATE_FAIL[predicate],
                    value)
                failures.append(msg)
        self.assertFalse(failures, msg='; '.join(failures))

    def assert_passes_fuzz(self, feature_spec, tries=1000):
        for i in xrange(tries):
            data_point = generate.generate(feature_spec.input_schema)
            try:
                feature = feature_spec(data_point)
            except Exception as e:
                self.fail("Error evaluating; input=%r error=%r" % (data_point, e))
            try:
                feature_spec.output_schema.validate(feature)
            except schema.SchemaError:
                self.fail("Invalid output schema; input=%r output=%r" %
                          (data_point, feature))


class BaseFeatureFixture(FeatureFixtureCheckMixin):
    feature = None  # Needs to be defined on subclasses

    def test_fixtures(self):
        self.assert_feature_passes_fixture(self.feature(), self.fixtures)

    def test_fuzz(self):
        self.assert_passes_fuzz(self.feature())

    def test_has_valid_default(self):
        f = self.feature()
        self.assertTrue(hasattr(f, 'default'))
        try:
            f.output_schema.validate(f.default)
        except schema.SchemaError:
            self.fail("Invalid default value according output schema")

### EXAMPLE ###

if __name__ == "__main__":
    from feature_bench.feature import Feature
    from schema import Schema, And
    import unittest

    class Length(Feature):
        input_schema = Schema(str)
        output_schema = And(int, lambda n: n >= 0)

        def _evaluate(self, data_point):
            return len(data_point)

    class TestLength(unittest.TestCase, FeatureFixtureCheckMixin):

        def test_f(self):
            fixture = dict(
                test_eq=('hello', EQ, 5),
                test_approx=('world!', APPROX, 6.00001),
                test_in=('hello', IN, (5, 6, 1)),
                test_raise=(None, RAISES, ValueError),
            )
            self.assert_feature_passes_fixture(Length(), fixture)

        def test_fuzz(self):
            self.assert_passes_fuzz(Length())

    unittest.main()
