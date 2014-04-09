from unittest import TestCase

import schema

from featureforge.feature import (
    Feature, ObjectSchema, make_feature, input_schema, output_schema,
    feature_name
)


class TestFeatureBuilding(TestCase):

    def test_make_feature_basic(self):
        # Check that make_feature builds a usable feature from a bare
        # function

        # Build a simple feature that tracks its calls
        witness = []
        data = object()

        def simple_feature(data):
            witness.append(data)
            return 123
        f = make_feature(simple_feature)

        # Check that the feature was built reasonably:
        # * The feature has the right type
        self.assertIsInstance(f, Feature)
        # * The feature takes its name from the function
        self.assertEqual(f.name, "simple_feature")

        # Try the feature
        result = f(data)

        # Check the result
        # * The feature returned its value
        self.assertEqual(result, 123)
        # * The function was actually called
        self.assertEqual(witness, [data])

    def test_feature_renaming(self):
        @feature_name("new name")
        def simple_feature(data):
            return 123
        f = make_feature(simple_feature)

        # Feature was renamed
        self.assertEqual(f.name, "new name")
        # Feature still works
        self.assertEqual(f(None), 123)

    def test_input_schema(self):
        @input_schema(str)
        def length(s):
            return len(s)
        f = make_feature(length)

        # input_schema does NOT modify the original function behavior:
        self.assertEqual(length([1, 2, 3]), 3)

        # for the feature, behavior is preserved when respecting the schema
        self.assertEqual(f("wxyz"), 4)

        # But if schema is violated, an exception is raised
        with self.assertRaises(f.InputValueError):
            f([1, 2, 3])

    def test_output_schema(self):
        @output_schema(str)
        def identity(x):
            return x
        f = make_feature(identity)
        # input_schema does NOT modify the original function behavior:
        self.assertEqual(identity(0), 0)

        # for the feature, behavior is preserved when respecting the schema
        self.assertEqual(f("x"), "x")

        # But if schema is violated, an exception is raised
        with self.assertRaises(f.OutputValueError):
            f(0)

    def test_complex_schema(self):
        # A data point class
        class DataPoint(object):
            pass

        # A feature with a complex schema
        @input_schema(DataPoint, lambda d: d.a == d.b, a=int, b=float, c=str)
        def identity(x):
            return x
        f = make_feature(identity)

        # This accepts a valid data point and calls the feature
        valid = DataPoint()
        valid.a, valid.b, valid.c = 2, 2.0, ""
        self.assertIs(f(valid), valid)

        # All arguments in the call are ANDed together, so if one fails,
        # the generated schema fails
        invalid = DataPoint()
        invalid.a, invalid.b, invalid.c = 2, 3.14, ""
        with self.assertRaises(f.InputValueError):
            f(invalid)

        # Keyword arguments detect type mismatches in the fields
        invalid = DataPoint()
        invalid.a, invalid.b, invalid.c = 2.0, 2.0, ""
        with self.assertRaises(f.InputValueError):
            f(invalid)

        # Keyword arguments detect missing fields
        invalid = DataPoint()
        invalid.a, invalid.b = 2, 2.0
        with self.assertRaises(f.InputValueError):
            f(invalid)

    def test_soft_schema(self):
        @input_schema({'key': int})
        def key(d):
            return d['key']
        f = make_feature(key)
        # Dictionary schemas work more or less normally...
        self.assertEqual(f({'key': 42}), 42)
        # except that they aren't strict with extra keys:
        self.assertEqual(f({'key': 42, 'another': 37}), 42)


class TestObjectSchema(TestCase):

    def setUp(self):
        class Data():
            pass
        self.data = Data()
        self.data.a = "hello"
        self.data.b = 99

    def test_simple(self):
        s = ObjectSchema(a="hello", b=int)
        output = s.validate(self.data)
        # validation preserves identity
        self.assertIs(output, self.data)
        # this schema does not change objects:
        self.assertEqual(output.a, "hello")
        self.assertEqual(output.b, 99)

        # It tolerates additional fields
        self.data.c = "another field"
        output = s.validate(self.data)
        self.assertIs(output, self.data)

        # It detects field schema mismatches
        self.data.a = "not hello"
        with self.assertRaises(schema.SchemaError):
            s.validate(self.data)

        # And it detects missing fields
        del self.data.a
        with self.assertRaises(schema.SchemaError):
            s.validate(self.data)

    def test_repr(self):
        s = ObjectSchema(a="hello", b=int)
        self.assertEqual(repr(s), "ObjectSchema(a='hello', b=%s)" % int)

    def test_inner_schema(self):
        s = ObjectSchema(
            b=schema.And(int, lambda x: x > 98, lambda x: x < 100)
        )
        output = s.validate(self.data)
        # validation preserves identity
        self.assertIs(output, self.data)
        # this schema does not change objects:
        self.assertEqual(output.a, "hello")
        self.assertEqual(output.b, 99)
