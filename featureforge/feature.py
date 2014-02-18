import functools

import schema


def soft_schema(**kwargs):
    """
        soft_schema(k1=schema1, k2=schema2, ...)

    Returns a schema for dicts having the keys k1, k2, ... and possibly other
    string keys not explicitly stated. The schema for the values is the one
    provided (i.e. schema for k2 is Schema(schema2)). Other keys have
    Schema(object).

    If one of the inner schemas given is a dict, it's interpreted as a soft
    dictionary schema too.
    """

    def _transform(d):
        result = d.copy()
        for k, v in result.items():
            if isinstance(v, dict):
                result[k] = _transform(v)
        result[schema.Optional(str)] = object
        return result

    return schema.Schema(_transform(kwargs))


def has_nones(data, data_schema):
    """
    Returns True iff data has any none in the places where the schema
    requires something else.

    Assumptions:
     * not schema.validate(data)
     * keys in dictionaries are just keys, not schemas

    """
    if data is None:
        return True
    while isinstance(data_schema, schema.Schema):
        data_schema = data_schema._schema
    if isinstance(data_schema, dict):
        for k in data_schema:
            if isinstance(k, schema.Optional):
                # Ignore optional values
                continue
            elif k in data:
                try:
                    v = schema.Schema(data_schema[k]).validate(data[k])
                except schema.SchemaError:
                    if has_nones(data[k], data_schema[k]):
                        return True
            else:
                # Schema failed because of missing keys, not because of a None
                return False
    elif isinstance(data_schema, (list, tuple)):
        or_schema = schema.Or(*data_schema)
        for v in data:
            try:
                v = or_schema.validate(v)
            except schema.SchemaError:
                if has_nones(v, or_schema):
                    return True
    elif isinstance(data_schema, schema.And):
        return has_nones(data, data_schema._args[0])
    return False


class Feature(object):

    input_schema = schema.Schema(object)
    output_schema = schema.Schema(object)

    class InputValueError(ValueError):
        pass

    class OutputValueError(ValueError):
        pass

    @property
    def name(self):
        return getattr(self, "_name", type(self).__name__)

    def __call__(self, data_point):
        try:
            data_point = self.input_schema.validate(data_point)
        except schema.SchemaError as e:
            if (hasattr(self, 'default') and
                has_nones(data_point, self.input_schema)):
                return self.default
            raise self.InputValueError(e)
        result = self._evaluate(data_point)
        try:
            return self.output_schema.validate(result)
        except schema.SchemaError as e:
            raise self.OutputValueError(e)

    def _evaluate(self, data_point):
        return None


# Extensions for schema of other objects
class ObjectSchema(schema.Schema):

    def __init__(self, **kwargs):
        self.attrs = kwargs

    def __repr__(self):
        attributes = ("%s=%s" % (n, repr(s)) for (n, s) in self.attrs.items())
        return '%s(%s)' % (type(self).__name__, ', '.join(attributes))

    def validate(self, data):
        for a, s in self.attrs.items():
            s = schema.Schema(s)
            try:
                value = getattr(data, a)
            except AttributeError:
                raise schema.SchemaError(" Missing attribute %r" % a, [])
            try:
                new_value = s.validate(value)
            except schema.SchemaError as e:
                raise schema.SchemaError(
                    "Invalid value for attribute %r: %s" % (a, e), [])
            setattr(data, a, new_value)
        return data


# Simple API
def make_feature(f):
    """
    Given a function f: data point -> feature that computes a feature, upgrade
    it to a feature instance.
    """
    if not callable(f):
        raise TypeError("f must be callable")
    result = Feature()
    result._evaluate = f
    result._name = getattr(f, "_feature_name", f.__name__)
    input_schema = getattr(f, "_input_schema", None)
    output_schema = getattr(f, "_output_schema", None)
    if input_schema is not None:
        result.input_schema = input_schema
    if output_schema is not None:
        result.output_schema = output_schema
    return result


def _build_schema(*args, **kwargs):
    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, dict):
            args[i] = soft_schema(**a)
    if kwargs:
        for k, a in kwargs.items():
            if isinstance(a, dict):
                args[k] = soft_schema(**a)
        attributes = [ObjectSchema(**kwargs)]
    else:
        attributes = []
    return schema.Schema(schema.And(*(args + attributes)))


def input_schema(*args, **kwargs):
    def decorate(f):
        f._input_schema = _build_schema(*args, **kwargs)
        return f
    return decorate


def output_schema(*args, **kwargs):
    def decorate(f):
        f._output_schema = _build_schema(*args, **kwargs)
        return f
    return decorate


def feature_name(name):
    def decorate(f):
        f._feature_name = name
        return f
    return decorate
