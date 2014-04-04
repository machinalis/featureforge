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


class Feature(object):
    """
    Feature instances represent a single feature. This class is never
    instantiated as is; you build an instance using make_feature on a
    decorated function, or you define a subclass and instantiate that.

    The instance has a callable interface, taking a data point and
    returning a feature value. The typical use case is overriding the
    `_evaluate` method and leaving the standard `__call__` in place
    (which wraps `_evaluate` adding input and output validation)

    Besides the `__call__` methods, the following class attributes are
    available:

     * `input_schema` is a `schema.Schema` object for validating
       inputs. This can be overriden in subclasses or as an instance
       attribute
     * `output_schema` is a `schema.Schema` object for validating
       output. This can be overriden in subclasses or as an instance
       attribute
     * `InputValueError` is an exception class raised when input fails
       to validate; it is a subclass of `ValueError`
     * `OutputValueError` is an exception class raised when output
       fails to validate; it is a subclass of `ValueError`
    """

    input_schema = schema.Schema(object)
    output_schema = schema.Schema(object)

    class InputValueError(ValueError):
        pass

    class OutputValueError(ValueError):
        pass

    @property
    def name(self):
        """A human readable name to be used in reports and error messages"""
        return getattr(self, "_name", type(self).__name__)

    def __call__(self, data_point):
        """Validate intput, evaluate, and validate result"""
        try:
            data_point = self.input_schema.validate(data_point)
        except schema.SchemaError as e:
            raise self.InputValueError(e)
        result = self._evaluate(data_point)
        try:
            return self.output_schema.validate(result)
        except schema.SchemaError as e:
            raise self.OutputValueError(e)

    def _evaluate(self, data_point):
        """Override this to provide your own evaluation function"""
        raise NotImplemented


# Extensions for schema of other objects
class ObjectSchema(schema.Schema):
    """
    ObjectSchema(attr1=schema1, attr2=schema2, ...) is a schema.Schema-like
    class providing validation for the fields of a python object. To validate,
    the object must have attr1, attr2, ... as attributes, and each attribute
    must validate with the corresponding schema.

    Note that schema1, schema2, ... do not need to be a schema, they can be any
    valid value useful for building is schema (i.e., you can use `int` instead
    of `schema.Schema(int)`

    """

    def __init__(self, **kwargs):
        self.attrs = kwargs

    def __repr__(self):
        attributes = ("%s=%s" % (n, repr(s)) for (n, s) in sorted(self.attrs.items()))
        return '%s(%s)' % (type(self).__name__, ', '.join(attributes))

    def validate(self, data):
        """
        Check that data has all the attributes specified, and validate each
        attribute with the schema provided on construction
        """
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
    Returns f if f is already a Feature instance
    """
    if not callable(f):
        raise TypeError("f must be callable")
    if isinstance(f, Feature):
        return f
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
    # Helper to build schema form the arguments of input_schema & output_schema
    args = list(args)
    for i, a in enumerate(args):
        # Dictionaries are "softened"
        if isinstance(a, dict):
            args[i] = soft_schema(**a)
    if kwargs:
        for k, a in kwargs.items():
            if isinstance(a, dict):
                args[k] = soft_schema(**a)
        # if there are kwargs, add an objectschema to the condition
        args.append(ObjectSchema(**kwargs))
    return schema.Schema(schema.And(*args))


def input_schema(*args, **kwargs):
    """
    @input_schema(s1, s2, ..., attr1=as1, attr2=as2, ...)
    def f(data_point): ...

    Annotate the schema for validating the data_point that f receives.

    s1, s2, ... are anything that allows building a schema, and are considered
    as conditions to "and" with each other (i.e., to validate, all s1, s2, ...
    must validate). If keyword arguments are specified, the keyword names must
    be attributes of the data points, and the value of each of those attributes
    is validated with the corresponding schema (i.e., as1 is used to validate
    data_point.attr1, etc.)

    If any of the schemas used is a dictionary, the semantics of that schema
    is modified to allow additional keys besides the ones specified explicitly
    """
    def decorate(f):
        f._input_schema = _build_schema(*args, **kwargs)
        return f
    return decorate


def output_schema(*args, **kwargs):
    """
    @output_schema(s1, s2, ..., attr1=as1, attr2=as2, ...)
    def f(data_point): ...

    Annotate the schema for validating the result of f.

    s1, s2, ... are anything that allows building a schema, and are considered
    as conditions to "and" with each other (i.e., to validate, all s1, s2, ...
    must validate). If keyword arguments are specified, the keyword names must
    be attributes of the result, and the value of each of those attributes
    is validated with the corresponding schema (i.e., as1 is used to validate
    f(data_point).attr1, etc.)

    If any of the schemas used is a dictionary, the semantics of that schema
    is modified to allow additional keys besides the ones specified explicitly
    """
    def decorate(f):
        f._output_schema = _build_schema(*args, **kwargs)
        return f
    return decorate


def feature_name(name):
    """
    @feature_name("name")
    def f(...): ...

    Annotate the name to be used when describing f as a feature. The default
    name when this decorator is used is the function name itself, but you can
    define any custom string here if you want to disambiguate or be more
    specific. The name provided will be used in some reports and error
    messages.
    """
    def decorate(f):
        f._feature_name = name
        return f
    return decorate
