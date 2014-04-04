from datetime import datetime
import random

from future.builtins import range, str
import schema
import string

MAX_LEN = 20


def generate_int():
    return random.randrange(-MAX_LEN, MAX_LEN)


def generate_str():
    l = random.randrange(MAX_LEN)
    return ''.join([random.choice(string.printable) for _ in range(l)])


def generate_float():
    return random.random()


def generate_bool():
    return random.random() > 0.5


def generate_datetime():
    rand_seconds = random.randrange(0x7fffffff)
    return datetime.utcfromtimestamp(rand_seconds)


def generate_dict():
    result = {}
    keys_nr = random.choice(range(1, 6))
    # we dont want infitite recursion
    value_factories = [f for t, f in VALUE_GENERATORS.items() if t is not dict]
    for idx in range(keys_nr):
        key = generate_str()
        result[key] = random.choice(value_factories)()
    return result

VALUE_GENERATORS = {
    int: generate_int,
    str: generate_str,
    float: generate_float,
    bool: generate_bool,
    datetime: generate_datetime,
    dict: generate_dict
}


def generate(sch, max_tries=200, ensure_valid=True):
    s = sch._schema
    while isinstance(s, schema.Schema):
        s = s._schema
    # Not using isinstance, because schema doesn't
    T = type(s)
    if T in (list, tuple, set, frozenset):
        count = random.randrange(0, MAX_LEN)
        items = [generate(schema.Schema(schema.Or(*s)), max_tries) for _ in range(count)]
        result = T(items)
    elif T is dict:
        result = {}
        for k, sv in s.items():
            if isinstance(k, schema.Optional):
                # Do not generate optional items
                continue
            if callable(getattr(k, 'validate', None)) or type(k) in (type, list, tuple, set, frozenset, dict) or callable(k):
                raise NotImplementedError
            result[k] = generate(schema.Schema(sv), max_tries)
            # Note: this consider optional items as mandatory
    elif T is schema.Or:
        option = random.choice(s._args)
        result = generate(schema.Schema(option), max_tries)
    elif T is schema.And:
        valid = False
        tries_left = max_tries
        while not valid and tries_left > 0:
            candidate = generate(schema.Schema(s._args[0]), max_tries)
            try:
                result = s.validate(candidate)
                valid = True
            except schema.SchemaError:
                tries_left -= 1
                if not ensure_valid:
                    # Accept candidate anyway
                    result = candidate
                    valid = True
        if not valid:
            raise ValueError("Couldn't satisfy And() schema")
    elif T is type:
        if s in VALUE_GENERATORS:
            result = VALUE_GENERATORS[s]()
        else:
            raise NotImplementedError
    elif callable(getattr(s, 'validate', None)):
        raise NotImplementedError
    else:
        result = s
    assert not ensure_valid or result == sch.validate(result)
    return result


def _mutate_insert(seq):
    if seq:
        i = random.randrange(len(seq))
        # duplicate i-th element
        return seq[:i] + seq[i:i + 1] + seq[i:]
    else:
        return type(seq)([None])


def _mutate_delete(seq):
    if seq:
        i = random.randrange(len(seq))
        # duplicate i-th element
        return seq[:i] + seq[i + 1:]
    else:
        return seq


def _mutate_modify(seq):
    if seq:
        i = random.randrange(len(seq))
        # duplicate i-th element
        return seq[:i] + type(seq)([_mutate(seq[i])]) + seq[i + 1:]
    else:
        return seq


def _mutate_swap(seq):
    if len(seq) >= 2:
        i = random.randrange(len(seq) - 1)
        # duplicate i-th element
        return seq[:i] + seq[i + 1:i - 1:-1] + seq[i + 2:]
    else:
        return seq

MUTATORS = {
    float: [
        lambda i: i + 1,
        lambda i: i - 1,
        lambda i: -i,
        lambda i: i + 1e-10,
        lambda i: i - 1e-10,
        str,
        int,
    ],
    int: [
        lambda i: i + 1,
        lambda i: i - 1,
        lambda i: -i,
        str,
        float,
    ],
    str: [
        lambda i: i.upper(),
        lambda i: i * 2,
        lambda i: ' ' + i,
        lambda i: i + ' ',
        lambda i: ''
    ],
    None: [
        str,
        lambda i: None
    ],
    list: [
        _mutate_insert,
        _mutate_delete,
        _mutate_modify,
        _mutate_swap,
        str,
        tuple,
    ],
    tuple: [
        _mutate_insert,
        _mutate_delete,
        _mutate_modify,
        _mutate_swap,
        str,
        tuple,
    ]
}


def _mutate(value):
    T = type(value)
    if T in MUTATORS:
        m = random.choice(MUTATORS[T])
        return m(value)
    else:
        raise TypeError("Can't mutate value of %r" % T)


def generate_invalid(sch, iterations=10):
    feature = generate(sch, 1, ensure_valid=False)
    for change in range(iterations):
        try:
            feature = sch.validate(feature)
            # If we're here, feature is still valid. Mutate
            feature = _mutate(feature)
        except schema.SchemaError:
            return feature
    raise ValueError("Couldn't falsify schema")
