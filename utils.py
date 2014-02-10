from copy import deepcopy
import json
import hashlib


class DictNormalizer(object):
    # In order to be able to hash a dict, we'll ensure that all data types are simple
    # enough, and that sets are treated as sorted lists

    class UnHashableDict(Exception):
        pass

    def normalize_value(self, value):
        # If it's a simple data type, just return it
        if isinstance(value, (int, float, long, complex, basestring, type(None))):
            return value
        # If it's a set, make it a sorted list instead, so it's deterministic
        if isinstance(value, set):
            value = sorted(list(value))

        # And now resolve the "recursive" cases
        if isinstance(value, dict):
            return self.map_to_key(value)
        elif isinstance(value, (list, tuple)):
            return self.seq_to_key(value)
        # If none of the previous, better crash than hidding the issue
        raise self.UnHashableDict('Cant hash "%s" of type "%s"' % (value, type(value)))

    def map_to_key(self, mapping):
        return dict((k, self.normalize_value(v)) for k, v in mapping.iteritems())

    def seq_to_key(self, sequence):
        SeqType = type(sequence)
        return SeqType(map(self.normalize_value, sequence))

    def __call__(self, obj):
        return self.normalize_value(obj)


def hash_dict(d):
    normalized = DictNormalizer()(deepcopy(d))
    serialized = json.dumps(normalized, sort_keys=True)
    return hashlib.md5(serialized).hexdigest()
