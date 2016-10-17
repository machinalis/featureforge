import os
import os.path
import sys

from future.builtins import str

PY2 = sys.version < '3'
# Tuple of the supported types for individual elements.
NORM_DICT_SIMPLE_TYPES = (int, float, bytes, str, complex, type(None))
if PY2:
    NORM_DICT_SIMPLE_TYPES += (long, )


class DictNormalizer(object):
    """Utility that's able to create hashes for "simple" dictionaries.

    By "simple" we mean that:
        - Both keys and values must be instances of builtin data types (ie,
          no custom datatypes are supported)
        - Sets are transformed into sorted lists.
            - recursivity is allowed (both in form of sequences or dicts)
    """
    # In order to be able to hash a dict, we'll ensure that all data types are simple
    # enough, and that sets are treated as sorted lists

    class UnHashableDict(Exception):
        pass

    def normalize_value(self, value):
        # If it's a simple data type, just return it
        if isinstance(value, NORM_DICT_SIMPLE_TYPES):
            return value
        # If it's a set, make it a sorted list instead, so it's deterministic
        if isinstance(value, set):
            value = sorted(list(value))

        # And now resolve the "recursive" cases
        if isinstance(value, dict):
            return self._map_to_key(value)
        elif isinstance(value, (list, tuple)):
            return self._seq_to_key(value)
        # If none of the previous, better crash than hidding the issue
        raise self.UnHashableDict('Cant hash "%s" of type "%s"' % (value, type(value)))

    def _map_to_key(self, mapping):
        return dict((k, self.normalize_value(v)) for k, v in mapping.items())

    def _seq_to_key(self, sequence):
        SeqType = type(sequence)
        return SeqType(map(self.normalize_value, sequence))

    def __call__(self, obj):
        return self.normalize_value(obj)


def get_git_info(repo_path):
    """
    Parse repo information, return a summary formatted like
    "hash [branch] extra"

    Where extra informs if there are local changes in some files
    """
    cwd = os.getcwd()
    try:
        # Switch to a dir where the repo is
        os.chdir(repo_path)
        # Get data
        head_hash = os.popen("git show-ref --head HEAD").read().split()[0]
        current_branch = os.popen("git symbolic-ref HEAD").read().strip()
        if current_branch.startswith("refs/heads/"):
            current_branch = current_branch[len("refs/heads/"):]
        changes = os.popen("git diff-index --name-only HEAD").read().split('\n')
        changes = filter(None, changes)  # Remove empty lines
        if changes:
            changelist = " local changes in " + ", ".join(changes)
        else:
            changelist = ""
        return "%s [%s]%s" % (head_hash[:10], current_branch, changelist)
    finally:
        # Always recover original running directory, just in case
        os.chdir(cwd)
