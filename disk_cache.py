import cPickle as pickle
from os import makedirs, path, remove
from getpass import getuser
from lockfile import FileLock
import logging


logger = logging.getLogger(__name__)


def only_if_not_disabled(func):
    def _func(self, *args, **kwargs):
        if self.disabled:
            return None
        else:
            return func(self, *args, **kwargs)
    return _func


class CacheOnDisk(object):
    """CacheOnDisk is simply like it's name, a caching useful for reusing the data between
    several processes running the same round of experiments, or to restart it after a
    failure.
    You need to be very careful of the key, to avoid reusing stuff from disk when you
    shouldn't.
    """
    cache_folder_path = '/tmp/cache_on_disk'
    lock_file = 'write.lock'
    disabled = False

    def __init__(self):
        self.cache_folder_path = '%s_%s' % (self.cache_folder_path, getuser())
        if not path.exists(self.cache_folder_path):
            makedirs(self.cache_folder_path)

    def _store_data(self, key, data):
        lock = FileLock(path.join(self.cache_folder_path, self.lock_file))
        with lock:
            file_path = path.join(self.cache_folder_path, key)
            if not path.exists(file_path):
                try:
                    with open(file_path, "wb") as fp:
                        pickler = pickle.Pickler(fp, pickle.HIGHEST_PROTOCOL)
                        pickler.fast = True
                        pickler.dump(data)
                    logger.debug('Stored cache on file %s' % file_path)
                except Exception, e:
                    logging.warning('Failed to dump on disk because of %s: %s' %
                                    (type(e).__name__, e)
                                    )
                    # Let's delete the file if it was created and left corrupt
                    if path.exists(file_path):
                        remove(file_path)

    def _get_data(self, key):
        file_path = path.join(self.cache_folder_path, key)
        logger.debug('Looking for file %s' % file_path)
        if path.exists(file_path):
            try:
                with open(file_path, "rb") as fp:
                    return pickle.load(fp)
            except Exception, e:
                logging.warning('Failed to load %s from disk because of %s: %s' %
                                (file_path, type(e).__name__, e)
                                )
                return None

    @only_if_not_disabled
    def adjust_key(self, key, samples):
        # If key is None, no adjustment will happen.
        # Key shall distinguish sets of experiments from each other.
        # Samples, are the actual samples to which the action needs to be taken,
        # for example, the train-samples-set. We extend the key with the
        # samples size (to make easier to understand pickled things on disk) and with a
        # hash of the samples pks.
        if key is not None:
            try:
                size = len(samples)
            except TypeError:
                logging.warning('Failed to transform key because samples is an '
                                'iterable so we cant compute its length. Skipping '
                                'disk-cache')
                key = None
            else:
                # Now we are sure is not an iterable, we compute the pks hash
                pks_hash = hash(tuple([s.get('pk', None) for s in samples]))
                key = '%s_%s_%s' % (key, pks_hash, size)

        return key

    @only_if_not_disabled
    def get_data(self, key):
        # If key is None, None is returned directly.
        if key is None:
            return None
        else:
            return self._get_data(key)

    @only_if_not_disabled
    def store_data(self, key, data):
        # If key is None, nothing will be stored, and a silent None is returned.
        if key is None:
            return None
        else:
            return self._store_data(key, data)
