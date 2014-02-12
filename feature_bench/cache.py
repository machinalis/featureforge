import logging
from functools32 import lru_cache
from feature_bench.disk_cache import CacheOnDisk
from feature_bench import settings

logger = logging.getLogger(__name__)


if 'ram' in settings.ENABLED_CACHES:
    logger.debug('RAM cache enabled')
else:
    def identity(*d_args, **d_kwargs):
        def _identity(func):
            return func
        return _identity
    lru_cache = identity
    logger.debug('RAM cache disabled')

_disk_cache_singleton = CacheOnDisk()
if 'disk' in settings.ENABLED_CACHES:
    logger.debug('DISK cache enabled')
else:
    _disk_cache_singleton.disabled = True
    logger.debug('DISK cache disabled')


def disk_cache(user_function):
    """This decorator was designed as a 2nd level to put after the ram cache like this:

    @lru_cache(maxsize=5)  # the number you want
    @disk_cache
    def my_function(key, disk_cache_key, ...stuff):
        ...

    Because of that, it's mandatory for the user-function to take the 2 arguments ram_key
    and disk_key, where the former is an object containing the samples on attribute "X",
    and the latter just a string.
    """
    def func(ram_key, disk_key, *args, **kwargs):
        samples = ram_key.X
        disk_cache_key = _disk_cache_singleton.adjust_key(disk_key, samples)
        from_disk = _disk_cache_singleton.get_data(disk_cache_key)
        if from_disk is not None:
            logger.info("Found on Disk.")
            return from_disk
        else:
            result = user_function(ram_key, disk_key, *args, **kwargs)
            _disk_cache_singleton.store_data(disk_cache_key, result)
            return result
    return func
