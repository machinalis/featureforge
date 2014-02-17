import logging
from functools32 import lru_cache
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
