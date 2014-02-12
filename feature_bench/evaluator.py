from collections import defaultdict
import logging

from feature_bench.cache import disk_cache, lru_cache


logger = logging.getLogger(__name__)


LOG_STEP = 500


class FeatureCacheKey(object):

    def __init__(self, X, features):
        self.X = X
        self.features = features

    def _key(self):
        return (id(self.X),) + tuple(f.name for f in self.features)

    def __eq__(self, other):
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


class FeatureEvaluator(object):

    def __init__(self, features, data_hash_key=None):
        self.features = features
        self.training_stats = None
        if data_hash_key is not None:
            self.disk_cache_key_transform = 'evaluation_transform_%s' % data_hash_key
        else:
            self.disk_cache_key_transform = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logger.info("Lookup feature evaluation id=%d from cache", id(X))
        features = self.features
        is_train = True
        if self.training_stats is not None:
            is_train = False  # We dont support incremental trainings.
            to_exclude = self.training_stats['excluded_features']
            features = [f for f in features if f not in to_exclude]
        k = FeatureCacheKey(X, features)
        result, stats = self._transform(k, self.disk_cache_key_transform, is_train)
        if self.training_stats is None:
            self.training_stats = stats
        return result

    @staticmethod
    @lru_cache(maxsize=4)
    @disk_cache
    def _transform(key, disk_cache_key, is_train):
        """This is a static method, because we need not to include "self" as an argument,
        otherwise lru_cache wont be able to it's magic when called from different
        instances"""
        X = key.X
        logger.info("Starting feature evaluation id=%d", id(X))
        ae = ActualEvaluator(key.features)
        result, stats = ae.transform(X, train_mode=is_train)
        logger.info("Finished feature evaluation id=%d", id(X))
        return result, stats


class ActualEvaluator(object):
    """Actual Evaluator Manager.

    The FeatureEvaluator is more a wrapper that handles caching or actual-evaluation
    depending on the case.
    This instead, is the implementation of the algorithm needed for computing the
    pure evaluation of features, with some tolerance to failures.
    """

    # When evaluating the first N data-points, if there's an error with a feature, that
    # feature is automatically excluded from the regressor (strict mode).
    # If error occurs on later data-points and FEATURE_MAX_ERRORS_ALLOWED is not exceeded
    # the feature default (if present) will be returned instead; if not, the feature will
    # be excluded from the regressor.
    FEATURE_STRICT_UNTIL = 100
    FEATURE_MAX_ERRORS_ALLOWED = 5

    class NoFeaturesLeftError(Exception):
        pass

    def __init__(self, features):
        self.features = features[:]
        self.excluded_features = set()
        self.failure_stats = {
            'discarded_samples': [],
            'features': defaultdict(list)
        }

    def transform(self, X, y=None, train_mode=True):
        result, X_to_retry = self._transform(X, y, train_mode)
        while X_to_retry:
            logger.info('Retrying for %s samples that were originally discarded.' %
                        len(X_to_retry))
            result_2, X_to_retry = self._transform(X_to_retry[:], y, train_mode)
            result += result_2
        stats = {'discarded_samples': self.failure_stats['discarded_samples'],
                 'excluded_features': self.excluded_features}
        return result, stats

    def _transform(self, X, y, train_mode):
        result = []
        self._samples_to_retry = []
        for i, d in enumerate(X):
            r = {}
            for feature in self.features[:]:
                try:
                    r[feature.name] = feature(d)
                except Exception, e:
                    self.process_failure(result, e, feature, d, i)
                    if not train_mode:
                        raise
                    break
            else:
                result.append(r)
            # some progress logging
            if i and i % LOG_STEP == 0:
                logger.debug('Features evaluated for %s samples' % i)
        logger.debug('Features evaluated for all (%s) samples' % (i+1))
        recommended_retries = self._samples_to_retry
        return result, recommended_retries

    def process_failure(self, partial_evaluation, error, feature, dpoint, d_index):
        logger.warning(u'Fail evaluating %s: %s %s' % (feature.name, type(error), error))
        self.failure_stats['discarded_samples'].append(dpoint.get('pk', 'PK-NOT-FOUND'))
        feature_errors = self.failure_stats['features'][feature.name]
        feature_errors.append(dpoint)
        if d_index < self.FEATURE_STRICT_UNTIL:
            self.exclude_feature(feature, partial_evaluation)
        elif len(feature_errors) > self.FEATURE_MAX_ERRORS_ALLOWED:
            self.exclude_feature(feature, partial_evaluation)

    def exclude_feature(self, feature, partial_evaluation):
        self.features.remove(feature)
        if not self.features:
            raise self.NoFeaturesLeftError()
        self.excluded_features.add(feature)
        for evaluation in partial_evaluation:
            evaluation.pop(feature.name, None)
        discarded_samples = self.failure_stats['features'][feature.name]
        self._samples_to_retry += discarded_samples
