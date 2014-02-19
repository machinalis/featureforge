from collections import defaultdict
from copy import copy
import logging

logger = logging.getLogger(__name__)


LOG_STEP = 500


class FeatureEvaluator(object):
    """Simple feature evaluator"""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.alive_features = tuple(self.features)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X, y=None):
        for d in X:
            yield tuple((f(d) for f in self.alive_features))


class TolerantFeatureEvaluator(object):

    def __init__(self, features):
        self.features = features
        self.training_stats = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = self.features
        is_train = True
        if self.training_stats is not None:
            is_train = False  # We dont support incremental trainings.
            to_exclude = self.training_stats['excluded_features']
            features = [f for f in features if f not in to_exclude]

        logger.info("Starting feature evaluation id=%d", id(X))
        ae = ActualEvaluator(features)
        result = ae.transform(X, train_mode=is_train)
        logger.info("Finished feature evaluation id=%d", id(X))

        if self.training_stats is None:
            self.training_stats = ae.get_training_stats()
        return result


class ActualEvaluator(object):
    """Actual Evaluator Manager.

    The FeatureEvaluator is more a wrapper that handles what to do differently
    if training or not.
    This instead, is the implementation of the algorithm needed for computing
    the pure evaluation of features, with some tolerance to failures.
    """

    # Tolerance to failures:
    # a) Samples are always discarded when failing:
    #     If a given sample fails when evaluating a feature with it, no matter
    #     what, no matter when, the sample is discarded.
    # b) Features can be discarded or not, depending on the configuration:
    #     - When evaluating the first N data-points, if there's an error with a
    #       feature, that feature is automatically excluded (strict mode).
    #     - If error occurs on later data-points and FEATURE_MAX_ERRORS_ALLOWED
    #       was exceeded for that feature, the feature is discarded. If not
    #       exceeded, an internal counter will be increated for that feature.
    #    Each time that a feature is discarded, the following things happen:
    #      - every sampled that was discarded because of this feature,
    #        is re-considered.
    #      - every previously evaluated sample is stripped off of the result
    #        of this feature.
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
        self.training_stats = {}

    def get_training_stats(self):
        # returns a copy of the stats computed during training
        return {
            'discarded_samples': copy(self.failure_stats['discarded_samples']),
            'excluded_features': copy(self.excluded_features)
        }

    def transform(self, X, train_mode=True):
        result, X_to_retry = self._transform(X, train_mode)
        while X_to_retry:
            logger.info('Retrying for %s samples that were originally discarded.' %
                        len(X_to_retry))
            result_2, X_to_retry = self._transform(X_to_retry[:], train_mode)
            result += result_2
        return (tuple(r) for r in result)

    def _transform(self, X, train_mode):
        result = []
        self._samples_to_retry = []
        for i, d in enumerate(X):
            r = []
            for feature in self.features[:]:
                try:
                    r.append(feature(d))
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
        logger.debug('Features evaluated for all (%s) samples' % (i + 1))
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
        idx = self.features.index(feature)
        self.features.remove(feature)
        if not self.features:
            raise self.NoFeaturesLeftError()
        self.excluded_features.add(feature)
        for evaluation in partial_evaluation:
            evaluation.pop(idx)
        discarded_samples = self.failure_stats['features'][feature.name]
        self._samples_to_retry += discarded_samples
