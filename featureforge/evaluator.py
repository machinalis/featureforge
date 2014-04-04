from collections import defaultdict
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
    """Feature Evaluator that tolerates broken features or samples when
     fitting.

     Tolerance to failures (only done during fit or fit_transform):
     a) Samples are always discarded when failing:
         If a given sample fails when evaluating a feature with it, no matter
         what, no matter when, the sample is discarded.
     b) Features can be discarded or not, depending on the configuration:
         - When evaluating the first N data-points, if there's an error with a
           feature, that feature is automatically excluded (STRICT mode).
         - If error occurs on later data-points and FEATURE_MAX_ERRORS_ALLOWED
           was exceeded for that feature, the feature is discarded. If not
           exceeded, an internal counter will be increated for that feature.
        Each time that a feature is discarded, the following things happen:
          - every sampled that was discarded because of this feature,
            is re-considered.
          - every previously evaluated sample is stripped off of the result
            of this feature.
    """
    FEATURE_STRICT_UNTIL = 100
    FEATURE_MAX_ERRORS_ALLOWED = 5

    class NoFeaturesLeftError(Exception):
        pass

    def __init__(self, features):
        self.features = features
        self.fitted = False

    def fit(self, X, y=None):
        self._fit_failure_stats = {
            'discarded_samples': [],
            'features': defaultdict(list)
        }
        self.alive_features = self.features[:]

        dataset = X
        # Caution to not work in strict mode when retrying
        last_sample_idx = -1
        while dataset:
            self._samples_to_retry = []
            for i, d in enumerate(dataset, last_sample_idx + 1):
                for feature in self.alive_features[:]:
                    try:
                        feature(d)
                    except Exception as e:
                        self.process_failure([], e, feature, d, i)
                        break
            last_sample_idx = i
            dataset = self._samples_to_retry

        self.alive_features = tuple(self.alive_features)
        self.fitted = True
        return self

    def transform(self, X, y=None):
        for d in X:
            yield tuple((f(d) for f in self.alive_features))

    def fit_transform(self, X, y=None):
        # Very similar to fit alone, but buffers samples evaluation for two
        # reasons:
        #   - simply because this is also a transform, so we need to
        #     return that
        #   - to be able to patch them if at some given point a Feature that
        #     was working is killed.
        self._fit_failure_stats = {
            'discarded_samples': [],
            'features': defaultdict(list)
        }
        self.alive_features = self.features[:]
        result = []

        dataset = X
        # Caution to not work in strict mode when retrying
        last_sample_idx = -1
        while dataset:
            self._samples_to_retry = []
            for i, d in enumerate(dataset, last_sample_idx + 1):
                r = []
                for feature in self.alive_features[:]:
                    try:
                        r.append(feature(d))
                    except Exception as e:
                        self.process_failure(result, e, feature, d, i)
                        break
                else:
                    result.append(r)
            last_sample_idx = i
            dataset = self._samples_to_retry

        self.alive_features = tuple(self.alive_features)
        self.fitted = True
        return (tuple(r) for r in result)

    def process_failure(self, partial_eval, error, feature, dpoint, d_index):
        logger.warning(u'Fail evaluating %s: %s %s' % (feature,
                                                       type(error), error))
        self._fit_failure_stats['discarded_samples'].append(
            dpoint.get('pk', 'PK-NOT-FOUND'))
        feature_errors = self._fit_failure_stats['features'][feature]
        feature_errors.append(dpoint)
        if d_index < self.FEATURE_STRICT_UNTIL:
            self.exclude_feature(feature, partial_eval)
        elif len(feature_errors) > self.FEATURE_MAX_ERRORS_ALLOWED:
            self.exclude_feature(feature, partial_eval)

    def exclude_feature(self, feature, partial_evaluation):
        idx = self.alive_features.index(feature)
        self.alive_features.remove(feature)
        if not self.alive_features:
            raise self.NoFeaturesLeftError()
        for evaluation in partial_evaluation:
            evaluation.pop(idx)
        discarded_samples = self._fit_failure_stats['features'][feature]
        self._samples_to_retry += discarded_samples
