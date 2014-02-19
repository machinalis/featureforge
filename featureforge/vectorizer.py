import logging

from featureforge.evaluator import FeatureEvaluator, TolerantFeatureEvaluator
from featureforge.feature import make_feature
from featureforge.flattener import FeatureMappingFlattener

logger = logging.getLogger(__name__)


class Vectorizer(object):

    def __init__(self, features, tolerant=False):
        # Upgrade `features` to `Feature` instances.
        features = map(make_feature, features)
        if tolerant:
            self.evaluator = TolerantFeatureEvaluator(features)
        else:
            self.evaluator = FeatureEvaluator(features)
        self.flattener = FeatureMappingFlattener()

    def fit(self, X, y=None):
        Xt = self.evaluator.fit_transform(X, y)
        self.flattener.fit(Xt, y)
        return self

    def fit_transform(self, X, y):
        Xt = self.evaluator.fit_transform(X, y)
        return self.flattener.fit_transform(Xt, y)

    def transform(self, X):
        Xt = self.evaluator.transform(X)
        return self.flattener.transform(Xt)

    def column_to_feature(self, i):
        """
        Given a column index in the vectorizer's output matrix it returns the
        feature that originates that column.

        The return value is a tuple (feature, value).
        `feature` is the feature given in the initialization and `value`
        depends on the kind of feature that the column represents:
            - If the feature spawns numbers then `value` is `None` and should
              be ignored.
            - If the feature spawns strings then `value` is the string that
              corresponds to the one-hot encoding of that column.
            - If the feature spawns an array then `value` is the index within
              the spawned array that corresponds to that column.
        """
        j, value = self.flattener.reverse[i]
        feature = self.evaluator.alive_features[j]
        return feature, value
