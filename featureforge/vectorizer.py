import logging
from featureforge.evaluator import FeatureEvaluator, TolerantFeatureEvaluator
from featureforge.flattener import FeatureMappingFlattener

logger = logging.getLogger(__name__)


class Vectorizer(object):

    def __init__(self, features, tolerant=False):
        if tolerant:
            self.evaluator = TolerantFeatureEvaluator(features)
        else:
            self.evaluator = FeatureEvaluator(features)
        self.flattener = FeatureMappingFlattener()

    def fit(self, X, y=None):
        Xt = self.evaluator.fit_transform(X, y)
        self.flattener.fit(Xt)
        return self

    def fit_transform(self, X, y):
        Xt = self.evaluator.fit_transform(X, y)
        return self.flattener.fit_transform(Xt, y)

    def transform(self, X):
        self.evaluator.transform(X)
        return self.flattener.transform(X)
