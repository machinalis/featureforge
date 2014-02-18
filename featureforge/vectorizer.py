import logging
from sklearn.pipeline import Pipeline
from featureforge.evaluator import FeatureEvaluator
from featureforge.flattener import FeatureMappingFlattener

logger = logging.getLogger(__name__)


class Vectorizer(object):

    def __init__(self, features):
        pipeline = []

        # Feature evaluation
        pipeline.append(("feature_evaluation", FeatureEvaluator(features)))

        # Feature vectorization
        pipeline.append(("feature_mapping", FeatureMappingFlattener()))

        # Build pipeline
        self.pipeline = Pipeline(pipeline)

    def fit(self, X, y=None):
        logger.info("fitting started")
        self.pipeline.fit(X, y)
        logger.info("fitting finished")
        return self

    def transform(self, X):
        logger.info("Hammer transforming started")
        return self.pipeline.transform(X)

    def predict(self, data_points):
        logger.info("Hammer prediction started")
        return self.pipeline.predict(data_points)
