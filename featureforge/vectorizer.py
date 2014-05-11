import logging

from future.builtins import map

from featureforge.evaluator import FeatureEvaluator, TolerantFeatureEvaluator
from featureforge.feature import make_feature
from featureforge.flattener import FeatureMappingFlattener

logger = logging.getLogger(__name__)


class Vectorizer(object):
    """
    Vectorizer(features) provides a scikit-learn compatible component that
    given a collection of data points turns it into a matrix of vectors,
    where each vector contains the evaluation of every given feature for a
    single data point.

    Numerical features are mapped to a column of the resulting matrix.
    Enumerated features are mapped to multiple columns (one for each possible
    enumerated value), using 0 or 1 to indicate the presence of the enumerated
    value. Vectorial features are mapped to multiple columns.

    The API of this class follows scikit-learn conventions, see
    http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html


    Vectorizer(features, tolerant=True) changes the feature evaluation strategy
    to one that is more tolerant to failures when evaluating features. It is
    useful for testing and to run rough experiments when you still aren't sure
    if your data is clean or your features are correct. See the documentation
    for featureforge.evaluator.TolerantFeatureEvaluator

    Vectorizer(features, sparse=True) changes the result data type, returning a
    sparse numpy matrix instead of a dense matrix. See the documentation on
    featureforge.flattener.Flattener
    """

    def __init__(self, features, tolerant=False, sparse=True):
        # Upgrade `features` to `Feature` instances.
        features = list(map(make_feature, features))
        if tolerant:
            self.evaluator = TolerantFeatureEvaluator(features)
        else:
            self.evaluator = FeatureEvaluator(features)
        self.flattener = FeatureMappingFlattener(sparse=sparse)

    def fit(self, X, y=None):
        Xt = self.evaluator.fit_transform(X, y)
        self.flattener.fit(Xt, y)
        return self

    def fit_transform(self, X, y=None):
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
