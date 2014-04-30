Feature evaluation
==================

The most typical use case for features is evaluating a set of them on a
collection of data points to feed that information into a machine learning
algorithm. FeatureForge provides an evaluation tool that, given a list of
features and a collection of data points allows you to produce a matrix
with the evaluation results in a format suitable for input into scikit-learn
algorithms.

In addition to that, some tools are provided to map back the results (which
are essentially numeric) into feature names in order to make analysis of the
data easier

Basic usage
-----------

The core class for applying features to values is
`featureforge.vectorizer.Vectorizer`. This class is instantiated with a list
of features to be used, and then follows the standard pipeline API from scikit
learn described at <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>.

Roughly, to use it standalone what you do is::

    v = Vectorizer([some_feature, some_other_feature])
    # data is a sequence of data points
    v.fit(data)
    result = v.transform(data)

After this, `result` contains a matrix with the feature evaluations (see the
*Generated output*) section below.

In a more typical use, you'll want to put this inside a scikit-learn pipeline,
something like this::

    from sklearn import Pipeline
    v = Vectorizer([some_feature, some_other_feature])
    ... build other steps like classificators/regressors ...
    p = Pipeline([v, step2, step3])

    p.fit(data)
    # Here you can use p methods depending on what you built. See the scikit
    # documentation for examples

Generated output
----------------

The output for the vectorizer (i.e., the result of its transform() method) is
a matrix with one row per each data point, and some columns for each feature.
The mapping from feature to columns depends on the type of the feature value:

 * Numeric values are mapped to a column with the numeric value
 * Enumerated values are mapped to one column for each possible value of the
   enumeration, with values 0 or 1. For example a feature with possible values
   "animal", "vegetable", "mineral" will be mapped to 3 columns, and for each
   row only one will have a 1 and the rest will have zeroes.
 * Vector features are mapped to a number of columns equal to the size of the
   vector.
 * Bag of Words values, very similary to Enumerated values, are mapped to one
   column for each possible value, with values from 0 to the number of times
   that each value appears in the bag. For example a feature with possible
   values "red", "green", "blue" will be mapped to 3 columns, and for each row
   may be all zeroes if evaluated with empty sequence, all ones if evaluated
   with ["red", "green", "blue"], or a 3 and two zeroes if evaluated with
   ["red", "red", "red"]

One consequence of this is that any tool that operates on your data afterwards
and returns column indexes will return numbers that are not trivially mapped to
your features. For example, some scikit-learn algorithms provide some analysis
of the matrix telling you which columns are best correlated with some property.
If you use those you will get a result like "columns 37 and 42 have high
correlation", but you probably want to know the name of the features which
are related to columns 37 and 42. To do so, the vectorizer provides a
`column_to_feature(i)` method, which takes a column number and returns a tuple
(feature, info), where feature is the original feature (remember that you can
get a better description printing f.name when f is a feature). The second
field in the result tuple is `None` for numeric features, the label for
enumerated or bag features (i.e., "animal", "vegetable" or "mineral") and the index
for vector features.


Sparse vs Dense Matrices
------------------------

By default, Vectorizer will construct a sparse numpy matrix which in the general case will consume significanly less memory.
Anyway, by passing `sparse=False` as an argument when instantiating `Vectorizer` you can change this to use a dense matrix instead.


Tolerant evaluation
-------------------

If your data is not completely clean or there are bugs in the implementation of
your features, it is possible that during vectorization an exception will be
raised, which stops the process. This is specially annoying if this happens
after a long run time (computing features can take a lot of time for big
datasets and or complex features) just because a silly bug that is triggered in
a single broken or unusual datapoint among millions of them.

For many experiments, the result you get is useful even if you drop some
samples, or even if you ignore a few buggy features while letting the other ones
be evaluated.

FetureForge provides a "tolerant" version of the vectorizer that can be used
by passing `tolerant=True` as an argument when instantiating `Vectorizer`.
When in tolerant mode, the following things happen:

 * Data points are always discarded when failing: If a given sample fails when
   evaluating a feature with it, no matter what, no matter when, the data point
   is discarded.
 * If a feature evaluation fails in the first 100 samples, it's assumed to be
   broken and it's excluded.
 * If a feature evaluation fails more than 5 times, it's assumed to be
   broken and it's excluded.
 * When a feature is excluded, samples discarded because of that feature are
   re-checked.

Right now, the configuration values for the policy are hardcoded.

Note that the process described above can result on a matrix that is missing
some rows (data points) and some columns (features).
