Feature Forge
=============

This library provides a set of tools that can be useful in many machine
learning applications (classification, clustering, regression, etc.), and
particularly helpful if you use scikit-learn (although this can work if
you have a different algorithm).

Most machine learning problems involve an step of feature definition and
preprocessing. Feature Forge helps you with:

 * Defining and documenting features
 * Testing your features against specified cases and against randomly generated
   cases (stress-testing). This helps you making your application more robust
   against invalid/misformatted input data. This also helps you checking that
   low-relevance results when doing feature analysis is actually because the
   feature is bad, and not because there's a slight bug in your feature code.
 * Evaluating your features on a data set, producing a feature evaluation
   matrix. The evaluator has a robust mode that allows you some tolerance both
   for invalid data and buggy features.
 * Experimentation: running, registering, classifying and reproducing
   experiments for determining best settings for your problems.

Installation
------------

Just `pip install featureforge`.

Documentation
-------------

Documentation is available at http://feature-forge.readthedocs.org/en/latest/

Contact information
-------------------

Feature Forge is © 2014 Machinalis (http://www.machinalis.com/). Its primary
authors are:

 * Javier Mansilla <jmansilla@machinalis.com> (jmansilla at github)
 * Daniel Moisset <dmoisset@machinalis.com> (dmoisset at github)
 * Rafael Carrascosa <rcarrascosa@machinalis.com> (rafacarrascosa at github)

Any contributions or suggestions are welcome, the official channel for this is
submitting github pull requests or issues.

Changelog
---------
0.1.6:
    - Bug fixes related to sparse matrices.
    - Small documentation improvements.
    - Reduced default logging verbosity.

0.1.5:
    - Using sparse numpy matrices by default.

0.1.4:
    - Discarded the need of using forked version of Schema library.

0.1.3:
    - Added support for running and generating stats for experiments

0.1.2:
    - Fixing installer dependencies

0.1.1:
    - Added support for python 3
    - Added support for bag-of-words features

0.1:
    - Initial release
