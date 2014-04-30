Experimentation Support
=======================

Key concepts
------------

As a Machine Learning developer, we face a lot of times the need of doing some
fine tuning of some feature, some classifier, regressor, etc. And when the
theory was not enough, we just experiment.
Running the same script several times, each with small differences on the
arguments, is not hard.
But when it comes to be able to later decide which of the executions provided
the best performance/accuracy/whatever you want to measure, it's not that easy.
And to be honest, it's a bit boring to have to be writing down each experiment
result (and it's initial configuration) by hand.


The basics
----------

Feature Forge **experimentation** module can help you to automate the
execution of your experiments, storing the results of each one on a database
that can be queried later for decision taking.

In its most basic form, on an empty file (let's name it *my_experiments.py*) you just define an experiment as a function

.. code-block:: python

    def train_and_evaluate_classifier(config_dict):
        # creates a classifier based on config_dict,
        # later trains it with some fixed train data and
        # evaluates it with some other fixed test data
        result = do_stuff(...)
        return {
            'accuracy': ...,
            'elapsed_time': ...,
            'other_metric': ...
        }

Your function should take a single argument (which is a dictionary that defines
the configuration of the experiment) and return another dictionary with the
results of the experiment.

After that, add a few lines wrapping your function like this:

.. code-block:: python

    from featureforge.experimentation import runner


    def train_and_evaluate_classifier(config_dict):
        ... # your function


    if __name__ == '__main__':
        runner.main(train_and_evaluate_classifier)

And that's it, you have a ready to use experiments runner, that will:

 - accept from the command line a JSON file containing a sequence of dicts (each of those will be passed to your function as an experiment configuration)
 - log the results on a MongoDB (you need to provide URI and database name)

For more details, just run:

.. code-block:: bash

    $ python my_experiments.py -h


Parallelism
-----------

The experiment runner provides a simple but effective support for running
several experiments in parallel, by simply running the same script several
times.
Each time that an experiment is about to be started, the runner attempts to
book it on the database. If it was already booked, then that experiment
will be ignored by this runner, and the next configuration will be attempted.

In this way, you can run this script as many times as desired, even from different
computers, all of them booking and saving experiment results to a shared
database.

Tips:
 - Monitor the memory usage of each experiment. Running several in parallel may use all the memory available, slowing down the entire experimentation.
 - Bookings are not forever. By default they last 10 minutes, but you can set it to whatever you want. Once that an experiment booking expires, it may be booked again and re-run by anyone.


Dynamic experiment configuration
--------------------------------

In the general case, among the static experiment configuration, it's very
important to also provide some info about the context in which the experiment
ran.

So, following our example of experimenting to get the best classifier, besides having the classifier name and it's parameters as part of a given experiment, like this

.. code-block:: python

    single_experiment_config = {
        "regression_method": "dtree",
        "regression_method_configuration": {
            "min_samples_split": 25
            }
        }

it's equally important to be sure that all experiments were run with a the same
version of code, or that the data-sets used for training and testing are always
the same, etc. If you have more than one evaluation data set it's important
to be able to find out which results correspond to each data set.

Because of that need, we highly recommend you to define a *configuration
extender*, like this


.. code-block:: python

    from featureforge.experimentation import runner

    def train_and_evaluate_classifier(config_dict):
        ... # your function

    def extender(config):
        """
        Receives a copy of the the experiment configuration before
        attempting to book, and returns it modified with the extra
        details desired.
        """
        # whatever you want, for instance:
        config['train_data_hash'] = your_md5_function('train', ...)
        config['test_data_hash'] = your_md5_function('test', ...)
        config['code_version'] = your_definition_of_current_version(...)
        ...
        return config

    if __name__ == '__main__':
        runner.main(train_and_evaluate_classifier, extender)

We provide a built-in utility for using the current git branch (and modifications) as part of the configuration:

.. code-block:: python

    if __name__ == '__main__':
        runner.main(
            train_and_evaluate_classifier,
            use_git_info_from_path='/path/to/my_repo/')

For other version control systems, or any other things you may need, use the
*extender* callback.


Exploring the finished experiments
----------------------------------

Once you run all the experiments, you will have everything stored on the MongoDB.
For each experiment, it's configuration and results will be stored on a single Document, like this:

    - Field "`marshalled_key`": string text representing the hashed experiment configuration. Used as identifier for bookings.
    - Field "`experiment_status`": one of the following
        - "`status_booked`": experiment was booked but not finished yet.
        - "`status_solved`": experiment was reported as finished.
    - Field "`booked_at`": time-stamp of the experiment booking.
    - Field "`results`": only available for finished experiments. It's a dictionary that contain as sub-fields all the results of the experiment.
    - Any other field on the Document, was part of the experiment configuration.

You can access, filter and see the finished experiments simply using the mongo shell, or with python like this:

.. code-block:: python

    from featureforge.experimentation.stats_manager import StatsManager

    sm = StatsManager(None, 'Your-database-name')

    for experiment in sm.iter_results():
        print(experiment.results)



Important Notes and Details
---------------------------

Configuration dicts
~~~~~~~~~~~~~~~~~~~

- Simple data types:

    In order to easily create booking-tickets from configuration dictionaries, they can't contain more than built-in objects (sets, lists, tuples, strings, booleans or numbers).

- Lists, tuples or sets, be careful with the ordering:

    Be very careful with config value that are sequences. If your experiment
    configuration needs to provide the features to use, probably their ordering
    is not important, so you should pass them as a `set`, and not as a tuple
    or a list. Otherwise, these 2 configurations are going to be treated as
    different when they shouldn't:

    .. code-block:: python

        config_a = {
            "regression_method": "dtree",
            "regression_method_configuration": {
                "min_samples_split": 25
                },
            "features": ["FeatureA", "FeatureB", "FeatureC"]
            }

        config_a_again = {
            "regression_method": "dtree",
            "regression_method_configuration": {
                "min_samples_split": 25
                },
            "features": ["FeatureC", "FeatureA", "FeatureB"]
            }
