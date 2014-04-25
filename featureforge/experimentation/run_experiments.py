# -*- coding: utf-8 -*-
u"""Run all experiments defined on a json file, storing results on database.

Usage:
    run_experimentation.py <configs.json> <server> <dbname> <experiment_runner> [--dynconf=<dynconf>]

Options:
 -h --help              Show this screen.
 --version              Show Version.
 --dynconf=<dynconf>    Path to function that dynamically augment each config


Where server is the uri of the mongodb server that stores the experimental
results... typically "ip:port".

Experiment runner <experiment_runner> is the python path for importing the function that
tooks a config as arguments, executes the experiment, and returns a python dictionary
describing the results.
Dynamic Config <dynconf> is a python path for a callable that extends on run time each
configuration dict. Examples of usage of this are:
    - attaching the current git_version (with utils.get_git_info)
    - some hash value of some dataset you are using
    - etc

"""
from __future__ import division
import importlib
import json
import logging

from docopt import docopt
from progress.bar import Bar

from featureforge.experimentation.stats_manager import StatsManager


def main(opts, runner, conf_extender):
    stats = StatsManager(db_uri=opts[u"<server>"],
                         db_name=opts[u"<dbname>"])

    configs = json.load(open(opts[u"<configs.json>"]))
    bar = Bar(u'Processing', max=len(configs))
    for config in configs:
        # Extend individual experiment config with the dynamic extender, if any
        if conf_extender is not None:
            conf_extender(config)

        # Book experiment
        ticket = stats.book_if_available(config)
        if ticket is None:
            bar.next()
            continue

        # Run experiment
        try:
            result = runner(config)
        except KeyboardInterrupt:
            logging.error(u"Interrupted by keyboard, terminating...")
            break
        except Exception as e:
            bar.next()
            logging.error(u"Experiment failed because of {} {}, "
                          u"skipping...".format(type(e).__name__, e))
            continue
        else:
            # Store result
            bar.next()
            if not stats.store_results(ticket, result):
                logging.error(u"Experiment successful but could not stored! "
                              "Skipping... ")
    bar.finish()


def import_from_string(string):
    import_path, import_name = string.rsplit('.', 1)
    mod = importlib.import_module(import_path)
    return getattr(mod, import_name)


if __name__ == u"__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format=u"\n%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    options = docopt(__doc__, version=u'Run configurations 0.1')
    runner = import_from_string(options[u'<experiment_runner>'])
    if options.get(u'--dynconf', None):
        conf_extender = import_from_string(options[u'--dynconf'])
    else:
        conf_extender = None

    main(options, runner, conf_extender)
