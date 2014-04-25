# -*- coding: utf-8 -*-
u"""Run all experiments defined on a json file, storing results on database.

Usage:
    run_experiments.py <configs.json> <dbname> [--dbserver=<dbserver>]

Options:
 -h --help              Show this screen.
 --version              Show Version.
 --dbserver=<dbserver>  URI of the mongodb server for storing results. Typically "ip:port" [default: localhost]
"""
from __future__ import division
import json
import logging
import sys

from docopt import docopt
from progress.bar import Bar

from featureforge.experimentation.stats_manager import StatsManager


def _run(opts, runner, conf_extender):
    stats = StatsManager(db_uri=opts[u"--dbserver"],
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


def main(runner, conf_extender=None, version=u'Run experiments 0.1'):
    command_name = sys.argv[0]
    custom__doc__ = __doc__.replace(u'run_experiments.py', command_name)
    logging.basicConfig(level=logging.DEBUG,
                        format=u"\n%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    options = docopt(custom__doc__, version=version)
    _run(options, runner, conf_extender)
