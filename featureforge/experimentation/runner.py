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
from copy import copy
import json
import logging
import sys

from docopt import docopt
from progress.bar import Bar

from featureforge.experimentation.stats_manager import StatsManager
from featureforge.experimentation.utils import get_git_info

# Measured in seconds
BOOKING_DURATION = 10 * 60  # just a default


def main(single_runner,
         conf_extender=None,
         booking_duration=BOOKING_DURATION,
         use_git_info_from_path=None,
         version=u'Run experiments 0.1'):
    command_name = sys.argv[0]
    custom__doc__ = __doc__.replace(u'run_experiments.py', command_name)
    logging.basicConfig(level=logging.DEBUG,
                        format=u"\n%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    opts = docopt(custom__doc__, version=version)

    stats = StatsManager(booking_duration=booking_duration,
                         db_name=opts[u"<dbname>"],
                         db_uri=opts[u"--dbserver"])

    experiment_configurations = json.load(open(opts[u"<configs.json>"]))
    bar = Bar(u'Processing', max=len(experiment_configurations))
    if use_git_info_from_path is not None:
        GIT_INFO = get_git_info(use_git_info_from_path)
    else:
        GIT_INFO = None
    for config in experiment_configurations:
        # Extend individual experiment config with the dynamic extender, if any
        config = copy(config)
        if conf_extender is not None:
            config = conf_extender(config)
        # Adding GIT info to the config if computed and not present
        if GIT_INFO is not None and u'git_info' not in config:
            config[u'git_info'] = GIT_INFO

        # Book experiment
        ticket = stats.book_if_available(config)
        if ticket is None:
            bar.next()
            continue

        # Run experiment
        try:
            result = single_runner(config)
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
