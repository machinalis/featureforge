from copy import deepcopy
from datetime import datetime, timedelta
import json
import hashlib
import logging

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from future.builtins import str

from featureforge.experimentation.utils import DictNormalizer

logger = logging.getLogger(__name__)


EXPERIMENTS_COLLECTION_NAME = 'experiment_data'


def mongo_dict_key_sanitizer(mapping):
    # Mongo does not accept dots or $ to be part of keys. We'll replace them
    items = []
    for k, v in mapping.items():
        if isinstance(k, (str, bytes)):
            k = k.replace('.', ',').replace('$', '&')
        if isinstance(v, dict):
            v = mongo_dict_key_sanitizer(v)
        elif type(v) in (list, tuple, set):
            # we want NamedTuples not to be checked
            _v = []
            for vi in list(v):
                if isinstance(vi, dict):
                    vi = mongo_dict_key_sanitizer(vi)
                _v.append(vi)
            v = type(v)(_v)
        items.append((k, v))
    return dict(items)


class StatsManager(object):
    marshalled_key = 'marshalled_key'
    experiment_status = 'experiment_status'
    results_key = 'results'
    booking_at_key = 'booked_at'
    STATUS_BOOKED = 'status_booked'
    STATUS_SOLVED = 'status_solved'

    def __init__(self, booking_duration, db_name, db_uri=None,
                 keep_running_on_errors=True):
        """
        Creates new instance of Stats Manager.
        Parameters:
            - booking_duration,
            - db_name: Name of the mongo database to use (will be created if needed)
            - db_uri: Default is None, which will be treated as localhost and the default
                dbserver port.
            - keep_running_on_errors: Default True. Indicates if errors shall be raised,
                or if we shall attempt to recover from issues and keep running (errors
                will be always logged to stderr)
        """
        self.keep_running_on_errors = keep_running_on_errors
        self._db_config = {'uri': db_uri, 'name': db_name}
        self.booking_delta = timedelta(seconds=booking_duration)
        self.setup_database_connection()
        self.normalizer = DictNormalizer()

    def _db_connect(self):
        # This method is here instead of inside setup_database_connection only
        # to make easier to mock MongoDB on tests
        cfg = self._db_config
        db = MongoClient(cfg['uri'])[cfg['name']]
        return db

    def setup_database_connection(self):
        self.db = self._db_connect()
        self.data = self.db[EXPERIMENTS_COLLECTION_NAME]
        self.data.ensure_index(self.marshalled_key, unique=True)

    def get_normalized_and_key(self, config):
        normalized = self.normalizer(deepcopy(config))
        serialized = json.dumps(normalized, sort_keys=True)
        return normalized, hashlib.md5(serialized.encode('utf-8')).hexdigest()

    def book_if_available(self, experiment_configuration):
        """
        Books the experiment configuration returning the booking_ticket of the
        experiment if available. None will be returned in any other case.

        If was already booked within BOOKING_DURATION, None will be returned instead,
        assuming that the experiment was booked by someone else that's running it right
        now.
        """
        ticket = None
        now = datetime.now()
        try:
            normalized_config, key = self.get_normalized_and_key(experiment_configuration)
        except self.normalizer.UnHashableDict as e:
            logger.critical(
                "Couldn't serialize experiment configuration because of %s. "
                "Complete configuration is %s." % (e, experiment_configuration)
            )
            if self.keep_running_on_errors:
                # Act as if the experiment had already been booked
                return None
            else:
                raise
        normalized_config[self.marshalled_key] = key
        normalized_config[self.experiment_status] = self.STATUS_BOOKED
        normalized_config[self.booking_at_key] = now
        try:
            ticket = self.data.insert(normalized_config)
            logger.info("Created new booking with ticket %s" % ticket)
        except DuplicateKeyError:
            # Ok, experiment is already registered. Let's see if it was already solved or
            # not. If not, and if it was booked "long time ago", we'll steal the booking
            query = {self.marshalled_key: key,
                     self.experiment_status: self.STATUS_BOOKED,
                     self.booking_at_key: {'$lte': now - self.booking_delta}
                     }
            update = {'$set': {self.booking_at_key: now}}
            experiment = self.data.find_and_modify(
                query, update=update,
                new=True  # So the modified object is returned
            )
            if experiment:
                logger.info("Stolen booking ticket %s" % key)
                ticket = experiment[u'_id']
        return ticket

    def store_results(self, booking_ticket, results):
        """
        The only way of storing experiment results is by having the "booking ticket" (ie,
        the result of a successfull booking).

        Returns True if the storage succedded, and False if not.
        Be aware that if you attempt to store results after the booking time expired,
        it's totally possible that same experiment was booked for someone else.
        """
        query = {u'_id': booking_ticket,
                 self.experiment_status: self.STATUS_BOOKED}
        update = {
            '$set': {self.experiment_status: self.STATUS_SOLVED,
                     self.results_key: mongo_dict_key_sanitizer(results)
                     },
        }
        experiment = self.data.find_and_modify(query, update)
        if experiment is None:
            logger.warning(
                "Experiment with booking_ticket %s wasn't stored, because not found on "
                "stats database as waiting-results." % booking_ticket)
            return False
        else:
            logger.info("Stored experiment results for ticket %s" % booking_ticket)
            return True

    def iter_results(self):
        return self.data.find({self.experiment_status: self.STATUS_SOLVED})
