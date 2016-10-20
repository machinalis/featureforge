from datetime import timedelta
import mock
from unittest import TestCase
import warnings

from featureforge.experimentation.stats_manager import StatsManager

DEPRECATION_MSG = (
    'Init arguments will change. '
    'Take a look to http://feature-forge.readthedocs.io/en/latest/experimentation.html'
    '#exploring-the-finished-experiments'
)

DB_CONNECTION_PATH = 'featureforge.experimentation.stats_manager.StatsManager.setup_database_connection'   # NOQA


class TestStatsManager(TestCase):

    def setUp(self):
        self.db_name = 'a_db_name'
        self.booking_duration = 10

    def test_init_with_db_name_as_first_parameter_and_booking_duration_as_second(self):
        with mock.patch(DB_CONNECTION_PATH):
            st = StatsManager(db_name=self.db_name, booking_duration=self.booking_duration)
            self.assertEqual(st._db_config['name'], self.db_name)
            self.assertEqual(st.booking_delta, timedelta(seconds=self.booking_duration))

    def test_if_init_with_db_name_as_second_argument_will_warning(self):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always", DeprecationWarning)
            # Trigger a warning.
            with mock.patch(DB_CONNECTION_PATH):
                StatsManager(self.booking_duration, self.db_name)
                # Verify some things
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
                self.assertEqual(str(w[-1].message), DEPRECATION_MSG)

    def test_if_use_db_name_as_second_argument_warnings_but_can_continue(self):
        with warnings.catch_warnings(record=True):
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always", DeprecationWarning)
            # Trigger a warning.
            with mock.patch(DB_CONNECTION_PATH):
                st = StatsManager(self.booking_duration, self.db_name)
                self.assertEqual(st._db_config['name'], self.db_name)
                self.assertEqual(st.booking_delta, timedelta(seconds=self.booking_duration))
