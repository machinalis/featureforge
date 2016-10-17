from datetime import timedelta
from unittest import TestCase
import warnings

from featureforge.experimentation.stats_manager import StatsManager

DEPRECATION_MSG = (
    'Init arguments will change position. '
    'Take a look to http://feature-forge.readthedocs.io/en/latest/experimentation.html'
    '#exploring-the-finished-experiments'
)


class TestStatsManager(TestCase):

    def setUp(self):
        self.db_name = 'a_db_name'

    def test_init_with_db_name_as_first_parameter_and_booking_duration_as_second(self):
        booking_duration = 10
        st = StatsManager(db_name=self.db_name, booking_duration=booking_duration)
        self.assertEqual(st._db_config['name'], self.db_name)
        self.assertEqual(st.booking_delta, timedelta(seconds=booking_duration))

    def test_if_init_with_db_name_as_second_argument_will_warning(self):
        booking_duration = 10
        with self.assertWarns(DeprecationWarning, msg=DEPRECATION_MSG):
            # Trigger a warning.
            StatsManager(booking_duration, self.db_name)

    def test_if_use_db_name_as_second_argument_warnings_but_can_continue(self):
        booking_duration = 10
        with self.assertWarns(DeprecationWarning, msg=DEPRECATION_MSG):
            st = StatsManager(booking_duration, self.db_name)
            self.assertEqual(st._db_config['name'], self.db_name)
            self.assertEqual(st.booking_delta, timedelta(seconds=booking_duration))
