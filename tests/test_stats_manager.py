from datetime import datetime, timedelta
import ming
import mock
import unittest

from feature_bench.stats_manager import StatsManager


class StatsManagerTests(unittest.TestCase):
    def setUp(self):
        super(StatsManagerTests, self).setUp()
        patcher = mock.patch.object(StatsManager, '_db_connect')
        db = ming.create_datastore('mim://').db
        for collection in db.collection_names():
            db[collection].remove()
        mock_get_db = patcher.start()
        mock_get_db.return_value = db
        self.addCleanup(patcher.stop)
        self.SM = StatsManager()

    def test_collection_experiments_is_created_if_not_there(self):
        self.assertIn('experiments', self.SM.db.collection_names())

    def test_simple_booking(self):
        config = {'something': 'nice'}
        ticket = self.SM.book_if_available(config)
        self.assertTrue(ticket)

    def test_after_booking_experiment_status_is_booked(self):
        config = {'something': 'nice'}
        ticket = self.SM.book_if_available(config)
        exp = self.SM.db.experiments.find_one(ticket)
        self.assertEqual(exp[self.SM.experiment_status], self.SM.STATUS_BOOKED)

    def test_booking_something_booked_fails(self):
        config = {'something': 'nice'}
        self.SM.book_if_available(config)
        ticket_2 = self.SM.book_if_available(config)
        self.assertIsNone(ticket_2)

    def test_booking_can_be_stealed_if_time_pass(self):
        config = {'something': 'nice'}
        ticket = self.SM.book_if_available(config)
        with mock.patch('feature_bench.stats_manager.datetime') as mock_datetime:
            time_limit = datetime.now() + self.SM.booking_delta
            mock_datetime.now.return_value = time_limit - timedelta(seconds=1)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            ticket_again = self.SM.book_if_available(config)
            self.assertIsNone(ticket_again)
            mock_datetime.now.return_value += timedelta(seconds=2)
            ticket_again = self.SM.book_if_available(config)
            self.assertIsNotNone(ticket_again)
            self.assertEqual(ticket, ticket_again)

    def test_simple_store_results_stores_results(self):
        # pretty obvious, ah?
        config = {'something': 'nice'}
        ticket = self.SM.book_if_available(config)
        results = {'metric_1': '1', 'metric_2': 'awesome'}
        stored = self.SM.store_results(ticket, results)
        self.assertTrue(stored)
        exp = self.SM.db.experiments.find_one(ticket)
        for k, v in results.iteritems():
            self.assertIn(k, exp['results'].keys())
            self.assertEqual(v, exp['results'][k])

    def test_after_storing_results_experiment_status_is_solved(self):
        config = {'something': 'nice'}
        ticket = self.SM.book_if_available(config)
        self.SM.store_results(ticket, {})
        exp = self.SM.db.experiments.find_one(ticket)
        self.assertEqual(exp[self.SM.experiment_status], self.SM.STATUS_SOLVED)

    def test_although_time_passed_booking_cant_be_stealed_if_was_already_solved_(self):
        config = {'something': 'nice'}
        ticket = self.SM.book_if_available(config)
        self.SM.store_results(ticket, {})
        with mock.patch('feature_bench.stats_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = (datetime.now() + self.SM.booking_delta +
                                              timedelta(seconds=999))
            ticket_again = self.SM.book_if_available(config)
            self.assertIsNone(ticket_again)

    def test_trying_to_solve_for_second_time_an_experiment_fails(self):
        config = {'something': 'nice'}
        ticket = self.SM.book_if_available(config)
        self.SM.store_results(ticket, {})  # solved with empty results
        results = {'metric_1': '1', 'metric_2': 'awesome'}
        stored = self.SM.store_results(ticket, results)
        self.assertFalse(stored)
        exp = self.SM.db.experiments.find_one(ticket)
        self.assertEqual(exp['results'], {})  # results were not updated

    def test_iter_results(self):
        self.assertEqual(len(list(self.SM.iter_results())), 0)
        ticket = self.SM.book_if_available({'something': 'nice'})
        self.assertEqual(len(list(self.SM.iter_results())), 0)
        self.SM.store_results(ticket, {})
        self.assertEqual(len(list(self.SM.iter_results())), 1)
