#!/usr/bin/env trial
import copy
import difflib
import gflags
import logging
import mock
import os
import sys

from ct.client import log_client
from ct.client.db import sqlite_connection as sqlitecon
from ct.client.db import sqlite_log_db
from ct.client import state
from ct.client import monitor
from ct.crypto import error
from ct.crypto import merkle
from ct.proto import client_pb2
from twisted.internet import defer
from twisted.trial import unittest
from twisted.web import iweb
from zope.interface import implements

FLAGS = gflags.FLAGS

#TODO(ekasper) to make this setup common to all tests
gflags.DEFINE_bool("verbose_tests", False, "Print test logs")


def dummy_compute_projected_sth(old_sth):
    sth = client_pb2.SthResponse()
    sth.timestamp = old_sth.timestamp
    sth.tree_size = size = old_sth.tree_size
    tree = merkle.CompactMerkleTree(
        merkle.TreeHasher(), size, ["a"] * merkle.count_bits_set(size))
    f = mock.Mock(return_value=(sth, tree))
    f.dummy_sth = sth
    f.dummy_tree = tree
    old_sth.sha256_root_hash = tree.root_hash()
    return f

# TODO(robpercival): This is a relatively complicated fake, and may hide subtle
# bugs in how the Monitor interacts with the real EntryProducer. Using the real
# EntryProducer with a FakeAgent, as async_log_client_test does, may be an
# improvement.
class FakeEntryProducer(object):
    def __init__(self, start, end, batch_size=None, throw=None):
        self._start = start
        self._end = end
        self._real_start = start
        self._real_end = end
        self.throw = throw
        self.batch_size = batch_size if batch_size else end - start + 1
        self.stop = False

    @defer.deferredGenerator
    def produce(self):
        if self.throw:
            raise self.throw
        for i in range(self._start, self._end, self.batch_size):
            entries = []
            for j in range(i, min(i + self.batch_size, self._end)):
                entry = client_pb2.EntryResponse()
                entry.leaf_input = "leaf_input-%d" % j
                entry.extra_data = "extra_data-%d" % j
                entries.append(entry)
            d = self.consumer.consume(entries)
            wfd = defer.waitForDeferred(d)
            yield wfd
            wfd.getResult()
            if self.stop:
                break

        if not self.stop:
            self.done.callback(self._end - self._start + 1)

    def startProducing(self, consumer):
        self.stop = False
        self._start = self._real_start
        self._end = self._real_end
        self.consumer = consumer
        self.done = defer.Deferred()
        d = self.produce()
        d.addErrback(self.stopProducing)
        return self.done

    def change_range_after_start(self, start, end):
        """Changes query interval exactly when startProducing is ran.

        EntryConsumer in Monitor uses Producer interval, so in one of the tests
        we have to be able to change that interval when producing is started,
        but after consumer is created."""
        self._real_start = start
        self._real_end = end

    def stopProducing(self, failure=None):
        self.stop = True
        if failure:
            self.done.errback(failure)


class FakeLogClient(object):
    def __init__(self, sth, servername="log_server", batch_size=None,
                 get_entries_throw=None):
        self.servername = servername
        self.sth = sth
        self.batch_size = batch_size
        self.get_entries_throw = get_entries_throw

    def get_sth(self):
        d = defer.Deferred()
        d.callback(self.sth)
        return d

    def get_entries(self, start, end):
        return FakeEntryProducer(start, end, self.batch_size,
                                 self.get_entries_throw)

    def get_sth_consistency(self, old_tree, new_tree):
        d = defer.Deferred()
        d.callback([])
        return d

class InMemoryStateKeeper(object):
    def __init__(self, state=None):
        self.state = state
    def write(self, state):
        self.state = state
    def read(self, state_type):
        if not self.state:
            raise state.FileNotFoundError("Boom!")
        return_state = state_type()
        return_state.CopyFrom(self.state)
        return return_state

class MonitorTest(unittest.TestCase):
    _DEFAULT_STH = client_pb2.SthResponse()
    _DEFAULT_STH.timestamp = 2000
    _DEFAULT_STH.tree_size = 10
    _DEFAULT_STH.tree_head_signature = "sig"
    _DEFAULT_STH_compute_projected = dummy_compute_projected_sth(_DEFAULT_STH)

    _NEW_STH = client_pb2.SthResponse()
    _NEW_STH.timestamp = 3000
    _NEW_STH.tree_size = _DEFAULT_STH.tree_size + 10
    _NEW_STH.tree_head_signature = "sig2"
    _NEW_STH_compute_projected = dummy_compute_projected_sth(_NEW_STH)

    _DEFAULT_STATE = client_pb2.MonitorState()
    _DEFAULT_STATE.verified_sth.CopyFrom(_DEFAULT_STH)
    _DEFAULT_STH_compute_projected.dummy_tree.save(
            _DEFAULT_STATE.unverified_tree)
    _DEFAULT_STH_compute_projected.dummy_tree.save(
            _DEFAULT_STATE.verified_tree)

    def setUp(self):
        if not FLAGS.verbose_tests:
          logging.disable(logging.CRITICAL)
        self.db = sqlite_log_db.SQLiteLogDB(
            sqlitecon.SQLiteConnectionManager(":memory:", keepalive=True))
        # We can't simply use DB in memory with keepalive True, because different
        # thread is writing to the database which results in an sqlite exception.
        self.cert_db = mock.MagicMock()

        self.state_keeper = InMemoryStateKeeper(copy.deepcopy(self._DEFAULT_STATE))
        self.verifier = mock.Mock()
        self.hasher = merkle.TreeHasher()

        # Make sure the DB knows about the default log server.
        log = client_pb2.CtLogMetadata()
        log.log_server = "log_server"
        self.db.add_log(log)

    def verify_state(self, expected_state):
        if self.state_keeper.state != expected_state:
            state_diff = difflib.unified_diff(
                    str(expected_state).splitlines(),
                    str(self.state_keeper.state).splitlines(),
                    fromfile="expected", tofile="actual", lineterm="", n=5)

            raise unittest.FailTest("State is incorrect\n" +
                                    "\n".join(state_diff))

    def verify_tmp_data(self, start, end):
        # TODO: we are no longer using the temp db
        # all the callsites should be updated to test the main db instead
        pass

    def create_monitor(self, client, skip_scan_entry=True):
        m = monitor.Monitor(client, self.verifier, self.hasher, self.db,
                            self.cert_db, 7, self.state_keeper)
        if m:
            m._scan_entries = mock.Mock()
        return m

    def check_db_state_after_successful_updates(self, number_of_updates):
        audited_sths = list(self.db.scan_latest_sth_range("log_server"))
        for index, audited_sth in enumerate(audited_sths):
            if index % 2 != 0:
                self.assertEqual(client_pb2.UNVERIFIED,
                                 audited_sth.audit.status)
            else:
                self.assertEqual(client_pb2.VERIFIED,
                                 audited_sth.audit.status)
        self.assertEqual(len(audited_sths), number_of_updates * 2)

    def test_update(self):
        client = FakeLogClient(self._NEW_STH)

        m = self.create_monitor(client)
        m._compute_projected_sth_from_tree = self._NEW_STH_compute_projected
        def check_state(result):
            # Check that we wrote the state...
            expected_state = client_pb2.MonitorState()
            expected_state.verified_sth.CopyFrom(self._NEW_STH)
            m._compute_projected_sth_from_tree.dummy_tree.save(
                    expected_state.verified_tree)
            m._compute_projected_sth_from_tree.dummy_tree.save(
                    expected_state.unverified_tree)
            self.verify_state(expected_state)

            self.verify_tmp_data(self._DEFAULT_STH.tree_size,
                                 self._NEW_STH.tree_size-1)
            self.check_db_state_after_successful_updates(1)
            for audited_sth in self.db.scan_latest_sth_range(m.servername):
                self.assertEqual(self._NEW_STH, audited_sth.sth)

        return m.update().addCallback(self.assertTrue).addCallback(check_state)

    def test_first_update(self):
        client = FakeLogClient(self._DEFAULT_STH)

        self.state_keeper.state = None
        m = self.create_monitor(client)
        m._compute_projected_sth_from_tree = self._DEFAULT_STH_compute_projected
        def check_state(result):
            # Check that we wrote the state...
            self.verify_state(self._DEFAULT_STATE)

            self.verify_tmp_data(0, self._DEFAULT_STH.tree_size-1)
            self.check_db_state_after_successful_updates(1)
            for audited_sth in self.db.scan_latest_sth_range(m.servername):
                self.assertEqual(self._DEFAULT_STH, audited_sth.sth)

        d = m.update().addCallback(self.assertTrue
                                   ).addCallback(check_state)
        return d

    def test_update_no_new_entries(self):
        client = FakeLogClient(self._DEFAULT_STH)

        m = self.create_monitor(client)
        d = m.update()
        d.addCallback(self.assertTrue)

        def check_state(result):
            # Check that we kept the state...
            self.verify_state(self._DEFAULT_STATE)

            # ...and wrote no entries.
            self.check_db_state_after_successful_updates(0)
        d.addCallback(check_state)
        return d

    def test_update_recovery(self):
        client = FakeLogClient(self._NEW_STH)

        # Setup initial state to be as though an update had failed part way
        # through.
        initial_state = copy.deepcopy(self._DEFAULT_STATE)
        initial_state.pending_sth.CopyFrom(self._NEW_STH)
        self._NEW_STH_compute_projected.dummy_tree.save(
                initial_state.unverified_tree)
        self.state_keeper.write(initial_state)

        m = self.create_monitor(client)
        m._compute_projected_sth_from_tree = self._NEW_STH_compute_projected

        d = m.update()
        d.addCallback(self.assertTrue)

        def check_state(result):
            # Check that we wrote the state...
            expected_state = copy.deepcopy(initial_state)
            expected_state.ClearField("pending_sth")
            expected_state.verified_sth.CopyFrom(self._NEW_STH)
            m._compute_projected_sth_from_tree.dummy_tree.save(
                    expected_state.verified_tree)
            m._compute_projected_sth_from_tree.dummy_tree.save(
                    expected_state.unverified_tree)
            self.verify_state(expected_state)

            self.check_db_state_after_successful_updates(1)
            for audited_sth in self.db.scan_latest_sth_range(m.servername):
                self.assertEqual(self._NEW_STH, audited_sth.sth)
        d.addCallback(check_state)
        return d

    def test_update_rolls_back_unverified_tree_on_scan_error(self):
        client = FakeLogClient(self._NEW_STH)

        m = self.create_monitor(client)
        m._compute_projected_sth_from_tree = self._NEW_STH_compute_projected
        m._scan_entries = mock.Mock(side_effect=ValueError("Boom!"))

        def check_state(result):
            # The changes to the unverified tree should have been discarded,
            # so that entries are re-fetched and re-consumed next time.
            expected_state = copy.deepcopy(self._DEFAULT_STATE)
            expected_state.pending_sth.CopyFrom(self._NEW_STH)
            self.verify_state(expected_state)
            # The new STH should have been verified prior to the error.
            audited_sths = list(self.db.scan_latest_sth_range(m.servername))
            self.assertEqual(len(audited_sths), 2)
            self.assertEqual(audited_sths[0].audit.status, client_pb2.VERIFIED)
            self.assertEqual(audited_sths[1].audit.status, client_pb2.UNVERIFIED)

        return m.update().addCallback(self.assertFalse).addCallback(check_state)

    def test_update_call_sequence(self):
        # Test that update calls update_sth and update_entries in sequence,
        # and bails on first error, so we can test each of them separately.
        # Each of these functions checks if functions were properly called
        # and runs step in sequence of updates.
        def check_calls_sth_fails(result):
            m._update_sth.assert_called_once_with()
            m._update_entries.assert_called_once_with()

            m._update_sth.reset_mock()
            m._update_entries.reset_mock()
            m._update_sth.return_value = copy.deepcopy(d_false)
            return m.update().addCallback(self.assertFalse)

        def check_calls_entries_fail(result):
            m._update_sth.assert_called_once_with()
            self.assertFalse(m._update_entries.called)

            m._update_sth.reset_mock()
            m._update_entries.reset_mock()
            m._update_sth.return_value = copy.deepcopy(d_true)
            m._update_entries.return_value = copy.deepcopy(d_false)
            return m.update().addCallback(self.assertFalse)

        def check_calls_assert_last_calls(result):
            m._update_sth.assert_called_once_with()
            m._update_entries.assert_called_once_with()

        client = FakeLogClient(self._DEFAULT_STH)

        m = self.create_monitor(client)
        d_true = defer.Deferred()
        d_true.callback(True)
        d_false = defer.Deferred()
        d_false.callback(False)
        #check regular correct update
        m._update_sth = mock.Mock(return_value=copy.deepcopy(d_true))
        m._update_entries = mock.Mock(return_value=copy.deepcopy(d_true))
        d = m.update().addCallback(self.assertTrue)
        d.addCallback(check_calls_sth_fails)
        d.addCallback(check_calls_entries_fail)
        d.addCallback(check_calls_assert_last_calls)
        return d

    def test_update_sth(self):
        client = FakeLogClient(self._NEW_STH)

        m = self.create_monitor(client)

        def check_state(result):
            # Check that we updated the state.
            expected_state = copy.deepcopy(self._DEFAULT_STATE)
            expected_state.pending_sth.CopyFrom(self._NEW_STH)
            self.verify_state(expected_state)
            audited_sths = list(self.db.scan_latest_sth_range(m.servername))
            self.assertEqual(len(audited_sths), 2)
            self.assertEqual(audited_sths[0].audit.status, client_pb2.VERIFIED)
            self.assertEqual(audited_sths[1].audit.status, client_pb2.UNVERIFIED)

        return m._update_sth().addCallback(self.assertTrue
                                           ).addCallback(check_state)

    def test_update_sth_fails_for_invalid_sth(self):
        client = FakeLogClient(self._NEW_STH)
        self.verifier.verify_sth.side_effect = error.VerifyError("Boom!")

        m = self.create_monitor(client)
        def check_state(result):
            # Check that we kept the state.
            self.verify_state(self._DEFAULT_STATE)
            self.check_db_state_after_successful_updates(0)

        return m._update_sth().addCallback(self.assertFalse
                                           ).addCallback(check_state)

    def test_update_sth_fails_for_stale_sth(self):
        sth = client_pb2.SthResponse()
        sth.CopyFrom(self._DEFAULT_STH)
        sth.tree_size -= 1
        sth.timestamp -= 1
        client = FakeLogClient(sth)

        m = self.create_monitor(client)
        d = defer.Deferred()
        d.callback(True)
        m._verify_consistency = mock.Mock(return_value=d)
        def check_state(result):
            self.assertTrue(m._verify_consistency.called)
            args, _ = m._verify_consistency.call_args
            self.assertTrue(args[0].timestamp < args[1].timestamp)

            # Check that we kept the state.
            self.verify_state(self._DEFAULT_STATE)

        return m._update_sth().addCallback(self.assertFalse
                                           ).addCallback(check_state)

    def test_update_sth_fails_for_inconsistent_sth(self):
        client = FakeLogClient(self._NEW_STH)
        # The STH is in fact OK but fake failure.
        self.verifier.verify_sth_consistency.side_effect = (
            error.ConsistencyError("Boom!"))

        m = self.create_monitor(client)
        def check_state(result):
            # Check that we kept the state.
            self.verify_state(self._DEFAULT_STATE)
            audited_sths = list(self.db.scan_latest_sth_range(m.servername))
            self.assertEqual(len(audited_sths), 2)
            self.assertEqual(audited_sths[0].audit.status,
                             client_pb2.VERIFY_ERROR)
            self.assertEqual(audited_sths[1].audit.status,
                             client_pb2.UNVERIFIED)
            for audited_sth in audited_sths:
                self.assertEqual(self._DEFAULT_STH.sha256_root_hash,
                                 audited_sth.sth.sha256_root_hash)

        return m._update_sth().addCallback(self.assertFalse
                                           ).addCallback(check_state)

    def test_update_sth_fails_on_client_error(self):
        client = FakeLogClient(self._NEW_STH)
        def get_sth():
            return defer.maybeDeferred(mock.Mock(side_effect=log_client.HTTPError("Boom!")))
        client.get_sth = get_sth
        m = self.create_monitor(client)
        def check_state(result):
            # Check that we kept the state.
            self.verify_state(self._DEFAULT_STATE)
            self.check_db_state_after_successful_updates(0)

        return m._update_sth().addCallback(self.assertFalse
                                           ).addCallback(check_state)


    def test_update_entries_fails_on_client_error(self):
        client = FakeLogClient(self._NEW_STH,
                               get_entries_throw=log_client.HTTPError("Boom!"))
        client.get_entries = mock.Mock(
                return_value=client.get_entries(0, self._NEW_STH.tree_size - 2))

        m = self.create_monitor(client)

        # Get the new STH, then try (and fail) to update entries
        d = m._update_sth().addCallback(self.assertTrue)
        d.addCallback(lambda x: m._update_entries()).addCallback(self.assertFalse)

        def check_state(result):
            # Check that we wrote no entries.
            expected_state = copy.deepcopy(self._DEFAULT_STATE)
            expected_state.pending_sth.CopyFrom(self._NEW_STH)
            self.verify_state(expected_state)
        d.addCallback(check_state)

        return d

    def test_update_entries_fails_not_enough_entries(self):
        client = FakeLogClient(self._NEW_STH)
        faker_fake_entry_producer = FakeEntryProducer(0,
                                                      self._NEW_STH.tree_size)
        faker_fake_entry_producer.change_range_after_start(0, 5)
        client.get_entries = mock.Mock(
                return_value=faker_fake_entry_producer)

        m = self.create_monitor(client)
        m._compute_projected_sth = self._NEW_STH_compute_projected
        # Get the new STH first.
        return m._update_sth().addCallback(self.assertTrue).addCallback(
                lambda x: m._update_entries().addCallback(self.assertFalse))

    def test_update_entries_fails_in_the_middle(self):
        client = FakeLogClient(self._NEW_STH)
        faker_fake_entry_producer = FakeEntryProducer(
                self._DEFAULT_STH.tree_size,
                self._NEW_STH.tree_size)
        faker_fake_entry_producer.change_range_after_start(
            self._DEFAULT_STH.tree_size, self._NEW_STH.tree_size - 5)
        client.get_entries = mock.Mock(return_value=faker_fake_entry_producer)

        m = self.create_monitor(client)
        m._compute_projected_sth = self._NEW_STH_compute_projected
        fake_fetch = mock.MagicMock()
        def try_again_with_all_entries(_):
            m._fetch_entries = fake_fetch
            return m._update_entries()
        # Get the new STH first.
        return m._update_sth().addCallback(self.assertTrue).addCallback(
                lambda _: m._update_entries().addCallback(self.assertFalse)
                ).addCallback(try_again_with_all_entries).addCallback(lambda _:
                    fake_fetch.assert_called_once_with(15, 19))

if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
