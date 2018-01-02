#!/usr/bin/env trial
import gflags
import json
import mock
import sys
import urlparse

from ct.client import log_client
from ct.client import log_client_test_util as test_util
from ct.client.db import database
from twisted.internet import defer
from twisted.internet import task
from twisted.internet import reactor
from twisted.python import failure
from twisted.test import proto_helpers
from twisted.trial import unittest

FLAGS = gflags.FLAGS


class ResponseBodyHandlerTest(unittest.TestCase):
    def test_send(self):
        finished = defer.Deferred()
        handler = log_client.ResponseBodyHandler(finished)
        transport = proto_helpers.StringTransportWithDisconnection()
        handler.makeConnection(transport)
        transport.protocol = handler
        handler.dataReceived("test")
        transport.loseConnection()
        finished.addCallback(self.assertEqual, "test")
        return finished

    def test_send_chunks(self):
        test_msg = "x"*1024
        chunk_size = 100
        finished = defer.Deferred()
        handler = log_client.ResponseBodyHandler(finished)
        transport = proto_helpers.StringTransportWithDisconnection()
        handler.makeConnection(transport)
        transport.protocol = handler
        sent = 0
        while sent < len(test_msg):
            handler.dataReceived(test_msg[sent:sent + chunk_size])
            sent += chunk_size
        transport.loseConnection()
        finished.addCallback(self.assertEqual, test_msg)
        return finished

    def test_buffer_overflow(self):
        original = FLAGS.response_buffer_size_bytes
        FLAGS.response_buffer_size_bytes = 10
        test_msg = "x"*11
        finished = defer.Deferred()
        handler = log_client.ResponseBodyHandler(finished)
        transport = proto_helpers.StringTransportWithDisconnection()
        handler.makeConnection(transport)
        transport.protocol = handler
        handler.dataReceived(test_msg)
        transport.loseConnection()
        # TODO(ekasper): find a more elegant and robust way to save flags.
        FLAGS.response_buffer_size_bytes = original
        return self.assertFailure(finished,
                                  log_client.HTTPResponseSizeExceededError)


class AsyncLogClientTest(unittest.TestCase):
    class FakeHandler(test_util.FakeHandlerBase):

        # A class that mimics twisted.web.iweb.IResponse. Note: the IResponse
        # interface is only partially implemented.
        class FakeResponse(object):
            def __init__(self, code, reason, json_content=None):
                self.code = code
                self.phrase = reason
                self.headers = AsyncLogClientTest.FakeHandler.FakeHeader()
                if json_content is not None:
                    self._body = json.dumps(json_content)
                else:
                    self._body = ""

            def deliverBody(self, protocol):
                transport = proto_helpers.StringTransportWithDisconnection()
                protocol.makeConnection(transport)
                transport.protocol = protocol
                protocol.dataReceived(self._body)
                transport.loseConnection()

        @classmethod
        def make_response(cls, code, reason, json_content=None):
            return cls.FakeResponse(code, reason, json_content=json_content)

        class FakeHeader(object):
            def getAllRawHeaders(self):
                return []

    # Twisted doesn't (yet) have an official fake Agent:
    # https://twistedmatrix.com/trac/ticket/4024
    class FakeAgent(object):
        def __init__(self, responder):
            self._responder = responder

        def request(self, method, uri):
            if method != "GET":
                return defer.fail(failure.Failure())
            # Naive, for testing.
            path, _, params = uri.partition("?")
            params = urlparse.parse_qs(params)
            # Take the first value of each parameter.
            if any([len(params[key]) != 1 for key in params]):
                return defer.fail(failure.Failure())
            params = {key: params[key][0] for key in params}
            response = self._responder.get_response(path, params=params)
            return defer.succeed(response)

    class FakeDB(object):
        def scan_entries(self, first, last):
            raise database.KeyError("boom!")

        def store_entries(self, entries):
            self.entries = list(entries)

    def setUp(self):
        self.clock = task.Clock()

    def one_shot_client(self, json_content):
        """Make a one-shot client and give it a mock response."""
        mock_handler = mock.Mock()
        response = self.FakeHandler.make_response(200, "OK",
                                                  json_content=json_content)
        mock_handler.get_response.return_value = response
        return log_client.AsyncLogClient(self.FakeAgent(mock_handler),
                                         test_util.DEFAULT_URI,
                                         reactor=self.clock)

    def default_client(self, entries_db=None, reactor_=None):
        # A client whose responder is configured to answer queries for the
        # correct uri.
        if reactor_ is None:
            reactor_ = self.clock
        return log_client.AsyncLogClient(self.FakeAgent(
            self.FakeHandler(test_util.DEFAULT_URI)), test_util.DEFAULT_URI,
                                         entries_db=entries_db,
                                         reactor=reactor_)

    def test_get_sth(self):
        client = self.default_client()
        self.assertEqual(test_util.DEFAULT_STH,
                         self.successResultOf(client.get_sth()))

    def test_get_sth_raises_on_invalid_response(self):
        json_sth = test_util.sth_to_json(test_util.DEFAULT_STH)
        json_sth.pop("timestamp")
        client = self.one_shot_client(json_sth)
        return self.assertFailure(client.get_sth(),
                                  log_client.InvalidResponseError)

    def test_get_sth_raises_on_invalid_base64(self):
        json_sth = test_util.sth_to_json(test_util.DEFAULT_STH)
        json_sth["tree_head_signature"] = "garbagebase64^^^"
        client = self.one_shot_client(json_sth)
        return self.assertFailure(client.get_sth(),
                                  log_client.InvalidResponseError)

    class EntryConsumer(object):
        def __init__(self):
            self.received = []
            self.consumed = defer.Deferred()

        def done(self, result):
            self.result = result
            self.consumed.callback("Done")

        def consume(self, entries):
            self.received += entries
            d = defer.Deferred()
            d.callback(None)
            return d

    # Helper method.
    def get_entries(self, client, start, end, batch_size=0):
        producer = client.get_entries(start, end, batch_size=batch_size)
        consumer = self.EntryConsumer()
        d = producer.startProducing(consumer)
        d.addBoth(consumer.done)
        # Ensure the tasks scheduled in the reactor are invoked.
        # Since start of get entries is delayed, we have to pump to make up for
        # that delay. If some test is going to force get_entries to do more than
        # one fetch, then that test has to take care of additional pumping.
        self.pump_get_entries()
        return consumer

    def pump_get_entries(self,
                     delay=FLAGS.get_entries_retry_delay,
                     pumps=1):
        # Helper method which advances time past get_entries delay
        for _ in range(0, pumps):
            self.clock.pump([0, delay])

    def test_get_entries(self):
        client = self.default_client()
        consumer = self.get_entries(client, 0, 9)
        self.assertEqual(10, consumer.result)
        self.assertTrue(test_util.verify_entries(consumer.received, 0, 9))

    def test_get_sth_consistency(self):
        client = self.default_client()
        self.assertEqual([],
                         self.successResultOf(client.get_sth_consistency(0, 9)))

    def test_get_entries_raises_on_invalid_response(self):
        json_entries = test_util.entries_to_json(test_util.make_entries(0, 9))
        json_entries["entries"][5]["leaf_input"] = "garbagebase64^^^"

        client = self.one_shot_client(json_entries)
        producer = client.get_entries(0, 9)
        # remove exponential back-off
        producer._calculate_retry_delay = lambda _: 1
        consumer = self.EntryConsumer()
        d = producer.startProducing(consumer)
        d.addBoth(consumer.done)
        # pump through retries (with retries there are 2 delays per request and
        # and initial delay)
        self.pump_get_entries(1, FLAGS.get_entries_max_retries * 2 + 1)
        self.assertTrue(consumer.result.check(log_client.InvalidResponseError))
        # The entire response should be discarded upon error.
        self.assertFalse(consumer.received)

    def test_get_entries_raises_on_too_large_response(self):
        large_response = test_util.entries_to_json(
            test_util.make_entries(4, 5))

        client = self.one_shot_client(large_response)
        producer = client.get_entries(4, 4)
        # remove exponential back-off
        producer._calculate_retry_delay = lambda _: 1
        consumer = self.EntryConsumer()
        d = producer.startProducing(consumer)
        d.addBoth(consumer.done)
        # pump through retries (with retries there are 2 delays per request and
        # initial delay)
        self.pump_get_entries(1, FLAGS.get_entries_max_retries * 2 + 1)
        self.assertTrue(consumer.result.check(log_client.InvalidResponseError))

    def test_get_entries_succedes_after_retry(self):
        json_entries = test_util.entries_to_json(test_util.make_entries(0, 9))
        json_entries["entries"][5]["leaf_input"] = "garbagebase64^^^"
        client = self.one_shot_client(json_entries)
        producer = client.get_entries(0, 9)
        # remove exponential back-off
        producer._calculate_retry_delay = lambda _: 1
        consumer = self.EntryConsumer()
        d = producer.startProducing(consumer)
        d.addBoth(consumer.done)
        # pump retries halfway through (there are actually two delays before
        # firing requests, so this loop will go only through half of retries)
        self.pump_get_entries(1, FLAGS.get_entries_max_retries)
        self.assertFalse(hasattr(consumer, 'result'))
        json_entries = test_util.entries_to_json(test_util.make_entries(0, 9))
        response = self.FakeHandler.make_response(200, "OK",
                                                  json_content=json_entries)
        client._handler._agent._responder.get_response.return_value = response
        self.pump_get_entries(1)
        self.assertTrue(test_util.verify_entries(consumer.received, 0, 9))

    def test_get_entries_raises_if_query_is_larger_than_tree_size(self):
        client = log_client.AsyncLogClient(
            self.FakeAgent(self.FakeHandler(
                test_util.DEFAULT_URI, tree_size=3)), test_util.DEFAULT_URI,
            reactor=self.clock)
        consumer = self.get_entries(client, 0, 9)
        # also pump error
        self.pump_get_entries()
        self.assertTrue(consumer.result.check(log_client.HTTPClientError))

    def test_get_entries_returns_all_in_batches(self):
        mock_handler = mock.Mock()
        fake_responder = self.FakeHandler(test_util.DEFAULT_URI)
        mock_handler.get_response.side_effect = (
            fake_responder.get_response)

        client = log_client.AsyncLogClient(self.FakeAgent(mock_handler),
                                           test_util.DEFAULT_URI,
                                           reactor=self.clock)
        consumer = self.get_entries(client, 0, 9, batch_size=4)
        self.assertEqual(10, consumer.result)
        self.assertTrue(test_util.verify_entries(consumer.received, 0, 9))
        self.assertEqual(3, len(mock_handler.get_response.call_args_list))

    def test_get_entries_returns_all_for_limiting_server(self):
        client = log_client.AsyncLogClient(
            self.FakeAgent(
                self.FakeHandler(test_util.DEFAULT_URI, entry_limit=3)),
            test_util.DEFAULT_URI, reactor=self.clock)
        consumer = self.get_entries(client, 0, 9)
        # 1 pump in get_entries and 3 more so we fetch everything
        self.pump_get_entries(pumps=3)
        self.assertTrue(test_util.verify_entries(consumer.received, 0, 9))

    class PausingConsumer(object):
        def __init__(self, pause_at):
            self.received = []
            self.pause_at = pause_at
            self.already_paused = False
            self.result = None

        def registerProducer(self, producer):
            self.producer = producer

        def done(self, result):
            self.result = result

        def consume(self, entries):
            self.received += entries
            if (not self.already_paused and
                len(self.received) >= self.pause_at):
                self.producer.pauseProducing()
                self.already_paused = True
            d = defer.Deferred()
            d.callback(None)
            return d

    def test_get_entries_pause_resume(self):
        client = self.default_client()
        producer = client.get_entries(0, 9, batch_size=4)
        consumer = self.PausingConsumer(4)
        consumer.registerProducer(producer)
        d = producer.startProducing(consumer)
        d.addBoth(consumer.done)
        # fire all pending callbacks, and then fire request
        self.pump_get_entries()
        self.assertTrue(test_util.verify_entries(consumer.received, 0, 3))
        self.assertEqual(4, len(consumer.received))
        self.assertIsNone(consumer.result)
        producer.resumeProducing()
        # pump next 2 batches
        self.pump_get_entries(pumps=2)
        self.assertEqual(10, consumer.result)
        self.assertTrue(test_util.verify_entries(consumer.received, 0, 9))

    def test_get_entries_use_stored_entries(self):
        fake_db = self.FakeDB()
        # if client tries to fetch entries instead of taking them from db, then
        # he will get 0 - 9 entries. If he uses db then he will get 10 - 19
        fake_db.scan_entries = mock.Mock(
                return_value=test_util.make_entries(10, 19))
        client = self.default_client(entries_db=fake_db, reactor_=reactor)
        consumer = self.get_entries(client, 0, 9)
        consumer.consumed.addCallback(lambda _:
                                  self.assertEqual(len(consumer.received), 10))
        consumer.consumed.addCallback(lambda _:
            [self.assertEqual(test_util.make_entry(i + 10), consumer.received[i])
             for i in range(0, 9)])

    def test_get_entries_tries_to_fetch_if_not_available_in_db(self):
        fake_db = self.FakeDB()
        fake_db.scan_entries = mock.Mock(return_value=None)
        client = self.default_client(entries_db=fake_db)
        consumer = self.get_entries(client, 0, 9)
        test_util.verify_entries(consumer.received, 0, 9)

    def test_get_entries_stores_entries(self):
        fake_db = self.FakeDB()
        client = self.default_client(entries_db=fake_db, reactor_=reactor)
        consumer = self.get_entries(client, 0, 9)
        consumer.consumed.addCallback(lambda _:
                            test_util.verify_entries(consumer.received, 0, 9))
        consumer.consumed.addCallback(lambda _:
                            test_util.verify_entries(fake_db.entries, 0, 9))
        return consumer.consumed

    class BadEntryConsumer(EntryConsumer):
        def consume(self, entries):
            self.received += entries
            d = defer.Deferred()
            d.errback(ValueError("Boom!"))
            return d

    def test_get_entries_fires_done_if_consumer_raises(self):
        client = self.default_client()
        producer = client.get_entries(0, 9)
        consumer = self.BadEntryConsumer()
        d = producer.startProducing(consumer)
        d.addBoth(consumer.done)
        self.pump_get_entries()
        self.assertTrue(consumer.result.check(ValueError))

if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
