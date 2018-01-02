#!/usr/bin/env python

import unittest

import base64
import json
import mock
import requests
import sys

from ct.client import log_client
from ct.client import log_client_test_util as test_util
from ct.crypto import merkle
from ct.proto import client_pb2
import gflags

FLAGS = gflags.FLAGS


class LogClientTest(unittest.TestCase):
    class FakeHandler(test_util.FakeHandlerBase):

        # A class that mimics requests.models.Response
        class FakeResponse(object):
            def __init__(self, code, reason, json_content=None):
                self.status_code = code
                self.reason = reason
                self.headers = ''
                if json_content is not None:
                    self.content = json.dumps(json_content)
                else:
                    self.content = ""

        @classmethod
        def make_response(cls, code, reason, json_content=None):
            return cls.FakeResponse(code, reason, json_content=json_content)

    @staticmethod
    def one_shot_client(json_content):
        """Make a one-shot client and give it a mock response."""
        mock_handler = mock.Mock()
        mock_handler.get_response_body.return_value = json.dumps(json_content)
        return log_client.LogClient("some address", handler=mock_handler)

    def default_client(self):
        # A client whose responder is configured to answer queries for the
        # correct uri.
        return log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI))

    def test_get_sth(self):
        client = self.default_client()
        sth_response = client.get_sth()

        self.assertEqual(sth_response.timestamp,
                         test_util.DEFAULT_STH.timestamp)
        self.assertEqual(sth_response.tree_size,
                         test_util.DEFAULT_STH.tree_size)
        self.assertEqual(sth_response.sha256_root_hash,
                         test_util.DEFAULT_STH.sha256_root_hash)
        self.assertEqual(sth_response.tree_head_signature,
                         test_util.DEFAULT_STH.tree_head_signature)

    def test_get_sth_raises_on_invalid_response(self):
        json_sth = test_util.sth_to_json(test_util.DEFAULT_STH)
        json_sth.pop("timestamp")
        client = self.one_shot_client(json_sth)
        self.assertRaises(log_client.InvalidResponseError, client.get_sth)

    def test_get_sth_raises_on_invalid_base64(self):
        json_sth = test_util.sth_to_json(test_util.DEFAULT_STH)
        json_sth["tree_head_signature"] = "garbagebase64^^^"
        client = self.one_shot_client(json_sth)
        self.assertRaises(log_client.InvalidResponseError, client.get_sth)

    def test_get_entries(self):
        client = self.default_client()
        returned_entries = list(client.get_entries(0, 9))
        self.assertTrue(test_util.verify_entries(returned_entries, 0, 9))

    def test_get_entries_raises_on_invalid_response(self):
        json_entries = test_util.entries_to_json(test_util.make_entries(4, 4))
        json_entries["entries"][0].pop("leaf_input")

        client = self.one_shot_client(json_entries)
        entries = client.get_entries(4, 4)
        self.assertRaises(log_client.InvalidResponseError,
                          entries.next)

    def test_get_entries_raises_immediately_on_invalid_base64(self):
        json_entries = test_util.entries_to_json(test_util.make_entries(3, 4))
        json_entries["entries"][1]["leaf_input"] = "garbagebase64^^^"

        client = self.one_shot_client(json_entries)
        entries = client.get_entries(3, 4)
        # We shouldn't see anything, even if the first entry appeared valid.
        self.assertRaises(log_client.InvalidResponseError,
                          entries.next)

    def test_get_entries_raises_on_empty_response(self):
        empty_entries = test_util.entries_to_json([])
        client = self.one_shot_client(empty_entries)

        entries = client.get_entries(4, 4)
        self.assertRaises(log_client.InvalidResponseError,
                          entries.next)

    def test_get_entries_raises_on_too_large_response(self):
        large_response = test_util.entries_to_json(
            test_util.make_entries(4, 5))

        client = self.one_shot_client(large_response)
        entries = client.get_entries(4, 4)
        self.assertRaises(log_client.InvalidResponseError,
                          entries.next)

    def test_get_entries_returns_all_in_batches(self):
        mock_handler = mock.Mock()
        fake_responder = self.FakeHandler(test_util.DEFAULT_URI)
        mock_handler.get_response_body.side_effect = (
            fake_responder.get_response_body)

        client = log_client.LogClient(test_util.DEFAULT_URI,
                                      handler=mock_handler)
        returned_entries = list(client.get_entries(0, 9, batch_size=4))
        self.assertTrue(test_util.verify_entries(returned_entries, 0, 9))
        self.assertEqual(3, len(mock_handler.get_response_body.call_args_list))

        # Same as above, but using a flag to control the batch size.
        mock_handler.reset_mock()
        # TODO(ekasper): find a more elegant and robust way to save flags.
        original = FLAGS.entry_fetch_batch_size
        FLAGS.entry_fetch_batch_size = 4
        returned_entries = list(client.get_entries(0, 9))
        FLAGS.entry_fetch_batch_size = original
        self.assertTrue(test_util.verify_entries(returned_entries, 0, 9))
        self.assertEqual(3, len(mock_handler.get_response_body.call_args_list))

    def test_get_entries_returns_all_for_limiting_server(self):
        client = log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI, entry_limit=3))
        returned_entries = list(client.get_entries(0, 9))
        self.assertTrue(test_util.verify_entries(returned_entries, 0, 9))

    def test_get_entries_returns_partial_if_log_returns_partial(self):
        client = log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI, tree_size=3))
        entries = client.get_entries(0, 9)
        partial = []
        for _ in range(3):
            partial.append(entries.next())
        self.assertTrue(test_util.verify_entries(partial, 0, 2))
        self.assertRaises(log_client.HTTPClientError, entries.next)

    def test_get_sth_consistency(self):
        client = log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI, tree_size=3))
        proof = client.get_sth_consistency(1, 2)
        self.assertEqual(proof, test_util.DEFAULT_FAKE_PROOF)

    def test_get_sth_consistency_trivial(self):
        client = log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI, tree_size=3))
        self.assertEqual(client.get_sth_consistency(0, 0), [])
        self.assertEqual(client.get_sth_consistency(0, 2), [])
        self.assertEqual(client.get_sth_consistency(2, 2), [])

    def test_get_sth_consistency_raises_on_invalid_input(self):
        client = log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI, tree_size=3))
        self.assertRaises(log_client.InvalidRequestError,
                          client.get_sth_consistency, -1, 1)
        self.assertRaises(log_client.InvalidRequestError,
                          client.get_sth_consistency, -3, -1)
        self.assertRaises(log_client.InvalidRequestError,
                          client.get_sth_consistency, 3, 1)

    def test_get_sth_consistency_raises_on_client_error(self):
        client = log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI, tree_size=3))
        self.assertRaises(log_client.HTTPClientError,
                          client.get_sth_consistency, 1, 5)

    def test_get_sth_consistency_raises_on_invalid_response(self):
        client = self.one_shot_client({})
        self.assertRaises(log_client.InvalidResponseError,
                          client.get_sth_consistency, 1, 2)

    def test_get_sth_consistency_raises_on_invalid_base64(self):
        json_proof = {"consistency": ["garbagebase64^^^"]}
        client = self.one_shot_client(json_proof)
        self.assertRaises(log_client.InvalidResponseError,
                          client.get_sth_consistency, 1, 2)

    def test_get_roots(self):
        client = self.default_client()
        roots = client.get_roots()
        self.assertEqual(roots, test_util.DEFAULT_FAKE_ROOTS)

    def test_get_roots_raises_on_invalid_response(self):
        client = self.one_shot_client({})
        self.assertRaises(log_client.InvalidResponseError, client.get_roots)

    def test_get_roots_raises_on_invalid_base64(self):
        json_roots = {"certificates": ["garbagebase64^^^"]}
        client = self.one_shot_client(json_roots)
        self.assertRaises(log_client.InvalidResponseError, client.get_roots)

    def test_get_entry_and_proof(self):
        client = self.default_client()
        entry_and_proof = client.get_entry_and_proof(1, 2)
        self.assertEqual(entry_and_proof.entry, test_util.make_entry(1))
        self.assertEqual(entry_and_proof.audit_path,
                         test_util.DEFAULT_FAKE_PROOF)

    def test_get_entry_and_proof_raises_on_invalid_input(self):
        client = self.default_client()
        self.assertRaises(log_client.InvalidRequestError,
                          client.get_entry_and_proof, -1, 1)
        self.assertRaises(log_client.InvalidRequestError,
                          client.get_entry_and_proof, -3, -1)
        self.assertRaises(log_client.InvalidRequestError,
                          client.get_entry_and_proof, 3, 1)

    def test_get_entry_and_proof_raises_on_client_error(self):
        client = log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI, tree_size=3))
        self.assertRaises(log_client.HTTPClientError,
                          client.get_entry_and_proof, 1, 5)

    def test_get_entry_and_proof_raises_on_invalid_response(self):
        json_response = test_util.entry_and_proof_to_json(
            test_util.make_entry(1), test_util.DEFAULT_FAKE_PROOF)
        json_response.pop("leaf_input")
        client = self.one_shot_client(json_response)
        self.assertRaises(log_client.InvalidResponseError,
                          client.get_entry_and_proof, 1, 2)

    def test_get_entry_and_proof_raises_on_invalid_base64(self):
        json_response = test_util.entry_and_proof_to_json(
            test_util.make_entry(1), test_util.DEFAULT_FAKE_PROOF)
        json_response["leaf_input"] = ["garbagebase64^^^"]
        client = self.one_shot_client(json_response)
        self.assertRaises(log_client.InvalidResponseError,
                          client.get_entry_and_proof, 1, 2)

    def test_get_proof_by_hash(self):
        client = self.default_client()
        entry = test_util.make_entry(1)
        hasher = merkle.TreeHasher()
        leaf_hash = hasher.hash_leaf(entry.leaf_input)

        proof_by_hash = client.get_proof_by_hash(leaf_hash, 2)
        self.assertEqual(proof_by_hash.audit_path, test_util.DEFAULT_FAKE_PROOF)
        self.assertEqual(proof_by_hash.leaf_index, 1)

    def test_get_proof_by_hash_raises_on_invalid_input(self):
        client = self.default_client()
        leaf_hash = "hash"
        self.assertRaises(log_client.InvalidRequestError,
                          client.get_proof_by_hash, leaf_hash, 0)
        self.assertRaises(log_client.InvalidRequestError,
                          client.get_proof_by_hash, leaf_hash, -1)

    def test_get_proof_by_hash_raises_on_unknown_hash(self):
        client = log_client.LogClient(test_util.DEFAULT_URI, self.FakeHandler(
            test_util.DEFAULT_URI, tree_size=3))
        leaf_hash = "bogus"
        self.assertRaises(log_client.HTTPClientError,
                          client.get_proof_by_hash, leaf_hash, 2)

    def test_get_proof_by_hash_raises_on_invalid_response(self):
        json_response = test_util.proof_and_index_to_json(
            test_util.DEFAULT_FAKE_PROOF, 1)
        json_response.pop("leaf_index")
        client = self.one_shot_client(json_response)
        self.assertRaises(log_client.InvalidResponseError,
                          client.get_proof_by_hash, "hash", 2)

    def test_get_proof_by_hash_raises_on_invalid_base64(self):
        json_response = test_util.proof_and_index_to_json(
            test_util.DEFAULT_FAKE_PROOF, 1)
        json_response["leaf_index"] = "garbagebase64^^^"
        client = self.one_shot_client(json_response)
        self.assertRaises(log_client.InvalidResponseError,
                          client.get_proof_by_hash, "hash", 2)

    def _verify_sct_contents(self, sct):
        LOG_ID = base64.b64decode(
            'pLkJkLQYWBSHuxOizGdwCjw1mAT5G9+443fNDsgN3BA=')
        self.assertEqual(client_pb2.V1, sct.version)
        self.assertEqual(LOG_ID, sct.id.key_id)
        self.assertEqual(1373015623951L, sct.timestamp)
        self.assertEqual(client_pb2.DigitallySigned.SHA256,
                         sct.signature.hash_algorithm)
        self.assertEqual(client_pb2.DigitallySigned.ECDSA,
                         sct.signature.sig_algorithm)
        RAW_SIGNATURE = ('304402202080fb4a50c159e3398d9cf85cec0b3b551d4379db1d'
                         '820b3d6bca52107a32180220286b2f20f0d98039f14e3198f1f4'
                         '81c24975e5d0344d4a96e5ec761c253bc84f').decode('hex')
        self.assertEqual(RAW_SIGNATURE, sct.signature.signature)

    def test_add_valid_chain(self):
        certs_chain = ["one", "two", "three"]
        json_sct_response = (
            '{"sct_version":0,"id":"pLkJkLQYWBSHuxOizGdwCjw1m'
            'AT5G9+443fNDsgN3BA=","timestamp":1373015623951,\n'
            '"extensions":"",\n'
            '"signature":"BAMARjBEAiAggPtKUMFZ4zmNnPhc7As7VR1Dedsdggs9a8pSEHoy'
            'GAIgKGsvIPDZgDnxTjGY8fSBwkl15dA0TUqW5ex2HCU7yE8="}')
        mock_handler = mock.Mock()
        mock_handler.post_response_body.return_value = json_sct_response
        client = log_client.LogClient("http://ctlog", handler=mock_handler)
        received_sct = client.add_chain(certs_chain)
        mock_handler.post_response_body.assert_called_once_with(
            "http://ctlog/ct/v1/add-chain",
            post_data={'chain': [base64.b64encode(t) for t in certs_chain]})
        self._verify_sct_contents(received_sct)

    def test_fails_parsing_sct_invalid_version(self):
        json_sct_response = (
            '{"sct_version":2,"id":"pLkJkLQYWBSHuxOizGdwCjw1m'
            'AT5G9+443fNDsgN3BA=","timestamp":1373015623951,\n'
            '"extensions":"",\n'
            '"signature":"BAMARjBEAiAggPtKUMFZ4zmNnPhc7As7VR1Dedsdggs9a8pSEHoy'
            'GAIgKGsvIPDZgDnxTjGY8fSBwkl15dA0TUqW5ex2HCU7yE8="}')
        client = self.default_client()
        self.assertRaises(log_client.InvalidResponseError,
                          client._parse_sct, json_sct_response)

    def test_fails_parsing_sct_missing_contents(self):
        json_sct_response = (
            '{"sct_version":0,"id":"pLkJkLQYWBSHuxOizGdwCjw1m'
            'AT5G9+443fNDsgN3BA=",\n'
            '"extensions":"",\n'
            '"signature":"BAMARjBEAiAggPtKUMFZ4zmNnPhc7As7VR1Dedsdggs9a8pSEHoy'
            'GAIgKGsvIPDZgDnxTjGY8fSBwkl15dA0TUqW5ex2HCU7yE8="}')
        client = self.default_client()
        self.assertRaises(log_client.InvalidResponseError,
                          client._parse_sct, json_sct_response)

class RequestHandlerTest(unittest.TestCase):
    class RequestSideEffect:
        def __init__(self, num_failures, canned_response=None):
            self._num_failures = num_failures
            self._canned_response = canned_response

        def __call__(self, req_url, params, timeout, verify):
            if self._num_failures <= 0:
                return self._canned_response
            self._num_failures = self._num_failures - 1
            raise requests.exceptions.ConnectionError("incomplete read!")

    def test_uri_with_params(self):
        self.assertEqual(
            'http://www.google.com',
            log_client.RequestHandler._uri_with_params('http://www.google.com',
                                                      {}))

        self.assertIn(
            log_client.RequestHandler._uri_with_params('http://www.google.com',
                                                       {'a': 1, 'b': 2}),
            ['http://www.google.com?a=1&b=2', 'http://www.google.com?b=2&a=1'])

        self.assertIn(
            log_client.RequestHandler._uri_with_params('http://www.google.com/',
                                                       {'a': 1,
                                                        'b': 'foo bar'}),
            ['http://www.google.com/?a=1&b=foo+bar',
             'http://www.google.com/?b=foo+bar&a=1'])

    def test_get_response_one_retry(self):
        expected_body = 'valid_body'
        handler = log_client.RequestHandler(num_retries=1)
        canned_response = requests.models.Response()
        canned_response.status_code = 200
        canned_response._content = expected_body
        log_client.requests.get = mock.Mock(
                side_effect=
                self.RequestSideEffect(1, canned_response))

        received_response = handler.get_response('http://www.example.com')
        self.assertEqual(expected_body, received_response.content)
        self.assertEqual(200, received_response.status_code)

    def test_get_response_too_many_retries(self):
        handler = log_client.RequestHandler(num_retries=3)
        canned_response = requests.models.Response()
        canned_response.status_code = 200
        canned_response._content = 'body'
        log_client.requests.get = mock.Mock(
                side_effect=self.RequestSideEffect(
                    4, canned_response))

        self.assertRaises(log_client.HTTPError,
                          handler.get_response, ('http://www.example.com'))


if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
    unittest.main()
