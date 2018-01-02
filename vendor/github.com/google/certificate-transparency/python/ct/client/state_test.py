#!/usr/bin/env python

import unittest

import os
import tempfile

from ct.client import state
from ct.proto import client_pb2

class StateKeeperTest(unittest.TestCase):
    _DEFAULT_STATE = client_pb2.MonitorState()
    # fill with some data
    _DEFAULT_STATE.verified_sth.timestamp = 1234
    _DEFAULT_STATE.pending_sth.tree_size = 5678

    def test_read_write(self):
        handle, state_file = tempfile.mkstemp()
        os.close(handle)
        state_keeper = state.StateKeeper(state_file)
        state_keeper.write(self._DEFAULT_STATE)
        self.assertEqual(self._DEFAULT_STATE,
                         state_keeper.read(client_pb2.MonitorState))
        os.remove(state_file)

    def test_read_no_such_file(self):
        temp_dir = tempfile.mkdtemp()
        state_keeper = state.StateKeeper(temp_dir + "/foo")
        self.assertRaises(state.FileNotFoundError, state_keeper.read,
                          client_pb2.MonitorState)

    def test_read_corrupt_file(self):
        handle, state_file = tempfile.mkstemp()
        os.write(handle, "wibble")
        os.close(handle)
        state_keeper = state.StateKeeper(state_file)
        self.assertRaises(state.CorruptStateError,
                          state_keeper.read, client_pb2.MonitorState)
        os.remove(state_file)

if __name__ == "__main__":
    unittest.main()
