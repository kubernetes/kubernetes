#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from mock import MagicMock, patch
from path import Path
import pytest
import sys

d = Path('__file__').parent.abspath() / 'hooks'
sys.path.insert(0, d.abspath())

from lib.registrator import Registrator


class TestRegistrator():

    def setup_method(self, method):
        self.r = Registrator()

    def test_data_type(self):
        if type(self.r.data) is not dict:
            pytest.fail("Invalid type")

    @patch('json.loads')
    @patch('httplib.HTTPConnection')
    def test_register(self, httplibmock, jsonmock):
        self.r.register('foo', 80, '/v1/test')

        httplibmock.assert_called_with('foo', 80, timeout=12)
        requestmock = httplibmock().request
        requestmock.assert_called_with(
            "POST", "/v1/test",
            json.dumps(self.r.data),
            {"Content-type": "application/json",
                "Accept": "application/json"})

    def test_command_succeeded(self):
        response = MagicMock()
        result = json.loads('{"status": "Failure", "kind": "Status", "code": 409, "apiVersion": "v1", "reason": "AlreadyExists", "details": {"kind": "node", "name": "10.200.147.200"}, "message": "node \\"10.200.147.200\\" already exists", "creationTimestamp": null}')  # noqa
        response.status = 200
        self.r.command_succeeded(response, result)
        response.status = 409
        self.r.command_succeeded(response, result)
        response.status = 500
        with pytest.raises(RuntimeError):
            self.r.command_succeeded(response, result)
