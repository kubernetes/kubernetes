import json
from mock import MagicMock, patch, call
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
        result = self.r.register('foo', 80, '/v1beta1/test')

        httplibmock.assert_called_with('foo', 80)
        requestmock = httplibmock().request
        requestmock.assert_called_with(
                "POST", "/v1beta1/test",
                json.dumps(self.r.data),
                {"Content-type": "application/json",
                 "Accept": "application/json"})


    def test_command_succeeded(self):
        response = MagicMock()
        result = json.loads('{"status": "Failure", "kind": "Status", "code": 409, "apiVersion": "v1beta2", "reason": "AlreadyExists", "details": {"kind": "minion", "id": "10.200.147.200"}, "message": "minion \\"10.200.147.200\\" already exists", "creationTimestamp": null}')
        response.status = 200
        self.r.command_succeeded(response, result)
        response.status = 500
        with pytest.raises(RuntimeError):
            self.r.command_succeeded(response, result)
        response.status = 409
        with pytest.raises(ValueError):
            self.r.command_succeeded(response, result)
