#!/usr/bin/env python
import unittest

import mock
from ct.cert_analysis import base_check_test
from ct.cert_analysis import serial_number


class SerialNumberTest(base_check_test.BaseCheckTest):
    def test_serial_number_positive(self):
        certificate = mock.MagicMock()
        number = mock.Mock()
        number.value = 1
        certificate.serial_number = mock.Mock(return_value=number)
        check = serial_number.CheckNegativeSerialNumber()
        result = check.check(certificate)
        self.assertIsNone(result)

    def test_serial_number_negative(self):
        certificate = mock.MagicMock()
        number = mock.Mock()
        number.value = -1
        certificate.serial_number = mock.Mock(return_value=number)
        check = serial_number.CheckNegativeSerialNumber()
        result = check.check(certificate)
        self.assertEqual(len(result), 1)

if __name__ == '__main__':
    unittest.main()
