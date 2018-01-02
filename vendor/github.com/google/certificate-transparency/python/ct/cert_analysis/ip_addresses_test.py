#!/usr/bin/env python
import unittest

import mock

from ct.cert_analysis import base_check_test
from ct.cert_analysis import ip_addresses
from ct.crypto import cert



class IpAddressesTest(base_check_test.BaseCheckTest):
    class FakeIPAddress(object):
        def __init__(self, *args):
            self.octets = args

        def as_octets(self):
            return self.octets


    def test_corrupt_extension(self):
        certificate = mock.MagicMock()
        certificate.subject_ip_addresses = mock.Mock(
                side_effect=cert.CertificateError("Boom!"))
        check = ip_addresses.CheckCorruptIpAddresses()
        result = check.check(certificate)
        self.assertObservationIn(ip_addresses.CorruptIPAddress(), result)

    def test_private_ipv4(self):
        certificate = mock.MagicMock()
        certificate.subject_ip_addresses = mock.Mock(return_value=
                                            [self.FakeIPAddress(10, 0, 0, 5),
                                            self.FakeIPAddress(192, 168, 0, 1),
                                            self.FakeIPAddress(172, 16, 5, 5),
                                            self.FakeIPAddress(172, 31, 3, 3),
                                            self.FakeIPAddress(172, 27, 42, 4)])
        check = ip_addresses.CheckPrivateIpAddresses()
        result = check.check(certificate)
        self.assertEqual(len(result), 5)

    def test_not_private_ipv4(self):
        certificate = mock.MagicMock()
        certificate.subject_ip_addresses = mock.Mock(return_value=
                                             [self.FakeIPAddress(11, 0, 0, 5),
                                             self.FakeIPAddress(172, 32, 0, 5),
                                             self.FakeIPAddress(172, 5, 1, 1),
                                             self.FakeIPAddress(192, 15, 0, 0)])
        check = ip_addresses.CheckPrivateIpAddresses()
        result = check.check(certificate)
        self.assertEqual(len(result), 0)

    def test_not_private_ipv6(self):
        certificate = mock.MagicMock()
        certificate.subject_ip_addresses = mock.Mock(return_value=[
                self.FakeIPAddress(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15)])
        check = ip_addresses.CheckPrivateIpAddresses()
        result = check.check(certificate)
        self.assertEqual(len(result), 0)

    def test_private_ipv6(self):
        certificate = mock.MagicMock()
        certificate.subject_ip_addresses = mock.Mock(return_value=[
                self.FakeIPAddress(253, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                   13, 14, 15)])
        check = ip_addresses.CheckPrivateIpAddresses()
        result = check.check(certificate)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
