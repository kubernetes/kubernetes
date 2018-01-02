#!/usr/bin/env python
import unittest

import mock
import time
from ct.cert_analysis import base_check_test
from ct.cert_analysis import validity
from ct.crypto import cert

class ValidityTest(base_check_test.BaseCheckTest):
    def test_not_before_regular(self):
        certificate = mock.MagicMock()
        certificate.not_before = mock.Mock(return_value=time.gmtime())
        check = validity.CheckValidityNotBeforeFuture()
        result = check.check(certificate)
        self.assertIsNone(result)

    def test_validity_corrupt(self):
        certificate = mock.MagicMock()
        certificate.not_before = mock.Mock(
                side_effect=cert.CertificateError("Boom!"))
        certificate.not_after = mock.Mock(
                side_effect=cert.CertificateError("Boom!"))
        check = validity.CheckValidityCorrupt()
        result = check.check(certificate)
        self.assertEqual(len(result), 2)
        self.assertObservationIn(validity.NotBeforeCorrupt(), result)
        self.assertObservationIn(validity.NotAfterCorrupt(), result)

    def test_validity_not_before_future(self):
        certificate = mock.MagicMock()
        # can fail on weird or slow systems
        certificate.not_before = mock.Mock(
                return_value=time.localtime((time.time()+100000000)))
        check = validity.CheckValidityNotBeforeFuture()
        result = check.check(certificate)
        self.assertIn(validity.NotBeforeInFuture(1).description, [obs.description
                                                                  for obs in
                                                                  result])

    def test_validity_not_after_not_well_defined(self):
        certificate = mock.Mock()
        certificate.is_not_after_well_defined = mock.Mock(return_value=False)
        check = validity.CheckIsExpirationDateWellDefined()
        result = check.check(certificate)
        self.assertObservationIn(validity.NotAfterNotWellDefined(), result)

    def test_validity_not_after_well_defined(self):
        certificate = mock.Mock()
        certificate.is_not_after_well_defined = mock.Mock(return_value=True)
        check = validity.CheckIsExpirationDateWellDefined()
        result = check.check(certificate)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
