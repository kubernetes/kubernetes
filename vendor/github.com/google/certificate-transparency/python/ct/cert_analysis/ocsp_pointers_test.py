#!/usr/bin/env python
import unittest

import mock
from ct.cert_analysis import base_check_test
from ct.cert_analysis import ocsp_pointers
from ct.crypto import cert
from ct.test import test_config

CERT_WITH_OCSP = cert.Certificate.from_pem_file(
        test_config.get_test_file_path("aia.pem"))
CERT_WITHOUT_OCSP = cert.Certificate.from_pem_file(
        test_config.get_test_file_path("promise_com.pem"))

class OcspPointersTest(base_check_test.BaseCheckTest):
    def test_ocsp_existence_exist(self):
        check = ocsp_pointers.CheckOcspExistence()
        result = check.check(CERT_WITH_OCSP)
        self.assertIsNone(result)

    def test_ocsp_existence_doesnt_exist(self):
        check = ocsp_pointers.CheckOcspExistence()
        result = check.check(CERT_WITHOUT_OCSP)
        self.assertObservationIn(ocsp_pointers.LackOfOcsp(), result)

    def test_ocsp_extension_corrupt(self):
        certificate = mock.MagicMock()
        certificate.ocsp_responders = mock.Mock(
                side_effect=cert.CertificateError("Corrupt or unrecognized..."))
        check = ocsp_pointers.CheckCorruptOrMultipleAiaExtension()
        result = check.check(certificate)
        self.assertObservationIn(ocsp_pointers.CorruptAiaExtension(), result)

    def test_ocsp_extension_multiple(self):
        certificate = mock.MagicMock()
        certificate.ocsp_responders = mock.Mock(
                side_effect=cert.CertificateError("Multiple extension values"))
        check = ocsp_pointers.CheckCorruptOrMultipleAiaExtension()
        result = check.check(certificate)
        self.assertObservationIn(ocsp_pointers.MultipleOcspExtensions(), result)


if __name__ == '__main__':
    unittest.main()
