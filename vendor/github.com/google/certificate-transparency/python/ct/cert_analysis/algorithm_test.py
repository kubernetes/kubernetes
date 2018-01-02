#!/usr/bin/env python
import unittest

import mock
import time
from ct.cert_analysis import algorithm
from ct.cert_analysis import base_check_test
from ct.crypto.asn1 import oid
from ct.crypto.asn1 import x509_common

FAKE_SHA1_IDENTIFIER = mock.Mock(return_value=
        x509_common.AlgorithmIdentifier({"algorithm": oid.ECDSA_WITH_SHA1}))
FAKE_NOT_SHA1_IDENTIFIER = mock.Mock(return_value=
        x509_common.AlgorithmIdentifier({"algorithm": oid.ECDSA_WITH_SHA224}))
FAKE_NOT_AFTER_12122017 = mock.Mock(return_value=
                                time.strptime("12 Dec 18", "%d %b %y"))
FAKE_NOT_AFTER_06141992 = mock.Mock(return_value=
                                time.strptime("12 Apr 92", "%d %b %y"))

class AlgorithmTest(base_check_test.BaseCheckTest):
    def test_check_signature_algorithms_mismatch(self):
        certificate = mock.MagicMock()
        # use real types to make this test something harder than x != y
        certificate.signature = FAKE_SHA1_IDENTIFIER
        certificate.signature_algorithm = FAKE_NOT_SHA1_IDENTIFIER
        check = algorithm.CheckSignatureAlgorithmsMismatch()
        self.assertGreater(len(check.check(certificate)), 0)

    def test_check_signature_algorithms_match(self):
        certificate = mock.MagicMock()
        certificate.signature = FAKE_NOT_SHA1_IDENTIFIER
        certificate.signature_algorithm = FAKE_NOT_SHA1_IDENTIFIER
        check = algorithm.CheckSignatureAlgorithmsMismatch()
        self.assertEqual(check.check(certificate), None)

    def test_check_tbs_certificate_algorithm_sha1_after_2017(self):
        certificate = mock.MagicMock()
        certificate.signature_algorithm = FAKE_SHA1_IDENTIFIER
        certificate.not_after = FAKE_NOT_AFTER_12122017
        check = algorithm.CheckTbsCertificateAlgorithmSHA1Ater2017()
        result = check.check(certificate)
        self.assertEqual(len(result), 1)
        self.assertIn("SHA1", result[0].description)

    def test_check_tbs_certificate_algorithm_sha1_before_2017(self):
        certificate = mock.MagicMock()
        certificate.signature_algorithm = FAKE_SHA1_IDENTIFIER
        certificate.not_after = FAKE_NOT_AFTER_06141992
        check = algorithm.CheckTbsCertificateAlgorithmSHA1Ater2017()
        result = check.check(certificate)
        self.assertIsNone(result)

    def test_check_certificate_algorithm_sha1_after_2017(self):
        certificate = mock.MagicMock()
        certificate.signature = FAKE_SHA1_IDENTIFIER
        certificate.not_after = FAKE_NOT_AFTER_12122017
        check = algorithm.CheckCertificateAlgorithmSHA1After2017()
        result = check.check(certificate)
        self.assertEqual(len(result), 1)
        self.assertIn("SHA1", result[0].description)

    def test_check_certificate_algorithm_sha1_before_2017(self):
        certificate = mock.MagicMock()
        certificate.signature = FAKE_SHA1_IDENTIFIER
        certificate.not_after = FAKE_NOT_AFTER_06141992
        check = algorithm.CheckCertificateAlgorithmSHA1After2017()
        result = check.check(certificate)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
