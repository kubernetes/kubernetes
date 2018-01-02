#!/usr/bin/env python
import unittest

import mock
from ct.cert_analysis import base_check_test
from ct.cert_analysis import ca_field
from ct.crypto import cert


class CaFieldTest(base_check_test.BaseCheckTest):
    def test_ca_true_without_san_and_cn(self):
        certificate = mock.MagicMock()
        certificate.basic_constraint_ca = mock.Mock(return_value=mock.Mock())
        certificate.basic_constraint_ca.return_value.value = True
        certificate.subject_alternative_names = mock.Mock(return_value=[])
        certificate.subject_common_names = mock.Mock(return_value=[])
        check = ca_field.CheckCATrue()
        result = check.check(certificate)
        self.assertIsNone(result)

    def test_ca_true_with_san(self):
        certificate = mock.MagicMock()
        certificate.basic_constraint_ca = mock.Mock(return_value=mock.Mock())
        certificate.basic_constraint_ca.return_value.value = True
        certificate.subject_alternative_names = mock.Mock(return_value=[None])
        check = ca_field.CheckCATrue()
        result = check.check(certificate)
        self.assertObservationIn(ca_field.CaTrue(), result)

    def test_ca_not_set(self):
        certificate = mock.MagicMock()
        certificate.basic_constraint_ca = mock.Mock(return_value=mock.Mock())
        certificate.basic_constraint_ca.return_value.value = False
        check = ca_field.CheckCATrue()
        result = check.check(certificate)
        self.assertIsNone(result)

    def test_ca_raises_corrupt_extension(self):
        certificate = mock.MagicMock()
        certificate.basic_constraint_ca = mock.Mock(
                side_effect=cert.CertificateError("Boom!"))
        check = ca_field.CheckCorruptCAField()
        result = check.check(certificate)
        self.assertObservationIn(ca_field.CorruptOrMultiple(), result)

if __name__ == '__main__':
    unittest.main()
