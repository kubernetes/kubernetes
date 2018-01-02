#!/usr/bin/env python
import unittest

import copy
from ct.cert_analysis import base_check_test
from ct.cert_analysis import extensions
from ct.crypto.asn1 import oid
from ct.crypto.asn1 import types
from ct.crypto import cert

def remove_extension(certificate, ex_oid):
    # If given extension exists in certificate, this function will remove it
    extensions = certificate.get_extensions()
    for i, ext in enumerate(extensions):
        if ext["extnID"] == ex_oid:
            del extensions[i]
            break

def set_extension_criticality(certificate, ex_oid, value):
    extensions = certificate.get_extensions()
    for ext in extensions:
        if ext["extnID"] == ex_oid:
            ext["critical"] = types.Boolean(value)

CORRECT_LEAF = cert.Certificate.from_pem_file("ct/crypto/testdata/youtube.pem")
CORRECT_CA = cert.Certificate.from_pem_file("ct/crypto/testdata/subrigo_net.pem")
CORRECT_SUBORDINATE = cert.Certificate.from_pem_file("ct/crypto/testdata/"
                                             "verisign_intermediate.pem")


class ExtensionsTest(base_check_test.BaseCheckTest):
    def test_good_leaf_cert(self):
        check = extensions.CheckCorrectExtensions()
        result = check.check(CORRECT_LEAF)
        self.assertEqual(len(result), 0)

    def test_good_ca_cert(self):
        check = extensions.CheckCorrectExtensions()
        result = check.check(CORRECT_CA)
        self.assertEqual(len(result), 0)

    def test_good_subordinate_cert(self):
        check = extensions.CheckCorrectExtensions()
        result = check.check(CORRECT_SUBORDINATE)
        self.assertEqual(len(result), 0)

    def test_ca_missing_extension(self):
        certificate = copy.deepcopy(CORRECT_CA)
        remove_extension(certificate, oid.ID_CE_BASIC_CONSTRAINTS)
        check = extensions.CheckCorrectExtensions()
        result = check.check(certificate)
        self.assertObservationIn(
                extensions.LackOfRequiredExtension(extensions._ROOT,
                                                   extensions._oid_to_string(
                                                    oid.ID_CE_BASIC_CONSTRAINTS)),
                result)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
  unittest.main()
