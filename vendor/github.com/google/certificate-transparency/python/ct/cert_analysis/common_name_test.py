#!/usr/bin/env python
# coding=utf-8
import unittest

import mock
from ct.cert_analysis import base_check_test
from ct.cert_analysis import common_name
from ct.cert_analysis import tld_check
from ct.cert_analysis import tld_list
from ct.crypto import cert
from ct.test import test_config

def gen_common_name(name):
    common_name = mock.Mock()
    common_name.value = name
    return common_name

def cert_with_urls(*args):
    certificate = mock.MagicMock()
    certificate.subject_common_names = mock.Mock(return_value=list(args))
    return certificate

EXAMPLE = gen_common_name("example.com")
NOT_TLD = gen_common_name("asdf.asdf")
CA_NAME = gen_common_name("Trusty CA Ltd.")
WILDCARD_TLD = gen_common_name("*.com")
NON_UNICODE_TLD = gen_common_name("\xff\x00.com")

class CommonNameTest(base_check_test.BaseCheckTest):
    def setUp(self):
        tld_check.CheckTldMatches.TLD_LIST_ = tld_list.TLDList(
                tld_dir=test_config.get_tld_directory(),
                tld_file_name="test_tld_list")

    def test_common_name_tld_match(self):
        certificate = cert_with_urls(EXAMPLE)
        check = common_name.CheckSCNTldMatches()
        result = check.check(certificate)
        self.assertEqual(len(result), 0)

    def test_common_name_no_tld_match(self):
        certificate = cert_with_urls(NOT_TLD)
        check = common_name.CheckSCNTldMatches()
        result = check.check(certificate)
        self.assertIn(common_name.NoTldMatch().description, ''.join([
                obs.description for obs in result]))

    def test_common_name_actual_ca_name(self):
        certificate = cert_with_urls(CA_NAME)
        check = common_name.CheckSCNTldMatches()
        result = check.check(certificate)
        self.assertIn(common_name.NotAnAddress().description,
                      ''.join([obs.description for obs in result]))

    def test_common_name_wildcard_tld_match(self):
        certificate = cert_with_urls(WILDCARD_TLD)
        check = common_name.CheckSCNTldMatches()
        result = check.check(certificate)
        self.assertIn(common_name.GenericWildcard().description, ''.join([
                obs.description for obs in result]))

    def test_common_name_corrupt(self):
        certificate = mock.MagicMock()
        certificate.subject_common_names = mock.Mock(
                side_effect=cert.CertificateError("Boom!"))
        check = common_name.CheckCorruptSubjectCommonName()
        result = check.check(certificate)
        self.assertObservationIn(common_name.CorruptSubjectCommonNames(),
                                 result)

    def test_lack_of_subject_common_name(self):
        certificate = mock.MagicMock()
        certificate.subject_common_names = mock.Mock(return_value=[])
        check = common_name.CheckLackOfSubjectCommonName()
        result = check.check(certificate)
        self.assertObservationIn(common_name.NoSubjectCommonName(),
                                 result)

    def test_common_name_non_unicode_tld_match(self):
        certificate = cert_with_urls(NON_UNICODE_TLD)
        check = common_name.CheckSCNTldMatches()
        result = check.check(certificate)
        self.assertIn(common_name.NonUnicodeAddress().description,
                      ''.join([obs.description for obs in result]))
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
