#!/usr/bin/env python
# coding=utf-8
import unittest

import mock
from ct.cert_analysis import base_check_test
from ct.cert_analysis import dnsnames
from ct.cert_analysis import tld_list
from ct.cert_analysis import tld_check
from ct.test import test_config

def gen_dns_name(name):
    dns_name = mock.Mock()
    dns_name.value = name
    return dns_name

def cert_with_urls(*args):
    certificate = mock.MagicMock()
    certificate.subject_dns_names = mock.Mock(return_value=list(args))
    return certificate

EXAMPLE = gen_dns_name("example.com")
EXAMPLE_WILDCARD = gen_dns_name("*.example.com")
UTF8_URL = gen_dns_name("ćęrtifićątętrąńśpąręńćy.com")
NON_UTF8_URL = gen_dns_name("\xff.com")
URL_INVALID_CHARACTERS_5 = gen_dns_name("[][]].com")
EMAIL_ADDRESS = gen_dns_name("example@example.com")
NOT_TLD = gen_dns_name("asdf.asdf")
WILDCARD_TLD = gen_dns_name("*.com")
NON_UNICODE_TLD = gen_dns_name("\xff\x00.com")

class DnsnamesTest(base_check_test.BaseCheckTest):
    def setUp(self):
        tld_check.CheckTldMatches.TLD_LIST_ = tld_list.TLDList(
                tld_dir=test_config.get_tld_directory(),
                tld_file_name="test_tld_list")

    def test_dnsnames_valid(self):
        certificate = cert_with_urls(EXAMPLE)
        check = dnsnames.CheckValidityOfDnsnames()
        result = check.check(certificate)
        self.assertEqual(len(result), 0)

    def test_dnsnames_wildcard(self):
        certificate = cert_with_urls(EXAMPLE_WILDCARD)
        check = dnsnames.CheckValidityOfDnsnames()
        result = check.check(certificate)
        self.assertEqual(len(result), 0)

    def test_dnsnames_utf8(self):
        certificate = cert_with_urls(UTF8_URL)
        check = dnsnames.CheckValidityOfDnsnames()
        result = check.check(certificate)
        self.assertEqual(len(result), 0)

    def test_dnsnames_non_utf8(self):
        certificate = cert_with_urls(NON_UTF8_URL)
        check = dnsnames.CheckValidityOfDnsnames()
        result = check.check(certificate)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].reason)

    def test_dnsnames_invalid_chars(self):
        certificate = cert_with_urls(URL_INVALID_CHARACTERS_5)
        check = dnsnames.CheckValidityOfDnsnames()
        result = check.check(certificate)
        self.assertEqual(len(result), 5)
        for res in result:
            self.assertIsNotNone(res.details)

    def test_dnsnames_email(self):
        certificate = cert_with_urls(EMAIL_ADDRESS)
        check = dnsnames.CheckValidityOfDnsnames()
        result = check.check(certificate)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].reason)
        self.assertIn('@', ''.join(result[0].details))

    def test_dnsnames_multiple_names(self):
        certificate = cert_with_urls(EXAMPLE, EXAMPLE_WILDCARD, UTF8_URL,
                                     NON_UTF8_URL, URL_INVALID_CHARACTERS_5)
        check = dnsnames.CheckValidityOfDnsnames()
        result = check.check(certificate)
        # 1 from NON_UTF8, 5 from INVALID_CHARACTERS_5
        self.assertEqual(len(result), 6)

    def test_dnsnames_tld_match(self):
        certificate = cert_with_urls(EXAMPLE)
        check = dnsnames.CheckTldMatches()
        result = check.check(certificate)
        self.assertEqual(len(result), 0)

    def test_dnsnames_no_tld_match(self):
        certificate = cert_with_urls(NOT_TLD)
        check = dnsnames.CheckTldMatches()
        result = check.check(certificate)
        self.assertIn(dnsnames.NoTldMatch().description, ''.join([
                obs.description for obs in result]))

    def test_dnsnames_wildcard_tld_match(self):
        certificate = cert_with_urls(WILDCARD_TLD)
        check = dnsnames.CheckTldMatches()
        result = check.check(certificate)
        self.assertIn(dnsnames.GenericWildcard().description, ''.join([
                obs.description for obs in result]))

    def test_dnsnames_non_unicode_match(self):
        certificate = cert_with_urls(NON_UNICODE_TLD)
        check = dnsnames.CheckTldMatches()
        result = check.check(certificate)
        self.assertIn(dnsnames.NonUnicodeAddress().description, ''.join([
                obs.description for obs in result]))
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
