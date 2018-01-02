#!/usr/bin/env python
# coding=utf-8
import unittest

import mock
from ct.cert_analysis import tld_check

def gen_dns_name(name):
    dns_name = mock.Mock()
    dns_name.value = name
    return dns_name

EXAMPLE = gen_dns_name("example.com")

class TLDCheckTest(unittest.TestCase):
    @mock.patch('ct.cert_analysis.tld_list.TLDList')
    def test_tld_list_not_created_until_check_called(self, tld_list):
        instance = mock.Mock()
        instance.match_certificate_name.return_value = [False, False, False]
        tld_list.return_value = instance

        self.assertIsNone(tld_check.CheckTldMatches.TLD_LIST_)
        _ = tld_check.CheckTldMatches.check([EXAMPLE], "dNSNames: ")
        self.assertIsNotNone(tld_check.CheckTldMatches.TLD_LIST_)

if __name__ == '__main__':
  unittest.main()
