#!/usr/bin/env python
# coding=utf-8
import unittest

from ct.cert_analysis import tld_list
from ct.test import test_config

TLD_FILE  = "test_tld_list"

class TLDListTest(unittest.TestCase):
    def default_list(self):
        return tld_list.TLDList(tld_dir=test_config.get_tld_directory(),
                                tld_file_name=TLD_FILE)

    def test_tld_list_example_matches(self):
        url = "example.com"
        tlds = self.default_list()
        self.assertEqual("com", tlds.match(url))

    def test_tld_list_doesnt_match(self):
        url = "kokojambo.i.do.przodu"
        tlds = self.default_list()
        self.assertIsNone(tlds.match(url))

    def test_tld_list_match_unicode_address(self):
        end = unicode("বাংলা", 'utf-8')
        beg = "example"
        url = '.'.join((beg, end))
        tlds = self.default_list()
        self.assertEqual(end, tlds.match(url))

    def test_tld_list_match_idna(self):
        end = unicode("বাংলা", 'utf-8')
        beg = "example"
        url = '.'.join((beg, end)).encode('idna')
        tlds = self.default_list()
        self.assertEqual(end, tlds.match_idna(url))

    def test_wildcard_match(self):
        url = "hammersmashedface.kawasaki.jp"
        tlds = self.default_list()
        self.assertEqual(url, tlds.match(url))

    def test_exception_match(self):
        url = "city.kobe.jp"
        tlds = self.default_list()
        self.assertEqual("kobe.jp", tlds.match(url))


if __name__ == '__main__':
    unittest.main()
