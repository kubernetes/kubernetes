#!/usr/bin/env python

import unittest

from ct.crypto import error
from ct.crypto.asn1 import x509_time


class TimeTest(unittest.TestCase):
    def verify_time(self, time_struct, year, month, day, hour, minute, sec):
        self.assertEqual(year, time_struct.tm_year)
        self.assertEqual(month, time_struct.tm_mon)
        self.assertEqual(day, time_struct.tm_mday)
        self.assertEqual(hour, time_struct.tm_hour)
        self.assertEqual(minute, time_struct.tm_min)
        self.assertEqual(sec, time_struct.tm_sec)

    def test_time(self):
        t = x509_time.UTCTime(value="130822153902Z").gmtime()
        self.verify_time(t, 2013, 8, 22, 15, 39, 2)

        t = x509_time.GeneralizedTime(value="20130822153902Z").gmtime()
        self.verify_time(t, 2013, 8, 22, 15, 39, 2)

    def test_utc_time_1900(self):
        t = x509_time.UTCTime(value="500822153902Z").gmtime()
        self.verify_time(t, 1950, 8, 22, 15, 39, 2)

    def test_time_invalid(self):
        self.assertRaises(error.ASN1Error, x509_time.UTCTime,
                          value="131322153902Z")
        self.assertRaises(error.ASN1Error, x509_time.UTCTime,
                          value="201301322153902Z")
        t = x509_time.UTCTime(value="131322153902Z", strict=False)
        self.assertRaises(error.ASN1Error, t.gmtime)
        t = x509_time.UTCTime(value="201301322153902Z", strict=False)
        self.assertRaises(error.ASN1Error, t.gmtime)

    def test_time_no_seconds(self):
        t = x509_time.UTCTime(value="0001010000Z").gmtime()
        self.verify_time(t, 2000, 1, 1, 0, 0, 0)

    def test_time_alt_gmt(self):
        t = x509_time.UTCTime(value="121214093107+0000").gmtime()
        self.verify_time(t, 2012, 12, 14, 9, 31, 7)

    def test_time_alt_tz(self):
        """
        Test parsing a timezone with old +HHMM offset format
        Right now, it is ignored.
        """
        t = x509_time.UTCTime(value="121214093107+1234").gmtime()
        self.verify_time(t, 2012, 12, 14, 9, 31, 7)

    def test_time_missing_z(self):
        self.assertRaises(x509_time.UTCTime, value="130822153902", strict=True)

        t2 = x509_time.UTCTime(value="130822153902", strict=False).gmtime()
        self.verify_time(t2, 2013, 8, 22, 15, 39, 2)


if __name__ == "__main__":
    unittest.main()
