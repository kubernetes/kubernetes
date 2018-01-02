#!/usr/bin/env python

import unittest
from ct.crypto.asn1 import print_util

class PrintUtilTest(unittest.TestCase):
    def test_bits_to_hex(self):
        bit_array = [0,1,1,0,1,0,1,1,1,0]
        self.assertEqual("01:ae", print_util.bits_to_hex(bit_array))
        self.assertEqual("01ae", print_util.bits_to_hex(bit_array, delimiter=""))
        self.assertEqual("", print_util.bits_to_hex(""))

    def test_bytes_to_hex(self):
        byte_array = "\x01\xae"
        self.assertEqual("01:ae", print_util.bytes_to_hex(byte_array))
        self.assertEqual("01ae", print_util.bytes_to_hex(byte_array, delimiter=""))
        self.assertEqual("", print_util.bytes_to_hex(""))

    def test_int_to_hex(self):
        integer = 1234 # 0x4d2
        self.assertEqual("04:d2", print_util.int_to_hex(integer))
        self.assertEqual("04d2", print_util.int_to_hex(integer, delimiter=""))
        negative_integer = -1234
        self.assertEqual(" -:04:d2", print_util.int_to_hex(negative_integer))

    def test_wrap_lines(self):
        long_multiline_string = "hello\nworld"
        self.assertEqual(["hel", "lo", "wor", "ld"],
                         print_util.wrap_lines(long_multiline_string, 3))

    def test_wrap_lines_no_wrap(self):
        long_multiline_string = "hello\nworld"
        self.assertEqual(["hello", "world"],
                         print_util.wrap_lines(long_multiline_string, 0))

    def test_append_lines_appends(self):
        buf = ["hello"]
        lines = ["beautiful", "world"]
        # "hellobeautiful" is more than 10 characters long
        print_util.append_lines(lines, 20, buf)
        self.assertEqual(["hellobeautiful", "world"], buf)

    def test_append_lines_honours_wrap(self):
        buf = ["hello"]
        lines = ["beautiful", "world"]
        # "hellobeautiful" is more than 10 characters long
        print_util.append_lines(lines, 10, buf)
        self.assertEqual(["hello", "beautiful", "world"], buf)

if __name__ == "__main__":
    unittest.main()
