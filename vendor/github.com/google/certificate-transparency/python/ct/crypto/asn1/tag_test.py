#!/usr/bin/env python

import unittest

from ct.crypto.asn1 import tag


class TagTest(unittest.TestCase):
    """Test tag encoding."""

    def test_encode_read(self):
        valid_tags = (
            # (initializers, encoding)
            ((0, tag.UNIVERSAL, tag.PRIMITIVE), "\x00"),
            ((1, tag.UNIVERSAL, tag.PRIMITIVE), "\x01"),
            ((16256, tag.UNIVERSAL, tag.PRIMITIVE), "\x1f\xff\x00"),
            ((16, tag.UNIVERSAL, tag.CONSTRUCTED), "\x30"),
            ((17, tag.UNIVERSAL, tag.CONSTRUCTED), "\x31"),
            ((16256, tag.UNIVERSAL, tag.CONSTRUCTED), "\x3f\xff\x00"),
            ((0, tag.APPLICATION, tag.PRIMITIVE), "\x40"),
            ((1, tag.APPLICATION, tag.PRIMITIVE), "\x41"),
            ((16256, tag.APPLICATION, tag.PRIMITIVE), "\x5f\xff\x00"),
            ((0, tag.APPLICATION, tag.CONSTRUCTED), "\x60"),
            ((1, tag.APPLICATION, tag.CONSTRUCTED), "\x61"),
            ((16256, tag.APPLICATION, tag.CONSTRUCTED), "\x7f\xff\x00"),
            ((0, tag.CONTEXT_SPECIFIC, tag.PRIMITIVE), "\x80"),
            ((1, tag.CONTEXT_SPECIFIC, tag.PRIMITIVE), "\x81"),
            ((16256, tag.CONTEXT_SPECIFIC, tag.PRIMITIVE), "\x9f\xff\x00"),
            ((0, tag.CONTEXT_SPECIFIC, tag.CONSTRUCTED), "\xa0"),
            ((1, tag.CONTEXT_SPECIFIC, tag.CONSTRUCTED), "\xa1"),
            ((16256, tag.CONTEXT_SPECIFIC, tag.CONSTRUCTED), "\xbf\xff\x00"),
            ((0, tag.PRIVATE, tag.PRIMITIVE), "\xc0"),
            ((1, tag.PRIVATE, tag.PRIMITIVE), "\xc1"),
            ((16256, tag.PRIVATE, tag.PRIMITIVE), "\xdf\xff\x00"),
            ((0, tag.PRIVATE, tag.CONSTRUCTED), "\xe0"),
            ((1, tag.PRIVATE, tag.CONSTRUCTED), "\xe1"),
            ((16256, tag.PRIVATE, tag.CONSTRUCTED), "\xff\xff\x00"),
            )

        for init, enc in valid_tags:
            number, tag_class, encoding = init
            t = tag.Tag(number, tag_class, encoding)
            self.assertEqual(t.number, number)
            self.assertEqual(t.tag_class, tag_class)
            self.assertEqual(t.encoding, encoding)
            self.assertEqual(t.value, enc)
            self.assertEqual((t, ""), tag.Tag.read(enc))
            self.assertEqual((t, "rest"), tag.Tag.read(enc + "rest"))

        for i in range(len(valid_tags)):
            for j in range(i+1, len(valid_tags)):
                self.assertNotEqual(tag.Tag(*valid_tags[i][0]),
                                    tag.Tag(*valid_tags[j][0]))


if __name__ == '__main__':
    unittest.main()
