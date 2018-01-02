#!/usr/bin/env python

import unittest

from ct.crypto import error
from ct.crypto.asn1 import oid
from ct.crypto.asn1 import type_test_base


class ObjectIdentifierTest(type_test_base.TypeTestBase):
    asn1_type = oid.ObjectIdentifier

    hashable = True

    initializers = (
        ((0, 0), "0.0"),
        ((1, 2), "1.2"),
        ((2, 5), "2.5"),
        ((1, 2, 3, 4), "1.2.3.4"),
        ((1, 2, 840, 113549), "1.2.840.113549"),
        ((1, 2, 840, 113549, 1), "1.2.840.113549.1"),
        )

    bad_initializers = (
        # Too short.
        ("0", ValueError),
        ((0,), ValueError),
        (("1"), ValueError),
        ((1,), ValueError),
        # Negative components.
        ("-1", ValueError),
        ((-1,), ValueError),
        ("1.2.3.-4", ValueError),
        ((1, 2, 3, -4), ValueError),
        # Invalid components.
        ("3.2.3.4", ValueError),
        ((3, 2, 3, 4), ValueError),
        ("0.40.3.4", ValueError),
        ((0, 40, 3, 4), ValueError),
        )

    encode_test_vectors = (
        # Example from ASN.1 spec.
        ("2.100.3", "0603813403"),
        # More examples.
        ("0.0", "060100"),
        ("1.2", "06012a"),
        ("2.5", "060155"),
        ("1.2.3.4", "06032a0304"),
        ("1.2.840", "06032a8648"),
        ("1.2.840.113549", "06062a864886f70d"),
        ("1.2.840.113549.1", "06072a864886f70d01")
        )

    bad_encodings = (
        # Empty OID.
        ("0600"),
        # Last byte has high bit set.
        ("06020080"),
        ("06032a86c8"),
        # Leading '80'-octets in component.
        ("06042a8086c8"),
        # Indefinite length.
        ("06808134030000")
        )

    bad_strict_encodings = ()

    def test_dictionary(self):
        rsa = oid.ObjectIdentifier(value=oid.RSA_ENCRYPTION)
        self.assertEqual("rsaEncryption", rsa.long_name)
        self.assertEqual("RSA", rsa.short_name)

    def test_unknown_oids(self):
        unknown = oid.ObjectIdentifier(value="1.2.3.4")
        self.assertEqual("1.2.3.4", unknown.long_name)
        self.assertEqual("1.2.3.4", unknown.short_name)


if __name__ == '__main__':
    unittest.main()
