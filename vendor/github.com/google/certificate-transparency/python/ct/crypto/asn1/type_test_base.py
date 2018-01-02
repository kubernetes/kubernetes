import unittest


from ct.crypto import error


class TypeTestBase(unittest.TestCase):
    # Test class for each concrete type should fill this in.
    asn1_type = None
    # Immutable types support hashing.
    immutable = True
    # Repeated types support lookup and assignment by index.
    repeated = True
    # Keyed types support lookup and assignment by key.
    keyed = True
    # A tuple of initializer tuples; components in each tuple should yield
    # equal objects. The first component in each tuple should be the canonical
    # value (returned by .value).
    initializers = None
    # A tuple of (bad_initializer, exception_raised) pairs.
    bad_initializers = None
    # A tuple of (value, hex_der_encoding) pairs.
    # Note: test vectors should include the complete encoding (including tag
    # and length). This is so we can lift test vectors directly from the ASN.1
    # spec and test that we recognize the correct tag for each type.
    # However test vectors for invalid encodings should focus on type-specific
    # corner cases. It's not necessary for each type to verify that invalid
    # tags and lengths are rejected: this is covered in separate tests.
    encode_test_vectors = None
    # A tuple of of serialized, hex-encoded values.
    bad_encodings = None
    # A tuple of (value, hex_encoding) pairs that can only be decoded
    # in non-strict mode.
    bad_strict_encodings = None

    def test_create(self):
        for initializer_set in self.initializers:
            value = initializer_set[0]
            # The canonical initializer.
            for i in initializer_set:
                o1 = self.asn1_type(value=i)
                self.assertEqual(o1.value, value)
                # And other initializers that yield the same value.
                for j in initializer_set:
                    o2 = self.asn1_type(value=j)
                    self.assertEqual(o2.value, value)
                    self.assertEqual(o1, o2)
                    if self.immutable:
                        self.assertEqual(hash(o1), hash(o2))
                    elif self.repeated:
                        self.assertEqual(len(o1), len(o2))
                        for i in range(len(o1)):
                            self.assertEqual(o1[i], o2[i])
                    elif self.keyed:
                        self.assertEqual(len(o1), len(o2))
                        self.assertEqual(o1.keys(), o2.keys())
                        for key in o1:
                            self.assertEqual(o1[key], o2[key])

        # Sanity-check: different initializers yield different values.
        for i in range(len(self.initializers)):
            for j in range(i+1, len(self.initializers)):
                o1 = self.asn1_type(value=self.initializers[i][0])
                o2 = self.asn1_type(value=self.initializers[j][0])
                self.assertNotEqual(o1, o2)
                if self.immutable:
                    self.assertNotEqual(hash(o1), hash(o2))
                self.assertNotEqual(o1.value, o2.value)

    def test_create_fails(self):
        for init, err in self.bad_initializers:
            self.assertRaises(err, self.asn1_type, init)

    def test_encode_decode(self):
        for value, enc in self.encode_test_vectors:
            o1 = self.asn1_type(value=value)
            o2 = self.asn1_type.decode(enc.decode("hex"))
            self.assertEqual(o1, o2)
            self.assertEqual(o1.value, o2.value)
            self.assertEqual(enc, o1.encode().encode("hex"))
            self.assertEqual(enc, o2.encode().encode("hex"))

    def test_decode_fails(self):
        for bad_enc in self.bad_encodings:
            self.assertRaises(error.ASN1Error, self.asn1_type.decode,
                bad_enc.decode("hex"))
            self.assertRaises(error.ASN1Error, self.asn1_type.decode,
                bad_enc.decode("hex"), strict=False)

    def test_strict_decode_fails(self):
        for value, bad_enc in self.bad_strict_encodings:
            o = self.asn1_type(value=value)
            self.assertRaises(error.ASN1Error,
                              self.asn1_type.decode, bad_enc.decode("hex"))
            o2 = self.asn1_type.decode(bad_enc.decode("hex"), strict=False)
            self.assertEqual(o, o2)
            # The object should keep its original encoding...
            self.assertEqual(bad_enc, o2.encode().encode("hex"))
            # ... which is not the canonical encoding.
            self.assertNotEqual(bad_enc, o.encode().encode("hex"))
