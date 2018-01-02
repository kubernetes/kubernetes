#!/usr/bin/env python

import unittest

from ct.proto import client_pb2
from ct.proto import test_message_pb2
from ct.serialization import tls_message


valid_test_message = test_message_pb2.TestMessage()
valid_test_message.uint_8 = 0
valid_test_message.uint_16 = 258
valid_test_message.uint_24 = 197637
valid_test_message.uint_32 = 101124105
valid_test_message.uint_48 = 11042563100175
valid_test_message.uint_64 = 255
valid_test_message.fixed_bytes = "\xff\x00"
valid_test_message.var_bytes = "hello"
valid_test_message.var_bytes2 = "world"
valid_test_message.vector_bytes.append("hello")
valid_test_message.vector_bytes.append("world")
valid_test_message.vector_uint32.append(1)
valid_test_message.vector_uint32.append(255)
valid_test_message.test_enum = test_message_pb2.TestMessage.ENUM_1
valid_test_message.select_uint32 = 2
valid_test_message.embedded_message.uint_32 = 3
valid_test_message.repeated_message.add().uint_32 = 4
valid_test_message.repeated_message.add().uint_32 = 256


# Test vectors are given as a list of serialized, hex-encoded components.
serialized_valid_test_message = [
 "00",  # 0: uint_8
 "0102",  # 1: uint_16
 "030405",  # 2: uint_24
 "06070809",  # 3: uint_32
 "0a0b0c0d0e0f",  # 4: uint_48
 "00000000000000ff",  # 5: uint_64
 "ff00",  # 6: fixed_bytes
 "05" + "hello".encode("hex"),  # 7: var_bytes
 "0005" + "world".encode("hex"),  # 8: var_bytes2
 "0c" + "05" + "hello".encode("hex") + "05" +
 "world".encode("hex"),  # 9: vector_bytes
 "0800000001000000ff", # 10: vector_uint32
 "0001", # 11: test_enum
 "00000002", # 12: select_uint32
 "0003",  # 13: embedded_message.uint_32
 "0400040100",  # 14: repeated_message
]


class TLSReaderTest(unittest.TestCase):
    def verify_decode(self, test_vector, test_message):
        serialized = "".join(test_vector).decode("hex")
        message = test_message_pb2.TestMessage()

        tls_message.decode(serialized, message)
        self.assertEqual(test_message, message,
                         msg = "%s vs %s" % (test_message, message))

    def verify_decode_fail(self, test_vector):
        serialized = "".join(test_vector).decode("hex")
        message = test_message_pb2.TestMessage()

        self.assertRaises(tls_message.TLSDecodingError,
                          tls_message.decode, serialized, message)

    def test_decode_valid(self):
        self.verify_decode(serialized_valid_test_message, valid_test_message)
        pass

    def test_decode_valid_select(self):
        test_vector = serialized_valid_test_message[:]
        test_vector[11] = "0000"
        test_vector[12] = ""

        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.test_enum = test_message_pb2.TestMessage.ENUM_0
        test_message.ClearField("select_uint32")
        self.verify_decode(test_vector, test_message)

    def test_decode_invalid_select_fails(self):
        test_vector = serialized_valid_test_message[:]
        test_vector[11] = "0000"

        self.verify_decode_fail(test_vector)

    def test_decode_too_short_fails(self):
        test_vector = serialized_valid_test_message[:]
        # var_bytes2 has a min length of 4
        test_vector[8] = "bit".encode("hex")

        self.verify_decode_fail(test_vector)

    def test_decode_empty(self):
        test_vector = serialized_valid_test_message[:]
        # var_bytes has no min length
        test_vector[7] = "00"

        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.var_bytes = ""
        self.verify_decode(test_vector, test_message)

    def test_decode_too_long_fails(self):
        test_vector = serialized_valid_test_message[:]
        # var_bytes has a max length of 16
        test_vector[7] = "16" + "Iamtoolongformyowngood".encode("hex")

        self.verify_decode_fail(test_vector)

    def test_decode_repeated_too_short_fails(self):
        test_vector = serialized_valid_test_message[:]
        # repeated_uint32 has a min total length of 4
        test_vector[10] = "00"

        self.verify_decode_fail(test_vector)

    def test_decode_repeated_too_long_fails(self):
        test_vector = serialized_valid_test_message[:]
        # repeated_uint32 has a max total length of 8
        test_vector[10] = "0c" + "00"*12

        self.verify_decode_fail(test_vector)

    def test_decode_repeated_invalid_contents_fails(self):
        test_vector = serialized_valid_test_message[:]
        # repeated_uint32 must be a multiple of 4
        test_vector[10] = "02" + "0000"

        self.verify_decode_fail(test_vector)

    def test_read_longer_buffer(self):
        test_vector = serialized_valid_test_message[:]
        test_vector.append("somegarbageintheend".encode("hex"))
        serialized = "".join(test_vector).decode("hex")
        message = test_message_pb2.TestMessage()

        reader = tls_message.TLSReader(serialized)
        reader.read(message)

        self.assertEqual(valid_test_message, message,
                         msg = "%s vs %s" % (valid_test_message, message))
        self.assertFalse(reader.finished())


class TLSWriterTest(unittest.TestCase):
    def verify_encode(self, test_message, test_vector):
        serialized = tls_message.encode(test_message)
        self.assertEqual("".join(test_vector), serialized.encode("hex"))

    def verify_encode_fails(self, test_message):
        self.assertRaises(tls_message.TLSEncodingError,
                          tls_message.encode, test_message)

    def test_encode(self):
        self.verify_encode(valid_test_message, serialized_valid_test_message)

    def test_encode_ignores_skipped_fields(self):
        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.skip_uint32 = 42
        self.verify_encode(test_message, serialized_valid_test_message)

    def test_encode_ignores_bad_select(self):
        test_vector = serialized_valid_test_message[:]
        test_vector[11] = "0000"
        test_vector[12] = ""

        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.test_enum = test_message_pb2.TestMessage.ENUM_0
        self.verify_encode(test_message, test_vector)

    def test_encode_too_large_value_fails(self):
        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.uint_8 = 65000
        self.verify_encode_fails(test_message)

    def test_encode_bad_length_fails(self):
        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.fixed_bytes = "hello"
        self.verify_encode_fails(test_message)

    def test_encode_too_short_fails(self):
        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.var_bytes2 = "sho"
        self.verify_encode_fails(test_message)

    def test_encode_too_long_fails(self):
        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.var_bytes = "Iamtoolongformyowngood"
        self.verify_encode_fails(test_message)

    def test_encode_repeated_too_long_fails(self):
        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.vector_uint32.extend([1, 2, 3, 4])
        self.verify_encode_fails(test_message)

    def test_encode_repeated_too_short_fails(self):
        test_message = test_message_pb2.TestMessage()
        test_message.CopyFrom(valid_test_message)
        test_message.ClearField("vector_uint32")
        self.verify_encode_fails(test_message)


class SCTEncodingTest(unittest.TestCase):
    def setUp(self):
        sct_proto = client_pb2.SignedCertificateTimestamp()
        sct_proto.version = client_pb2.V1
        sct_proto.id.key_id = (
            "a4b90990b418581487bb13a2cc67700a3c359804f91bdfb8e377cd0ec80ddc10"
            ).decode('hex')

        sct_proto.timestamp = 1365427532443
        sct_proto.signature.hash_algorithm = client_pb2.DigitallySigned.SHA256
        sct_proto.signature.sig_algorithm = client_pb2.DigitallySigned.ECDSA
        sct_proto.signature.signature = (
            "304502210089de897f603e590b1aa0d7c4236c2f697e90602795f7a469215fda5e"
            "460123fc022065ab501ce3dbaf49bd563d1c9ff0ac76120bc11f65a44122b3cd8b"
            "89fc77a48c").decode("hex")
        self._sct_proto = sct_proto

    def test_correctly_encodes_sct(self):
        sct = tls_message.encode(self._sct_proto)
        expected_sct = ("00a4b90990b418581487bb13a2cc67700a3c359804f91bdfb8e377"
                        "cd0ec80ddc100000013de9d2b29b000004030047304502210089de"
                        "897f603e590b1aa0d7c4236c2f697e90602795f7a469215fda5e46"
                        "0123fc022065ab501ce3dbaf49bd563d1c9ff0ac76120bc11f65a4"
                        "4122b3cd8b89fc77a48c").decode("hex")
        self.assertEqual(sct, expected_sct)

    def test_correctly_encodes_sct_list_one_sct(self):
        # Taken from the C++ serializer test, to ensure this encoder
        # produces results compatible with the C++ one.
        single_sct = ("0069616d617075626c69636b657973686174776f6669766573697864"
                      "696765737400000000000004d20000040300097369676e6174757265"
                      ).decode("hex")
        sct_list = client_pb2.SignedCertificateTimestampList()
        sct_list.sct_list.append(single_sct)
        encoded_sct_list = tls_message.encode(sct_list)
        self.assertEqual(encoded_sct_list[:4],
                         "003a0038".decode("hex"))
        self.assertEqual(encoded_sct_list[4:], single_sct)

    def test_correctly_encodes_sct_list_multiple_scts(self):
        first_sct = tls_message.encode(self._sct_proto)
        sct_proto_2 = client_pb2.SignedCertificateTimestamp()
        sct_proto_2.CopyFrom(self._sct_proto)
        sct_proto_2.timestamp = 1365427530000
        second_sct = tls_message.encode(sct_proto_2)

        sct_list = client_pb2.SignedCertificateTimestampList()
        sct_list.sct_list.extend([first_sct, second_sct])
        encoded_sct_list = tls_message.encode(sct_list)
        # First 2 bytes are list length prefix - 240 bytes in total
        # Next 2 bytes are the length of the first SCT: 118
        self.assertEqual(encoded_sct_list[:4],
                         "00f00076".decode("hex"))
        first_sct_end = len(first_sct) + 4
        # The actual SCT
        self.assertEqual(encoded_sct_list[4:first_sct_end], first_sct)
        # Next 2 bytes are the length of the second SCT (118 again)
        self.assertEqual(encoded_sct_list[first_sct_end:first_sct_end+2],
                         "0076".decode("hex"))
        # The 2nd SCT
        self.assertEqual(encoded_sct_list[first_sct_end+2:], second_sct)


if __name__ == "__main__":
    unittest.main()
