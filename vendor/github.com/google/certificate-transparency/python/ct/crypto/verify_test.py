#!/usr/bin/env python

import unittest

import base64
import gflags
import os
import sys

from ct.crypto import cert
from ct.crypto import error
from ct.crypto import pem
from ct.crypto import verify
from ct.proto import client_pb2
from ct.serialization import tls_message
import mock

FLAGS = gflags.FLAGS
gflags.DEFINE_string("testdata_dir", "../test/testdata",
                     "Location of test certs")

SYMANTEC_B64_KEY = (
    'MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEluqsHEYMG1XcDfy1lCdGV0JwOmkY4r'
    '87xNuroPS2bMBTP01CEDPwWJePa75y9CrsHEKqAy8afig1dpkIPSEUhg=='
)

VENAFI_B64_KEY = (
    'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAolpIHxdSlTXLo1s6H1OCdp'
    'Sj/4DyHDc8wLG9wVmLqy1lk9fz4ATVmm+/1iN2Nk8jmctUKK2MFUtlWXZBSpym97M7'
    'frGlSaQXUWyA3CqQUEuIJOmlEjKTBEiQAvpfDjCHjlV2Be4qTM6jamkJbiWtgnYPhJ'
    'L6ONaGTiSPm7Byy57iaz/hbckldSOIoRhYBiMzeNoA0DiRZ9KmfSeXZ1rB8y8X5urS'
    'W+iBzf2SaOfzBvDpcoTuAaWx2DPazoOl28fP1hZ+kHUYvxbcMjttjauCFx+JII0dmu'
    'ZNIwjfeG/GBb9frpSX219k1O4Wi6OEbHEr8at/XQ0y7gTikOxBn/s5wQIDAQAB'
)

def read_testdata_file(test_file):
    with open(os.path.join(FLAGS.testdata_dir, test_file), 'rb') as f:
        return f.read()


class LogVerifierTest(object):
    """Defines tests for STH and SCT verification logic.

    In order to run these tests, one or more derived classes must be created.
    These classes should also inherit from unittest.TestCase. The derived
    classes must define the following members for use by the tests:
    - self.key_info_fixture: A client_pb2.KeyInfo object
    - self.sth_fixture: A client_pb2.SthResponse object

    This is so that the tests can be run repeatedly with a variety of public
    keys and STHs (e.g. RSA, ECDSA).
    """

    def test_verify_sth(self):
        verifier = verify.LogVerifier(self.key_info_fixture)
        self.assertTrue(verifier.verify_sth(self.sth_fixture))

    def test_verify_sth_fails_for_bad_signature(self):
        verifier = verify.LogVerifier(self.key_info_fixture)
        sth_fixture = self.sth_fixture

        for i in range(len(sth_fixture.tree_head_signature)):
            # Skip the bytes that encode ASN.1 lengths: this is covered in a
            # separate test
            if i == 5 or i == 7 or i == 42:
                continue
            sth = client_pb2.SthResponse()
            sth.CopyFrom(sth_fixture)
            sth.tree_head_signature = (
                sth_fixture.tree_head_signature[:i] +
                chr(ord(sth_fixture.tree_head_signature[i]) ^ 1) +
                sth_fixture.tree_head_signature[i+1:])
            # Encoding- or SignatureError, depending on whether the modified
            # byte is a content byte or not.
            self.assertRaises((error.EncodingError, error.SignatureError),
                              verifier.verify_sth, sth)

    def test_verify_sth_consistency(self):
        old_sth = self.sth_fixture
        new_sth = client_pb2.SthResponse()
        new_sth.CopyFrom(old_sth)
        new_sth.tree_size = old_sth.tree_size + 1
        new_sth.timestamp = old_sth.timestamp + 1
        new_sth.sha256_root_hash = "a new hash"
        proof = ["some proof the mock does not care about"]

        mock_merkle_verifier = mock.Mock()
        mock_merkle_verifier.verify_tree_consistency.return_value = True

        verifier = verify.LogVerifier(self.key_info_fixture,
                                      mock_merkle_verifier)
        self.assertTrue(verifier.verify_sth_consistency(old_sth, new_sth,
                                                        proof))
        mock_merkle_verifier.verify_tree_consistency.assert_called_once_with(
            old_sth.tree_size, new_sth.tree_size, old_sth.sha256_root_hash,
            new_sth.sha256_root_hash, proof)

    def test_verify_sth_temporal_consistency(self):
        old_sth = self.sth_fixture
        new_sth = client_pb2.SthResponse()
        new_sth.CopyFrom(old_sth)
        new_sth.tree_size = old_sth.tree_size + 1
        new_sth.timestamp = old_sth.timestamp + 1

        # Merkle verifier is never used so simply set to None
        verifier = verify.LogVerifier(self.key_info_fixture,
                                      None)

        # Note we do not care about root hash inconsistency here.
        self.assertTrue(verifier.verify_sth_temporal_consistency(
            old_sth, new_sth))

    def test_verify_sth_temporal_consistency_equal_timestamps(self):
        old_sth = self.sth_fixture
        new_sth = client_pb2.SthResponse()
        new_sth.CopyFrom(old_sth)
        new_sth.tree_size = old_sth.tree_size + 1

        # Merkle verifier is never used so simply set to None
        verifier = verify.LogVerifier(self.key_info_fixture,
                                      None)

        self.assertRaises(error.ConsistencyError,
                          verifier.verify_sth_temporal_consistency,
                          old_sth, new_sth)

        new_sth.tree_size = old_sth.tree_size - 1
        self.assertRaises(error.ConsistencyError,
                          verifier.verify_sth_temporal_consistency,
                          old_sth, new_sth)

        # But identical STHs are OK
        self.assertTrue(verifier.verify_sth_temporal_consistency(
            old_sth, old_sth))

    def test_verify_sth_temporal_consistency_reversed_timestamps(self):
        old_sth = self.sth_fixture
        new_sth = client_pb2.SthResponse()
        new_sth.CopyFrom(old_sth)
        new_sth.timestamp = old_sth.timestamp + 1
        new_sth.tree_size = old_sth.tree_size + 1

        # Merkle verifier is never used so simply set to None
        verifier = verify.LogVerifier(self.key_info_fixture,
                                      None)

        self.assertRaises(ValueError,
                          verifier.verify_sth_temporal_consistency,
                          new_sth, old_sth)

    def test_verify_sth_temporal_consistency_newer_tree_is_smaller(self):
        old_sth = self.sth_fixture
        new_sth = client_pb2.SthResponse()
        new_sth.CopyFrom(old_sth)
        new_sth.timestamp = old_sth.timestamp + 1
        new_sth.tree_size = old_sth.tree_size - 1

        # Merkle verifier is never used so simply set to None
        verifier = verify.LogVerifier(self.key_info_fixture,
                                      None)

        self.assertRaises(error.ConsistencyError,
                          verifier.verify_sth_temporal_consistency,
                          old_sth, new_sth)

    def test_verify_sth_consistency_invalid_proof(self):
        old_sth = self.sth_fixture
        new_sth = client_pb2.SthResponse()
        new_sth.CopyFrom(old_sth)
        new_sth.tree_size = old_sth.tree_size + 1
        new_sth.timestamp = old_sth.timestamp + 1
        new_sth.sha256_root_hash = "a new hash"
        proof = ["some proof the mock does not care about"]

        mock_merkle_verifier = mock.Mock()
        mock_merkle_verifier.verify_tree_consistency.side_effect = (
            error.ConsistencyError("Evil"))

        verifier = verify.LogVerifier(self.key_info_fixture,
                                      mock_merkle_verifier)
        self.assertRaises(error.ConsistencyError,
                          verifier.verify_sth_consistency,
                          old_sth, new_sth, proof)

    def _test_verify_sct(self, proof, chain, fake_timestamp = None):
        sct = client_pb2.SignedCertificateTimestamp()
        tls_message.decode(read_testdata_file(proof), sct)
        if fake_timestamp is not None:
            sct.timestamp = fake_timestamp

        chain = map(lambda name: cert.Certificate.from_pem_file(
                        os.path.join(FLAGS.testdata_dir, name)), chain)

        key_info = client_pb2.KeyInfo()
        key_info.type = client_pb2.KeyInfo.ECDSA
        key_info.pem_key = read_testdata_file('ct-server-key-public.pem')

        verifier = verify.LogVerifier(key_info)
        return verifier.verify_sct(sct, chain)

    def _test_verify_embedded_scts(self, chain):
        chain = map(lambda name: cert.Certificate.from_pem_file(
                        os.path.join(FLAGS.testdata_dir, name)), chain)

        key_info = client_pb2.KeyInfo()
        key_info.type = client_pb2.KeyInfo.ECDSA
        key_info.pem_key = read_testdata_file('ct-server-key-public.pem')

        verifier = verify.LogVerifier(key_info)
        return verifier.verify_embedded_scts(chain)

    def test_verify_sct_valid_signature(self):
        self.assertTrue(self._test_verify_sct(
                                'test-cert.proof',
                                ['test-cert.pem', 'ca-cert.pem']))

    def test_verify_sct_invalid_signature(self):
        self.assertRaises(error.SignatureError,
                          self._test_verify_sct,
                          'test-cert.proof',
                          ['test-cert.pem', 'ca-cert.pem'],
                          fake_timestamp = 1234567)

    def test_verify_sct_precertificate_valid_signature(self):
        self.assertTrue(self._test_verify_sct(
                                'test-embedded-pre-cert.proof',
                                ['test-embedded-pre-cert.pem', 'ca-cert.pem']))

    def test_verify_sct_precertificate_invalid_signature(self):
        self.assertRaises(error.SignatureError,
                          self._test_verify_sct,
                          'test-embedded-pre-cert.proof',
                          ['test-embedded-pre-cert.pem', 'ca-cert.pem'],
                          fake_timestamp = 1234567)

    def test_verify_sct_precertificate_with_preca_valid_signature(self):
        self.assertTrue(self._test_verify_sct(
                                'test-embedded-with-preca-pre-cert.proof',
                                ['test-embedded-with-preca-pre-cert.pem',
                                 'ca-pre-cert.pem', 'ca-cert.pem']))

    def test_verify_sct_missing_leaf_cert(self):
        self.assertRaises(error.IncompleteChainError,
                          self._test_verify_sct,
                          'test-cert.proof',
                          [])

    def test_verify_sct_missing_issuer_cert(self):
        self.assertRaises(error.IncompleteChainError,
                          self._test_verify_sct,
                          'test-embedded-pre-cert.proof',
                          ['test-embedded-pre-cert.pem'])

    def test_verify_sct_with_preca_missing_issuer_cert(self):
        self.assertRaises(error.IncompleteChainError,
                          self._test_verify_sct,
                          'test-embedded-with-preca-pre-cert.proof',
                          ['test-embedded-with-preca-pre-cert.pem',
                           'ca-pre-cert.pem'])

    def test_verify_embedded_scts_valid_signature(self):
        sct = client_pb2.SignedCertificateTimestamp()
        tls_message.decode(read_testdata_file('test-embedded-pre-cert.proof'),
                           sct)

        result = self._test_verify_embedded_scts(
                    ['test-embedded-cert.pem', 'ca-cert.pem'])
        self.assertEqual(result, [(sct, True)])

    def test_verify_embedded_scts_invalid_signature(self):
        result = self._test_verify_embedded_scts(
                    ['test-invalid-embedded-cert.pem', 'ca-cert.pem'])
        self.assertFalse(result[0][1])

    def test_verify_embedded_scts_with_preca_valid_signature(self):
        sct = client_pb2.SignedCertificateTimestamp()
        tls_message.decode(
                read_testdata_file('test-embedded-with-preca-pre-cert.proof'),
                sct)

        result = self._test_verify_embedded_scts(
                    ['test-embedded-with-preca-cert.pem', 'ca-cert.pem'])
        self.assertEqual(result, [(sct, True)])


class LogVerifierRsaTest(LogVerifierTest, unittest.TestCase):
    sth_fixture = client_pb2.SthResponse()
    sth_fixture.tree_size = 1130
    sth_fixture.timestamp = 1442500998291
    sth_fixture.sha256_root_hash = (
        "58f4e84d26f179829da3359a23f2ec519f83e99d9230aad6bfb37e2faa82c663"
        ).decode("hex")

    sth_fixture.tree_head_signature = (
        "040101002595c278829d558feb560c5024048ce1ca9e5329cc79b074307f0b6168dda1"
        "5b27f84c94cce39f8371aa8205d73a7101b434b6aeaf3c852b8471daa05d654463b334"
        "5103c7406dbd4642c8cc89eababa84e9ad663ffb3cc87940c3689d0c2ac6246915f221"
        "5da254981206fed8505eed268bcc94e05cd83c8e8e5a14407a6d15c8071fabaed9728a"
        "02830c6aef95969b0576c7ae09d50bdfc8b0b58fa759458c6d62383d6fe1072c0da103"
        "1baddfa363b58ca78f93f329b1f1a15b9575988974dcba2421b9a1bb2a617d8b3f4046"
        "ead6095f8496075edc686ae4fa672d4974de0fb9326dc3c628f7e44c7675d2c56d1c66"
        "32bbb9e4a69e0a7e34bd1d6dc7b4b2").decode("hex")

    key_info_fixture = client_pb2.KeyInfo()
    key_info_fixture.type = client_pb2.KeyInfo.RSA
    key_info_fixture.pem_key = (
        "-----BEGIN PUBLIC KEY-----\n"
        "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAolpIHxdSlTXLo1s6H1OC\n"
        "dpSj/4DyHDc8wLG9wVmLqy1lk9fz4ATVmm+/1iN2Nk8jmctUKK2MFUtlWXZBSpym\n"
        "97M7frGlSaQXUWyA3CqQUEuIJOmlEjKTBEiQAvpfDjCHjlV2Be4qTM6jamkJbiWt\n"
        "gnYPhJL6ONaGTiSPm7Byy57iaz/hbckldSOIoRhYBiMzeNoA0DiRZ9KmfSeXZ1rB\n"
        "8y8X5urSW+iBzf2SaOfzBvDpcoTuAaWx2DPazoOl28fP1hZ+kHUYvxbcMjttjauC\n"
        "Fx+JII0dmuZNIwjfeG/GBb9frpSX219k1O4Wi6OEbHEr8at/XQ0y7gTikOxBn/s5\n"
        "wQIDAQAB\n"
        "-----END PUBLIC KEY-----\n")


class LogVerifierEcdsaTest(LogVerifierTest, unittest.TestCase):
    sth_fixture = client_pb2.SthResponse()
    sth_fixture.tree_size = 42
    sth_fixture.timestamp = 1348589667204
    sth_fixture.sha256_root_hash = (
        "18041bd4665083001fba8c5411d2d748e8abbfdcdfd9218cb02b68a78e7d4c23"
        ).decode("hex")

    sth_fixture.tree_head_signature = (
        "040300483046022100befd8060563763a5e49ba53e6443c13f7624fd6403178113736e"
        "16012aca983e022100f572568dbfe9a86490eb915c4ee16ad5ecd708fed35ed4e5cd1b"
        "2c3f087b4130").decode("hex")

    key_info_fixture = client_pb2.KeyInfo()
    key_info_fixture.type = client_pb2.KeyInfo.ECDSA
    key_info_fixture.pem_key = (
        "-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAES0AfBk"
        "jr7b8b19p5Gk8plSAN16wW\nXZyhYsH6FMCEUK60t7pem/ckoPX8hupuaiJzJS0ZQ0SEoJ"
        "GlFxkUFwft5g==\n-----END PUBLIC KEY-----\n")

    def test_verify_sth_for_bad_asn1_length(self):
        verifier = verify.LogVerifier(self.key_info_fixture)
        sth_fixture = self.sth_fixture

        # The byte that encodes the length of the ASN.1 signature sequence
        i = 5

        # Decreasing the length truncates the sequence and causes a decoding
        # error.
        sth = client_pb2.SthResponse()
        sth.CopyFrom(sth_fixture)
        sth.tree_head_signature = (
            sth_fixture.tree_head_signature[:i] +
            chr(ord(sth_fixture.tree_head_signature[i]) - 1) +
            sth_fixture.tree_head_signature[i+1:])
        self.assertRaises(error.EncodingError, verifier.verify_sth, sth)

        # Increasing the length means there are not enough ASN.1 bytes left to
        # decode the sequence, however the ecdsa module silently slices it.
        # Our ECDSA verifier checks for it and will fail.
        sth = client_pb2.SthResponse()
        sth.CopyFrom(sth_fixture)
        sth.tree_head_signature = (
            sth_fixture.tree_head_signature[:i] +
            chr(ord(sth_fixture.tree_head_signature[i]) + 1) +
            sth_fixture.tree_head_signature[i+1:])
        self.assertRaises(
            error.EncodingError,
            verifier.verify_sth,
            sth)

        # The byte that encodes the length of the first integer r in the
        # sequence (r, s). Modifying the length corrupts the second integer
        # offset and causes a decoding error.
        i = 7
        sth = client_pb2.SthResponse()
        sth.CopyFrom(sth_fixture)
        sth.tree_head_signature = (
            sth_fixture.tree_head_signature[:i] +
            chr(ord(sth_fixture.tree_head_signature[i]) - 1) +
            sth_fixture.tree_head_signature[i+1:])
        self.assertRaises(error.EncodingError, verifier.verify_sth, sth)

        sth = client_pb2.SthResponse()
        sth.CopyFrom(sth_fixture)
        sth.tree_head_signature = (
            sth_fixture.tree_head_signature[:i] +
            chr(ord(sth_fixture.tree_head_signature[i]) + 1) +
            sth_fixture.tree_head_signature[i+1:])
        self.assertRaises(error.EncodingError, verifier.verify_sth, sth)

        # The byte that encodes the length of the second integer s in the
        # sequence (r, s). Increasing this length leaves bytes unread which
        # is now also detected in the verify_ecdsa module.
        i = 42
        sth = client_pb2.SthResponse()
        sth.CopyFrom(sth_fixture)
        sth.tree_head_signature = (
            sth_fixture.tree_head_signature[:i] +
            chr(ord(sth_fixture.tree_head_signature[i]) - 1) +
            sth_fixture.tree_head_signature[i+1:])
        self.assertRaises(error.EncodingError, verifier.verify_sth, sth)

        sth = client_pb2.SthResponse()
        sth.CopyFrom(sth_fixture)
        sth.tree_head_signature = (
            sth_fixture.tree_head_signature[:i] +
            chr(ord(sth_fixture.tree_head_signature[i]) + 1) +
            sth_fixture.tree_head_signature[i+1:])
        self.assertRaises(error.EncodingError, verifier.verify_sth, sth)

        # Trailing garbage is correctly detected.
        sth = client_pb2.SthResponse()
        sth.CopyFrom(sth_fixture)
        sth.tree_head_signature = (
            sth_fixture.tree_head_signature[:3] +
            # Correct outer length to include trailing garbage.
            chr(ord(sth_fixture.tree_head_signature[3]) + 1) +
            sth_fixture.tree_head_signature[4:]) + "\x01"
        self.assertRaises(error.EncodingError, verifier.verify_sth, sth)

    def test_verify_sth_for_bad_asn1_signature(self):
        # www.google.com certificate for which a bad SCT was issued.
        google_cert = (
            '-----BEGIN CERTIFICATE-----',
            'MIIEgDCCA2igAwIBAgIIdJ7+eILLLSgwDQYJKoZIhvcNAQELBQAwSTELMAkGA1UE',
            'BhMCVVMxEzARBgNVBAoTCkdvb2dsZSBJbmMxJTAjBgNVBAMTHEdvb2dsZSBJbnRl',
            'cm5ldCBBdXRob3JpdHkgRzIwHhcNMTUxMDA3MTExMDM4WhcNMTYwMTA1MDAwMDAw',
            'WjBoMQswCQYDVQQGEwJVUzETMBEGA1UECAwKQ2FsaWZvcm5pYTEWMBQGA1UEBwwN',
            'TW91bnRhaW4gVmlldzETMBEGA1UECgwKR29vZ2xlIEluYzEXMBUGA1UEAwwOd3d3',
            'Lmdvb2dsZS5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCR6Knj',
            'TG6eyvY6C1VO7daC0AbWe3cenr9y9lVFQH2ej5r87znUvep4pC/bmG71aTd25wds',
            'ScpclWNR4lkR9Ph45j8K+SjMXU7syiqFiWPWgVzyi4N3bXZw4w83RoTzfyUTn4Kx',
            '9nsQLmjVS4wUMSEpWBmYfORwUwMF8BYp5qSkIUogZTADPY7Qr8tmwEq8jLHv9z62',
            'SiYd9JEcGdhnajgXg/+/f+iIb1jhkbjsTjFJBHClgrtRqLZHSU1THZCK6iULTd1B',
            '4yBNvXcHDaSBTPUSvZvZXo/msKfOqd0fHtny1icgl5CSU0tZrZPteomMnLMGdLlN',
            'KHyqIX7XsAd3pNoXAgMBAAGjggFLMIIBRzAdBgNVHSUEFjAUBggrBgEFBQcDAQYI',
            'KwYBBQUHAwIwGQYDVR0RBBIwEIIOd3d3Lmdvb2dsZS5jb20waAYIKwYBBQUHAQEE',
            'XDBaMCsGCCsGAQUFBzAChh9odHRwOi8vcGtpLmdvb2dsZS5jb20vR0lBRzIuY3J0',
            'MCsGCCsGAQUFBzABhh9odHRwOi8vY2xpZW50czEuZ29vZ2xlLmNvbS9vY3NwMB0G',
            'A1UdDgQWBBSUPOkxr+tGC3JYs2JIdXVB2R+f8zAMBgNVHRMBAf8EAjAAMB8GA1Ud',
            'IwQYMBaAFErdBhYbvPZotXb1gba7Yhq6WoEvMCEGA1UdIAQaMBgwDAYKKwYBBAHW',
            'eQIFATAIBgZngQwBAgIwMAYDVR0fBCkwJzAloCOgIYYfaHR0cDovL3BraS5nb29n',
            'bGUuY29tL0dJQUcyLmNybDANBgkqhkiG9w0BAQsFAAOCAQEAfBoIl5qeaJ7NZ6hB',
            'WqeBZwbDV/DOHCPg3/84n8YGlfYdfXQpQdOWC5hfgEkkinBT0yp8dDTdXMUIT9Al',
            'ZMrxE54xJ1cU6FPuZPDWOnzV+6YEW6P9RnTbqKgYCNkHFiFwVvFRm5RTEGei5TLv',
            'l0zFDBusT/mgyvYBMIfW3vVPteEKKEz+aRCZHRiLAHbmJHj2+blVJeHGSF+eKN5q',
            'GWgk7/pMww4JAXsLQ0mmL8qdJKivuiNcyyhbr8IeERiVcItKqfBsX1nwyUnYFWY3',
            'HPkV+sXAPnpTGuxgYvTjcYDf8UO9lgDX5QubEFjjTuTIYAAabmc6Z4UKOS0O46Ne',
            'z28m7Q==',
            '-----END CERTIFICATE-----')
        # The SCT with the bad signature.
        sct_bytes = (
            '00ddeb1d2b7a0d4fa6208b81ad8168707e2e8e9d01d55c888d3d11c4cdb6ecbecc'
            '00000150421dfbb6000004030047304502200035de73784699d2ad8c3631aeda77'
            'f70b2c899492b16f051fd6d38d46afc892022100a4d1b58c63002e5d0862a9f623'
            'f67c8ccf5fc934bd28133fbc8f240aae4cab38'
        ).decode('hex')

        symantec_sct = client_pb2.SignedCertificateTimestamp()
        tls_message.decode(sct_bytes, symantec_sct)
        key_info = client_pb2.KeyInfo()
        key_info.type = client_pb2.KeyInfo.ECDSA
        key_info.pem_key = pem.to_pem(
            base64.decodestring(SYMANTEC_B64_KEY),
            'PUBLIC KEY')
        verifier = verify.LogVerifier(key_info)
        self.assertRaises(
            error.EncodingError,
            verifier.verify_sct,
            symantec_sct,
            [cert.Certificate.from_pem("\n".join(google_cert)),])


class CreateKeyInfoTest(unittest.TestCase):
    def test_create_key_info_with_rsa_key(self):
        key_info = verify.create_key_info_from_raw_key(
                base64.decodestring(VENAFI_B64_KEY))
        self.assertEqual(key_info.type, client_pb2.KeyInfo.RSA)
        self.assertTrue('PUBLIC KEY' in key_info.pem_key)

    def test_create_key_info_with_ecdsa_key(self):
        key_info = verify.create_key_info_from_raw_key(
                base64.decodestring(SYMANTEC_B64_KEY))
        self.assertEqual(key_info.type, client_pb2.KeyInfo.ECDSA)
        self.assertTrue('PUBLIC KEY' in key_info.pem_key)


if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
    unittest.main()
