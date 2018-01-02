#!/usr/bin/env python
# coding=utf-8

import unittest

import time
from ct.crypto import cert
from ct.crypto import error
from ct.crypto.asn1 import oid
from ct.crypto.asn1 import x509_extension as x509_ext
from ct.crypto.asn1 import x509_name
from ct.test import test_config

class CertificateTest(unittest.TestCase):
    _PEM_FILE = "google_cert.pem"

    # Contains 3 certificates
    # C=US/ST=California/L=Mountain View/O=Google Inc/CN=www.google.com
    # C=US/O=Google Inc/CN=Google Internet Authority
    # C=US/O=Equifax/OU=Equifax Secure Certificate Authority
    _PEM_CHAIN_FILE = "google_chain.pem"
    _DER_FILE = "google_cert.der"
    # An X509v1 certificate
    _V1_PEM_FILE = "v1_cert.pem"

    # A old but common (0.5% of all certs as of 2013-10-01) SSL
    # cert that uses a different or older DER format for Boolean
    # values.
    _PEM_MATRIXSSL = "matrixssl_sample.pem"

    # Self-signed cert by marchnetworks.com for embedded systems
    # and uses start date in form of "0001010000Z" (no seconds)
    _PEM_MARCHNETWORKS = "marchnetworks_com.pem"

    # Self-signed cert by subrigo.net for embedded systems
    # and uses a start date in the form of 121214093107+0000
    _PEM_SUBRIGONET = "subrigo_net.pem"

    # Self-signed cert by promise.com (as of 2013-10-16) that
    # is in use by embedded systems.
    #
    # * has a start date in the format of 120703092726-1200
    # * uses a 512-key RSA key
    _PEM_PROMISECOM = "promise_com.pem"

    # This self-signed cert was used to test proper (or
    # improper) handling of UTF-8 characters in CN
    # See  CVE 2009-2408 for more details
    #
    # Mozilla bug480509
    # https://bugzilla.mozilla.org/show_bug.cgi?id=480509
    # Mozilla bug484111
    # https://bugzilla.mozilla.org/show_bug.cgi?id=484111
    # RedHat bug510251
    # https://bugzilla.redhat.com/show_bug.cgi?id=510251
    _PEM_CN_UTF8 = "cn_utf8.pem"

    # A self-signed cert with null characters in various names
    # Misparsing was involved in CVE 2009-2408 (above) and
    # CVE-2013-4248
    _PEM_NULL_CHARS = "null_chars.pem"

    # A certificate with a negative serial number, and, for more fun,
    # an extra leading ff-octet therein.
    _PEM_NEGATIVE_SERIAL = "negative_serial.pem"

    # A certificate with an ECDSA key and signature.
    _PEM_ECDSA = "ecdsa_cert.pem"

    # A certificate with multiple EKU extensions.
    _PEM_MULTIPLE_EKU = "multiple_eku.pem"

    # A certificate with multiple "interesting" SANs.
    _PEM_MULTIPLE_AN = "multiple_an.pem"

    # A certificate with multiple CN attributes.
    _PEM_MULTIPLE_CN = "multiple_cn.pem"

    # A certificate with authority cert issuer and authority cert serial.
    _PEM_AKID = "authority_keyid.pem"

    # A certificate chain with an EV policy.
    _PEM_EV_CHAIN = "ev_chain.pem"
    # EV OID for VeriSign Class 3 Public Primary Certification Authority
    _EV_POLICY_OID = oid.ObjectIdentifier(value="2.16.840.1.113733.1.7.23.6")

    _PEM_MULTIPLE_POLICIES = "multiple_policies.pem"

    # A certificate with a UserNotice containing a VisibleString.
    _PEM_USER_NOTICE = "user_notice.pem"

    # A certificate with an invalid (8-byte) IP address in a SAN.
    _PEM_INVALID_IP = "invalid_ip.pem"

    # A certificate with both kinds of AIA information.
    _PEM_AIA = "aia.pem"

    # A certificate with ASN1 indefinite length encoding.
    _PEM_INDEFINITE_LENGTH = "asn1_indefinite_length_encoding.pem"

    # A certificate with 99991231235959Z expiration date
    _PEM_NOT_WELL_DEFINED_EXPIRATION = "expiration_not_well_defined.pem"

    # A certificate with street address, postal code etc. provided
    _PEM_WITH_ADDRESS = "cert_with_address.pem"

    @property
    def pem_file(self):
        return test_config.get_test_file_path(self._PEM_FILE)

    def get_file(self, filename):
        return test_config.get_test_file_path(filename)

    def cert_from_pem_file(self, filename, strict=True):
        return cert.Certificate.from_pem_file(
            self.get_file(filename), strict_der=strict)

    def test_from_pem_file(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        self.assertTrue(isinstance(c, cert.Certificate))

    def test_certs_from_pem_file(self):
        certs = list(cert.certs_from_pem_file(self.get_file(
            self._PEM_CHAIN_FILE)))
        self.assertEqual(3, len(certs))
        self.assertTrue(all(map(lambda x: isinstance(x, cert.Certificate),
                                certs)))
        self.assertTrue("google.com" in certs[0].print_subject_name())
        self.assertTrue("Google Inc" in certs[1].print_subject_name())
        self.assertTrue("Equifax" in certs[2].print_subject_name())

    def test_from_pem(self):
        with open(self.get_file(self._PEM_FILE)) as f:
            c = cert.Certificate.from_pem(f.read())
        self.assertTrue(isinstance(c, cert.Certificate))

    def test_to_pem(self):
        with open(self.get_file(self._PEM_FILE)) as f:
            c = cert.Certificate.from_pem(f.read())
        # PEM files can and do contain arbitrary additional information,
        # so we can't assert equality with the original contents.
        # Instead, simply check that we can read the newly constructed PEM.
        new_pem = c.to_pem()
        c2 = cert.Certificate.from_pem(new_pem)
        self.assertTrue(c2.is_identical_to(c))

    def test_all_from_pem(self):
        with open(self.get_file(self._PEM_CHAIN_FILE)) as f:
            certs = list(cert.certs_from_pem(f.read()))
        self.assertEqual(3, len(certs))
        self.assertTrue(all(map(lambda x: isinstance(x, cert.Certificate),
                                certs)))
        self.assertTrue("google.com" in certs[0].print_subject_name())
        self.assertTrue("Google Inc" in certs[1].print_subject_name())
        self.assertTrue("Equifax" in certs[2].print_subject_name())

    def test_from_der_file(self):
        c = cert.Certificate.from_der_file(self.get_file(self._DER_FILE))
        self.assertTrue(isinstance(c, cert.Certificate))

    def test_from_der(self):
        with open(self.get_file(self._DER_FILE), "rb") as f:
            cert_der = f.read()
            c = cert.Certificate.from_der(cert_der)
        self.assertTrue(isinstance(c, cert.Certificate))
        self.assertEqual(c.to_der(), cert_der)

    def test_invalid_encoding_raises(self):
        self.assertRaises(error.EncodingError, cert.Certificate.from_der,
                          "bogus_der_string")
        self.assertRaises(error.EncodingError, cert.Certificate.from_pem,
                          "bogus_pem_string")

    def test_to_der(self):
        with open(self.get_file(self._DER_FILE), "rb") as f:
            der_string = f.read()
        c = cert.Certificate(der_string)
        self.assertEqual(der_string, c.to_der())

    def test_identical_to_self(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        self.assertTrue(c.is_identical_to(c))
        self.assertEqual(c, c)

    def test_identical(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        c2 = self.cert_from_pem_file(self._PEM_FILE)
        self.assertTrue(c.is_identical_to(c2))
        self.assertTrue(c2.is_identical_to(c))
        self.assertEqual(c2, c)

    def test_not_identical(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        c2 = self.cert_from_pem_file(self._V1_PEM_FILE)
        self.assertFalse(c2.is_identical_to(c))
        self.assertNotEqual(c2, c)
        self.assertNotEqual(c2, "foo")

    def test_hash(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        c2 = self.cert_from_pem_file(self._PEM_FILE)
        self.assertEqual(hash(c), hash(c))
        self.assertEqual(hash(c), hash(c2))

    def test_parse_matrixssl(self):
        """Test parsing of old MatrixSSL.org sample certificate

        As of 2013-10-01, about 0.5% of all SSL sites use an old
        sample certificate from MatrixSSL.org. It appears it's used
        mostly for various home routers.  Unfortunately it uses a
        non-DER encoding for boolean value: the DER encoding of True
        is 0xFF but this cert uses a BER encoding of 0x01. This causes
        pure DER parsers to break.  This test makes sure we can parse
        this cert without exceptions or errors.
        """
        self.assertRaises(error.ASN1Error,
                          self.cert_from_pem_file, self._PEM_MATRIXSSL)
        c = self.cert_from_pem_file(self._PEM_MATRIXSSL, strict=False)
        issuer = c.print_issuer_name()
        self.assertTrue("MatrixSSL Sample Server" in issuer)

    def test_parse_marchnetworks(self):
        """Test parsing certificates issued by marchnetworks.com."""
        c = self.cert_from_pem_file(self._PEM_MARCHNETWORKS)
        issuer = c.print_issuer_name()
        self.assertTrue("March Networks" in issuer)

        # 0001010000Z
        expected = [2000, 1, 1, 0, 0, 0, 5, 1, 0]
        self.assertEqual(list(c.not_before()), expected)

        # 3001010000Z
        expected = [2030, 1, 1, 0, 0, 0, 1, 1, 0]
        self.assertEqual(list(c.not_after()), expected)

    def test_parse_subrigonet(self):
        """Test parsing certificates issued by subrigo.net

        The certificates issued by subrigo.net (non-root)
        use an start date with time zone.

            Not Before: Dec 14 09:31:07 2012
            Not After : Dec 13 09:31:07 2022 GMT

        """
        c = self.cert_from_pem_file(self._PEM_SUBRIGONET)
        issuer = c.print_issuer_name()
        self.assertTrue("subrigo.net" in issuer)

        # timezone format -- 121214093107+0000
        expected = [2012, 12, 14, 9, 31, 7, 4, 349, 0]
        self.assertEqual(list(c.not_before()), expected)

        # standard format -- 221213093107Z
        expected = [2022, 12, 13, 9, 31, 7, 1, 347, 0]
        self.assertEqual(list(c.not_after()), expected)

    def test_utf8_names(self):
        c = self.cert_from_pem_file(self._PEM_CN_UTF8)
        nameutf8 = "ñeco ñýáěšžěšžřěčíě+ščýáíéřáíÚ"
        unicodename = u"ñeco ñýáěšžěšžřěčíě+ščýáíéřáíÚ"
        # Compare UTF-8 strings directly.
        self.assertEqual(c.print_subject_name(), "CN=" + nameutf8)
        self.assertEqual(c.print_issuer_name(), "CN=" + nameutf8)
        cns = c.subject_common_names()
        self.assertEqual(1, len(cns))
        self.assertEqual(cns[0], nameutf8)
        # Name comparison is unicode-based so decode and compare unicode names.
        # TODO(ekasper): implement proper stringprep-based name comparison
        # and use these test cases there.
        self.assertEqual(cns[0].value.decode("utf8"), unicodename)

    def test_null_chars_in_names(self):
        """Test handling null chars in subject and subject alternative names."""
        c = self.cert_from_pem_file(self._PEM_NULL_CHARS)
        cns = c.subject_common_names()
        self.assertEqual(1, len(cns))
        self.assertEqual("null.python.org\000example.org", cns[0])

        alt_names = c.subject_alternative_names()
        self.assertEqual(len(alt_names), 5)
        self.assertEqual(alt_names[0].component_key(), x509_name.DNS_NAME)
        self.assertEqual(alt_names[0].component_value(),
                         "altnull.python.org\000example.com")
        self.assertEqual(alt_names[1].component_key(), x509_name.RFC822_NAME)
        self.assertEqual(alt_names[1].component_value(),
                         "null@python.org\000user@example.org")
        self.assertEqual(alt_names[2].component_key(),x509_name.URI_NAME)
        self.assertEqual(alt_names[2].component_value(),
                         "http://null.python.org\000http://example.org")

        # the following does not contain nulls.
        self.assertEqual(alt_names[3].component_key(),
                         x509_name.IP_ADDRESS_NAME)
        self.assertEqual(alt_names[3].component_value().as_octets(),
                         (192, 0, 2, 1))
        self.assertEqual(alt_names[4].component_key(),
                         x509_name.IP_ADDRESS_NAME)
        self.assertEqual(alt_names[4].component_value().as_octets(),
                         (32, 1, 13, 184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1))

    def test_parse_promisecom(self):
        """Test parsing certificates issued by promise.com

        The certificates issued by promise.com (non-root)
        use an start date with time zone (and are 512-bit)

            Not Before: Jun 29 15:32:48 2011
            Not After : Jun 26 15:32:48 2021 GMT
        """

        c = self.cert_from_pem_file(self._PEM_PROMISECOM)
        issuer = c.print_issuer_name()
        self.assertTrue("Promise Technology Inc." in issuer)

        # 110629153248-1200
        expected = [2011,6,29,15,32,48,2,180,0]
        self.assertEqual(list(c.not_before()), expected)

        # 210626153248Z
        expected = [2021,6,26,15,32,48,5,177,0]
        self.assertEqual(list(c.not_after()), expected)

    def test_parse_ecdsa_cert(self):
        c = self.cert_from_pem_file(self._PEM_ECDSA)
        self.assertTrue("kmonos.jp" in c.print_subject_name())
        self.assertEquals(oid.ECDSA_WITH_SHA256, c.signature()["algorithm"])
        self.assertEquals(oid.ECDSA_WITH_SHA256,
                          c.signature_algorithm()["algorithm"])

    def test_print_subject_name(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        subject = c.print_subject_name()
        # C=US, ST=California, L=Mountain View, O=Google Inc, CN=*.google.com
        self.assertTrue("US" in subject)
        self.assertTrue("California" in subject)
        self.assertTrue("Mountain View" in subject)
        self.assertTrue("Google Inc" in subject)
        self.assertTrue("*.google.com" in subject)

    def test_print_issuer_name(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        issuer = c.print_issuer_name()
        # Issuer: C=US, O=Google Inc, CN=Google Internet Authority
        self.assertTrue("US" in issuer)
        self.assertTrue("Google Inc" in issuer)
        self.assertTrue("Google Internet Authority" in issuer)

    def test_subject_common_names(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        cns = c.subject_common_names()
        self.assertEqual(1, len(cns))
        self.assertEqual("*.google.com", cns[0])

    def test_multiple_subject_common_names(self):
        c = self.cert_from_pem_file(self._PEM_MULTIPLE_CN)
        cns = c.subject_common_names()
        self.assertItemsEqual(cns, ["www.rd.io", "rdio.com", "rd.io",
                                    "api.rdio.com", "api.rd.io",
                                    "www.rdio.com"])

    def test_subject_dns_names(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        dns_names = c.subject_dns_names()
        self.assertEqual(44, len(dns_names))
        self.assertTrue("*.youtube.com" in dns_names)

    def test_subject_ip_addresses(self):
        c = self.cert_from_pem_file(self._PEM_MULTIPLE_AN)
        ips = c.subject_ip_addresses()
        self.assertEqual(1, len(ips))
        self.assertEqual((129, 48, 105, 104), ips[0].as_octets())

    def test_invalid_ip_addresses(self):
        with self.assertRaises(error.ASN1Error) as fail:
            self.cert_from_pem_file(self._PEM_INVALID_IP)

        self.assertIn("00000000ffffff00", str(fail.exception))
        c = self.cert_from_pem_file(self._PEM_INVALID_IP, strict=False)
        ips = c.subject_ip_addresses()
        self.assertEqual(1, len(ips))
        self.assertEqual((0, 0, 0, 0, 255, 255, 255, 0), ips[0].as_octets())

    def test_subject_alternative_names(self):
        cert = self.cert_from_pem_file(self._PEM_MULTIPLE_AN)
        sans = cert.subject_alternative_names()
        self.assertEqual(4, len(sans))

        self.assertEqual(x509_name.DNS_NAME, sans[0].component_key())
        self.assertEqual("spires.wpafb.af.mil", sans[0].component_value())

        self.assertEqual(x509_name.DIRECTORY_NAME, sans[1].component_key())
        self.assertTrue(isinstance(sans[1].component_value(), x509_name.Name),
        sans[1].component_value())

        self.assertEqual(x509_name.IP_ADDRESS_NAME, sans[2].component_key())
        self.assertEqual((129, 48, 105, 104),
        sans[2].component_value().as_octets())

        self.assertEqual(x509_name.URI_NAME, sans[3].component_key())
        self.assertEqual("spires.wpafb.af.mil", sans[3].component_value())

    def test_no_alternative_names(self):
        c = cert.Certificate.from_pem_file(self.get_file(self._V1_PEM_FILE))
        self.assertEqual(0, len(c.subject_alternative_names()))
        self.assertEqual(0, len(c.subject_dns_names()))
        self.assertEqual(0, len(c.subject_ip_addresses()))

    def test_validity(self):
        certs = list(cert.certs_from_pem_file(
            self.get_file(self._PEM_CHAIN_FILE)))
        self.assertEqual(3, len(certs))
        # notBefore: Sat Aug 22 16:41:51 1998 GMT
        # notAfter: Wed Aug 22 16:41:51 2018 GMT
        c = certs[2]
        # These two will start failing in 2018.
        self.assertTrue(c.is_temporally_valid_now())
        self.assertFalse(c.is_expired())

        self.assertFalse(c.is_not_yet_valid())

        # Aug 22 16:41:51 2018
        self.assertTrue(c.is_temporally_valid_at(time.gmtime(1534956111)))
        # Aug 22 16:41:52 2018
        self.assertFalse(c.is_temporally_valid_at(time.gmtime(1534956112)))

        # Aug 22 16:41:50 1998
        self.assertFalse(c.is_temporally_valid_at(time.gmtime(903804110)))
        # Aug 22 16:41:51 1998
        self.assertTrue(c.is_temporally_valid_at(time.gmtime(903804111)))

    def test_basic_constraints(self):
        certs = list(cert.certs_from_pem_file(
            self.get_file(self._PEM_CHAIN_FILE)))
        self.assertFalse(certs[0].basic_constraint_ca())
        self.assertTrue(certs[1].basic_constraint_ca())
        self.assertIsNone(certs[0].basic_constraint_path_length())
        self.assertEqual(0, certs[1].basic_constraint_path_length())

    def test_version(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        self.assertEqual(3, c.version())

    def test_issuer_common_name(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        icn = c.issuer_common_name()
        self.assertIn("Google Internet Authority", icn[0].value)
        self.assertEqual(len(icn), 1)

    def test_issuer_country_name(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        icn = c.issuer_country_name()
        self.assertIn("US", icn)
        self.assertEqual(len(icn), 1)

    def test_subject_organization_name(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        icn = c.subject_organization_name()
        self.assertIn("Google Inc", icn)
        self.assertEqual(len(icn), 1)

    def test_subject_street_address(self):
        c = self.cert_from_pem_file(self._PEM_WITH_ADDRESS)
        address = c.subject_street_address()
        self.assertIn("CQ Mail Centre", address)
        self.assertIn("Building 19", address)

    def test_subject_locality_name(self):
        c = self.cert_from_pem_file(self._PEM_WITH_ADDRESS)
        locality_name = c.subject_locality_name()
        self.assertIn("Rockhampton", locality_name)

    def test_subject_state_or_province(self):
        c = self.cert_from_pem_file(self._PEM_WITH_ADDRESS)
        state_or_province = c.subject_state_or_province_name()
        self.assertIn("Queensland", state_or_province)

    def test_subject_postal_code(self):
        c = self.cert_from_pem_file(self._PEM_WITH_ADDRESS)
        postal_code = c.subject_postal_code()
        self.assertIn("4702", postal_code)

    def test_serial_number(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        self.assertEqual(454887626504608315115709, c.serial_number())

    def test_negative_serial_number(self):
        # Fails because of the leading ff-octet.
        self.assertRaises(error.ASN1Error, self.cert_from_pem_file,
            self._PEM_NEGATIVE_SERIAL)
        c = self.cert_from_pem_file(self._PEM_NEGATIVE_SERIAL,
                                    strict=False)
        self.assertEqual(-218943125988803304701934765446014018,
                          c.serial_number())

    def test_v1_cert(self):
        c = self.cert_from_pem_file(self._V1_PEM_FILE)
        self.assertEqual(1, c.version())
        self.assertIsNone(c.basic_constraint_ca())

    def test_fingerprint(self):
        c = cert.Certificate.from_der_file(self.get_file(self._DER_FILE))
        self.assertEqual(c.fingerprint().encode("hex"),
                         "570fe2e3bfee986ed4a158aed8770f2e21614659")
        self.assertEqual(c.fingerprint("sha1").encode("hex"),
                         "570fe2e3bfee986ed4a158aed8770f2e21614659")
        self.assertEqual(c.fingerprint("sha256").encode("hex"),
                         "6d4106b4544e9e5e7a0924ee86a577ffefaadae8b8dad73413a7"
                         "d874747a81d1")

    def test_key_usage(self):
        c = cert.Certificate.from_pem_file(self.get_file(self._PEM_FILE))
        self.assertTrue(c.key_usage(x509_ext.KeyUsage.DIGITAL_SIGNATURE))

        certs = [c for c in cert.certs_from_pem_file(self.get_file(
            self._PEM_CHAIN_FILE))]
        # This leaf cert does not have a KeyUsage extension.
        self.assertEqual([], certs[0].key_usages())
        self.assertIsNone(certs[0].key_usage(
            x509_ext.KeyUsage.DIGITAL_SIGNATURE))

        # The second cert has keyCertSign and cRLSign.
        self.assertIsNotNone(certs[1].key_usage(
            x509_ext.KeyUsage.DIGITAL_SIGNATURE))
        self.assertFalse(certs[1].key_usage(
            x509_ext.KeyUsage.DIGITAL_SIGNATURE))
        self.assertTrue(certs[1].key_usage(x509_ext.KeyUsage.KEY_CERT_SIGN))
        self.assertTrue(certs[1].key_usage(x509_ext.KeyUsage.CRL_SIGN))
        self.assertItemsEqual([x509_ext.KeyUsage.KEY_CERT_SIGN,
                               x509_ext.KeyUsage.CRL_SIGN],
                              certs[1].key_usages())

    def test_extended_key_usage(self):
        certs = [c for c in cert.certs_from_pem_file(self.get_file(
            self._PEM_CHAIN_FILE))]
        self.assertTrue(certs[0].extended_key_usage(oid.ID_KP_SERVER_AUTH))
        self.assertIsNotNone(
            certs[0].extended_key_usage(oid.ID_KP_CODE_SIGNING))
        self.assertFalse(certs[0].extended_key_usage(oid.ID_KP_CODE_SIGNING))
        self.assertItemsEqual([oid.ID_KP_SERVER_AUTH, oid.ID_KP_CLIENT_AUTH],
                              certs[0].extended_key_usages())

        # EKU is normally only found in leaf certs.
        self.assertIsNone(certs[1].extended_key_usage(oid.ID_KP_SERVER_AUTH))
        self.assertEqual([], certs[1].extended_key_usages())

    def test_multiple_extensions(self):
        self.assertRaises(error.ASN1Error, cert.Certificate.from_pem_file,
                          self.get_file(self._PEM_MULTIPLE_EKU))

        c = cert.Certificate.from_pem_file(self.get_file(self._PEM_MULTIPLE_EKU),
                                           strict_der=False)
        self.assertTrue("www.m-budget-mobile-abo.ch" in c.subject_common_names())
        self.assertRaises(cert.CertificateError, c.extended_key_usages)

    def test_key_identifiers(self):
        certs = [c for c in cert.certs_from_pem_file(self.get_file(
            self._PEM_CHAIN_FILE))]

        self.assertEqual("\x12\x4a\x06\x24\x28\xc4\x18\xa5\x63\x0b\x41\x6e\x95"
                         "\xbf\x72\xb5\x3e\x1b\x8e\x8f",
                         certs[0].subject_key_identifier())

        self.assertEqual("\xbf\xc0\x30\xeb\xf5\x43\x11\x3e\x67\xba\x9e\x91\xfb"
                         "\xfc\x6a\xda\xe3\x6b\x12\x24",
                         certs[0].authority_key_identifier())

        self.assertIsNone(certs[0].authority_key_identifier(
            identifier_type=x509_ext.AUTHORITY_CERT_ISSUER))
        self.assertIsNone(certs[0].authority_key_identifier(
            identifier_type=x509_ext.AUTHORITY_CERT_SERIAL_NUMBER))

        self.assertEqual(certs[0].authority_key_identifier(),
                         certs[1].subject_key_identifier())

        c = self.cert_from_pem_file(self._PEM_AKID)

        cert_issuers = c.authority_key_identifier(
            identifier_type=x509_ext.AUTHORITY_CERT_ISSUER)
        self.assertEqual(1, len(cert_issuers))

        # A DirectoryName.
        cert_issuer = cert_issuers[0]
        self.assertEqual(x509_name.DIRECTORY_NAME, cert_issuer.component_key())
        self.assertEqual(["KISA RootCA 1"],
                         cert_issuer.component_value().attributes(
                             oid.ID_AT_COMMON_NAME))
        self.assertEqual(10119, c.authority_key_identifier(
            identifier_type=x509_ext.AUTHORITY_CERT_SERIAL_NUMBER))

    def test_policies(self):
        certs = [c for c in cert.certs_from_pem_file(self.get_file(
            self._PEM_EV_CHAIN))]
        ev_cert = certs[0]
        policies = ev_cert.policies()
        self.assertEqual(1, len(policies))

        self.assertTrue(ev_cert.has_policy(self._EV_POLICY_OID))
        self.assertFalse(ev_cert.has_policy(oid.ANY_POLICY))

        policy = ev_cert.policy(self._EV_POLICY_OID)

        qualifiers = policy[x509_ext.POLICY_QUALIFIERS]
        self.assertEqual(1, len(qualifiers))

        qualifier = qualifiers[0]
        self.assertEqual(oid.ID_QT_CPS, qualifier[x509_ext.POLICY_QUALIFIER_ID])
        # CPS location is an Any(IA5String).
        self.assertEqual("https://www.verisign.com/cps",
                         qualifier[x509_ext.QUALIFIER].decoded_value)

        any_cert = certs[1]
        policies = any_cert.policies()
        self.assertEqual(1, len(policies))

        self.assertFalse(any_cert.has_policy(self._EV_POLICY_OID))
        self.assertTrue(any_cert.has_policy(oid.ANY_POLICY))

        policy = ev_cert.policy(self._EV_POLICY_OID)

        qualifiers = policy[x509_ext.POLICY_QUALIFIERS]
        self.assertEqual(1, len(qualifiers))

        qualifier = qualifiers[0]
        self.assertEqual(oid.ID_QT_CPS, qualifier[x509_ext.POLICY_QUALIFIER_ID])
        # CPS location is an IA5String.
        self.assertEqual("https://www.verisign.com/cps",
                         qualifier[x509_ext.QUALIFIER].decoded_value)

        no_policy_cert = certs[2]
        self.assertEqual(0, len(no_policy_cert.policies()))
        self.assertFalse(no_policy_cert.has_policy(self._EV_POLICY_OID))
        self.assertFalse(no_policy_cert.has_policy(oid.ANY_POLICY))

    def test_multiple_policies(self):
        c = self.cert_from_pem_file(self._PEM_MULTIPLE_POLICIES)
        policies = c.policies()
        self.assertEqual(2, len(policies))
        self.assertTrue(c.has_policy(oid.ObjectIdentifier(
            value="1.3.6.1.4.1.6449.1.2.2.7")))
        self.assertTrue(c.has_policy(oid.ObjectIdentifier(
            value="2.23.140.1.2.1")))
        self.assertFalse(c.has_policy(oid.ANY_POLICY))

    def test_user_notice(self):
        c = self.cert_from_pem_file(self._PEM_USER_NOTICE)
        policies = c.policies()
        self.assertEqual(1, len(policies))

        qualifiers = policies[0][x509_ext.POLICY_QUALIFIERS]
        self.assertEqual(2, len(qualifiers))

        qualifier = qualifiers[0]
        self.assertEqual(oid.ID_QT_UNOTICE,
                         qualifier[x509_ext.POLICY_QUALIFIER_ID])

        qualifier = qualifier[x509_ext.QUALIFIER].decoded_value
        self.assertIsNone(qualifier[x509_ext.NOTICE_REF])
        expected_text = ("For more details, please visit our website "
                         "https://www.cybertrust.ne.jp .")
        explicit_text = qualifier[x509_ext.EXPLICIT_TEXT].component_value()
        self.assertEqual(expected_text, explicit_text)

    def test_crl_distribution_points(self):
        c = self.cert_from_pem_file(self._PEM_FILE)
        crls = c.crl_distribution_points()
        self.assertEqual(1, len(crls))
        crl = crls[0]

        # Optional components, not present.
        self.assertIsNone(crl[x509_ext.REASONS])
        self.assertIsNone(crl[x509_ext.CRL_ISSUER])

        # This is the prevalent form of CRL distribution points.
        dist_points = crl[x509_ext.DISTRIBUTION_POINT]
        self.assertEqual(x509_ext.FULL_NAME, dist_points.component_key())
        self.assertEqual(1, len(dist_points.component_value()))

        # A GeneralName URI.
        dist_point = dist_points.component_value()[0]
        self.assertEqual("http://www.gstatic.com/GoogleInternetAuthority/"
                         "GoogleInternetAuthority.crl",
                         dist_point[x509_name.URI_NAME])

    def test_aia(self):
        c = self.cert_from_pem_file(self._PEM_AIA)

        ca_issuers = c.ca_issuers()
        self.assertEqual(1, len(ca_issuers))

        # A GeneralName URI.
        self.assertEqual("http://pki.google.com/GIAG2.crt",
                         ca_issuers[0][x509_name.URI_NAME])

        ocsp = c.ocsp_responders()
        self.assertEqual(1, len(ocsp))

        self.assertEqual("http://clients1.google.com/ocsp",
                         ocsp[0][x509_name.URI_NAME])

        # Cert has CA issuers but no OCSP responders.
        c = self.cert_from_pem_file(self._PEM_FILE)
        self.assertItemsEqual([], c.ocsp_responders())

    def test_is_self_signed_root(self):
        c = self.cert_from_pem_file(self._PEM_SUBRIGONET)
        self.assertTrue(c.is_self_signed())

    def test_is_self_signed_leaf(self):
        c = self.cert_from_pem_file(self._PEM_AIA)
        self.assertFalse(c.is_self_signed())

    def test_get_extensions(self):
        c = self.cert_from_pem_file(self._PEM_AIA)
        extensions = c.get_extensions()
        extensions_oids = [extension['extnID'] for extension in extensions]
        self.assertItemsEqual((oid.ID_CE_EXT_KEY_USAGE,
                               oid.ID_CE_SUBJECT_ALT_NAME,
                               oid.ID_PE_AUTHORITY_INFO_ACCESS,
                               oid.ID_CE_SUBJECT_KEY_IDENTIFIER,
                               oid.ID_CE_BASIC_CONSTRAINTS,
                               oid.ID_CE_AUTHORITY_KEY_IDENTIFIER,
                               oid.ID_CE_CERTIFICATE_POLICIES,
                               oid.ID_CE_CRL_DISTRIBUTION_POINTS),
                              extensions_oids)

    def test_indefinite_encoding(self):
        self.assertRaises(error.ASN1Error, self.cert_from_pem_file,
                          self._PEM_INDEFINITE_LENGTH)
        c = self.cert_from_pem_file(self._PEM_INDEFINITE_LENGTH, strict=False)
        issuer = c.print_issuer_name()
        self.assertTrue("VeriSign Class 1 CA" in issuer)

    def test_expiration_not_well_defined(self):
        c = self.cert_from_pem_file(self._PEM_NOT_WELL_DEFINED_EXPIRATION)
        self.assertFalse(c.is_not_after_well_defined())
        # Make sure that certificate with regular expiration date return true
        c = self.cert_from_pem_file(self._PEM_AIA)
        self.assertTrue(c.is_not_after_well_defined())


if __name__ == "__main__":
    unittest.main()
