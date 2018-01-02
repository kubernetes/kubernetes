#!/usr/bin/env python
import unittest

import sys
from collections import defaultdict
from ct.cert_analysis import asn1
from ct.cert_analysis import base_check_test
from ct.client import reporter
from ct.client.db import cert_desc
from ct.client.db import sqlite_cert_db
from ct.client.db import sqlite_connection as sqlitecon
from ct.crypto import cert
from ct.proto import certificate_pb2
from ct.proto import client_pb2
from ct.test import test_config
import gflags

STRICT_DER = cert.Certificate.from_der_file(
        test_config.get_test_file_path('google_cert.der'), False).to_der()
NON_STRICT_DER = cert.Certificate.from_pem_file(
        test_config.get_test_file_path('invalid_ip.pem'), False).to_der()

CHAIN_FILE = test_config.get_test_file_path('google_chain.pem')

CHAIN_DERS = [c.to_der() for c in cert.certs_from_pem_file(CHAIN_FILE)]

SELF_SIGNED_ROOT_DER = cert.Certificate.from_pem_file(
        test_config.get_test_file_path('subrigo_net.pem'), False).to_der()

def readable_dn(dn_attribs):
    return ",".join(["%s=%s" % (attr.type, attr.value) for attr in dn_attribs])

class FakeCheck(object):
    @staticmethod
    def check(certificate):
        return [asn1.Strict("Boom!")]

class BadCheck(object):
    def __init__(self):
        self.certs_checked = 0

    def check(self, cert):
        self.certs_checked += 1

        if self.certs_checked == 1:
            raise ValueError("Boom!")

class CertificateReportTest(base_check_test.BaseCheckTest):
    class CertificateReportBase(reporter.CertificateReport):
        def __init__(self, checks):
            super(CertificateReportTest.CertificateReportBase, self).__init__(
                    checks=checks)

        def certs(self):
            return self._certs

        def report(self):
            super(CertificateReportTest.CertificateReportBase, self).report()
            return self.observations

        def reset(self):
            self._certs = {}
            self.observations = defaultdict(list)

        def _batch_scanned_callback(self, result):
            for desc, log_index, observations in result:
                self._certs[log_index] = desc
                self.observations[log_index] += observations

    class FakeCheck(object):
        @staticmethod
        def check(certificate):
            return [asn1.Strict("Boom!")]

    def setUp(self):
        self.cert_db = sqlite_cert_db.SQLiteCertDB(
                sqlitecon.SQLiteConnectionManager(":memory:", keepalive=True))

    def test_scan_der_cert_no_checks(self):
        report = self.CertificateReportBase([])
        report.scan_der_certs([(0, STRICT_DER, [], client_pb2.X509_ENTRY)])
        result = report.report()
        self.assertEqual(len(sum(result.values(), [])), 0)

    def test_scan_der_cert_broken_cert(self):
        report = self.CertificateReportBase([])
        report.scan_der_certs([(0, "asdf", [], client_pb2.X509_ENTRY)])
        result = report.report()
        self.assertObservationIn(asn1.All(),
                      sum(result.values(), []))
        self.assertEqual(len(sum(result.values(), [])), 1)

    def test_scan_der_cert_check(self):
        report = self.CertificateReportBase([FakeCheck()])
        report.scan_der_certs([(0, STRICT_DER, [], client_pb2.X509_ENTRY)])
        result = report.report()

        self.assertObservationIn(asn1.Strict("Boom!"),
                                 sum(result.values(), []))
        self.assertEqual(len(result), 1)

    def test_scan_der_cert_check_non_strict(self):
        report = self.CertificateReportBase([FakeCheck()])
        report.scan_der_certs([(0, NON_STRICT_DER, [], client_pb2.X509_ENTRY)])
        result = report.report()
        # There should be FakeCheck and asn.1 strict parsing failure
        self.assertEqual(len(sum(result.values(), [])), 2)
        self.assertObservationIn(asn1.Strict("Boom!"), sum(result.values(), []))

    def test_entry_type_propogated(self):
        report = self.CertificateReportBase([])
        report.scan_der_certs([(0, STRICT_DER, [], client_pb2.PRECERT_ENTRY),
                               (1, STRICT_DER, [], client_pb2.X509_ENTRY)])
        result = report.report()
        self.assertEqual(len(sum(result.values(), [])), 0)

        certs = report.certs()
        self.assertEqual(len(certs), 2)
        self.assertEquals(certs[0].entry_type, client_pb2.PRECERT_ENTRY)
        self.assertEquals(certs[1].entry_type, client_pb2.X509_ENTRY)

    def test_issuer_and_root_issuer_populated_from_chain(self):
        self.assertEqual(3, len(CHAIN_DERS))

        report = self.CertificateReportBase([])
        report.scan_der_certs([(0, CHAIN_DERS[0], CHAIN_DERS[1:],
                                client_pb2.X509_ENTRY)])
        result = report.report()
        self.assertEqual(len(sum(result.values(), [])), 0)

        certs = report.certs()
        self.assertEqual(len(certs), 1)

        issuer_cert = cert_desc.from_cert(cert.Certificate(CHAIN_DERS[1]))
        root_cert = cert_desc.from_cert(cert.Certificate(CHAIN_DERS[2]))

        self.assertEqual(readable_dn(certs[0].issuer),
                         'C=US,O=Google Inc,CN=Google Internet Authority')
        self.assertEqual(readable_dn(certs[0].root_issuer),
                         'C=US,O=Equifax,OU=Equifax Secure Certificate Authority')

    def test_chain_containing_only_root_handled(self):
        report = self.CertificateReportBase([])
        report.scan_der_certs([(0, SELF_SIGNED_ROOT_DER, [], client_pb2.X509_ENTRY)])
        result = report.report()
        self.assertEqual(len(sum(result.values(), [])), 0)

        certs = report.certs()
        self.assertEqual(len(certs), 1)
        self.assertEquals(certs[0].entry_type, client_pb2.X509_ENTRY)

    def test_issuer_public_key_populated_from_chain(self):
        # Verify the test data is what is expected for this unit test.
        self.assertEqual(3, len(CHAIN_DERS))
        self.assertEqual(
            cert.Certificate(CHAIN_DERS[1]).key_hash(hashfunc="sha256").encode('hex'),
            'b6b95432abae57fe020cb2b74f4f9f9173c8c708afc9e732ace23279047c6d05')

        report = self.CertificateReportBase([])
        report.scan_der_certs([(0, CHAIN_DERS[0], CHAIN_DERS[1:],
                                client_pb2.X509_ENTRY)])
        result = report.report()
        self.assertEqual(len(sum(result.values(), [])), 0)

        certs = report.certs()
        self.assertEqual(len(certs), 1)
        self.assertEqual(certs[0].issuer_pk_sha256_hash.encode('hex'),
            'b6b95432abae57fe020cb2b74f4f9f9173c8c708afc9e732ace23279047c6d05')

    def test_if_scan_der_cert_check_raises_only_that_cert_skipped(self):
        certs = [(0, STRICT_DER, [], client_pb2.X509_ENTRY),
                 (1, STRICT_DER, [], client_pb2.X509_ENTRY)]

        report = self.CertificateReportBase([BadCheck()])
        report.scan_der_certs(certs)
        report.report()
        self.assertEqual(len(report.certs()), 1)

if __name__ == '__main__':
    sys.argv = gflags.FLAGS(sys.argv)
    unittest.main()
