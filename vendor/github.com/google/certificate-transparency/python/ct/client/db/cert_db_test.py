import abc
import hashlib
from ct.proto import certificate_pb2

def name_attribute(name):
    att = certificate_pb2.DNAttribute()
    att.type, att.value = "CN", name
    return att

SAMPLE_CERT = certificate_pb2.X509Description()
SAMPLE_CERT.der = "hello\x00"
SAMPLE_CERT.subject.extend([name_attribute("com.example")])
SAMPLE_CERT.sha256_hash = hashlib.sha256("hello\x00").digest()
FOUR_CERTS = []
for i in range(0, 4):
    tmp = certificate_pb2.X509Description()
    tmp.der = "hello-%d" % i
    tmp.subject.extend([name_attribute("com.domain-%d" % i)])
    tmp.sha256_hash = hashlib.sha256("hello-%d" % i).digest()
    FOUR_CERTS.append((tmp, i))
# This class provides common tests for all cert database implementations.
# It only inherits from object so that unittest won't attempt to run the test_*
# methods on this class. Derived classes should use multiple inheritance
# from CertDBTest and unittest.TestCase to get test automation.
class CertDBTest(object):
    """All cert database tests should derive from this class as well as
    unittest.TestCase."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def db(self):
        """Derived classes must override to initialize a database."""
        pass

    def test_store_lookup_cert(self):
        self.db().store_cert_desc(SAMPLE_CERT, 0, 0)
        sha256_hash = hashlib.sha256("hello\x00").digest()
        cert = self.db().get_cert_by_sha256_hash(sha256_hash)
        self.assertEqual(SAMPLE_CERT.der, cert)

    def test_lookup_returns_none_if_no_match(self):
        self.db().store_cert_desc(SAMPLE_CERT, 0, 0)
        sha256_hash = hashlib.sha256("bye").digest()
        self.assertIsNone(self.db().get_cert_by_sha256_hash(sha256_hash))

    def test_store_cert_desc_ignores_duplicate(self):
        self.db().store_cert_desc(SAMPLE_CERT, 0, 0)
        sha256_hash = hashlib.sha256("hello\x00").digest()
        cert = self.db().get_cert_by_sha256_hash(sha256_hash)
        self.assertEqual("hello\x00", cert)

        # Store again
        hello2 = certificate_pb2.X509Description()
        hello2.der = SAMPLE_CERT.der
        hello2.subject.extend([name_attribute("com.example2")])
        hello2.sha256_hash = sha256_hash
        self.db().store_cert_desc(hello2, 1, 0)
        certs = [c for c in self.db().scan_certs_by_subject("example.com")]
        self.assertEqual(1, len(certs))
        self.assertEqual(SAMPLE_CERT.der, certs[0])
        certs = [c for c in self.db().scan_certs_by_subject("example2.com")]
        self.assertEqual(0, len(certs))

    def test_store_cert_descs_stores_all(self):
        self.db().store_certs_desc(FOUR_CERTS, 0)

        for i in range(4):
            sha256_hash = hashlib.sha256("hello-%d" % i).digest()
            cert = self.db().get_cert_by_sha256_hash(sha256_hash)
            self.assertEqual("hello-%d" % i, cert)

    def test_scan_certs_finds_all_certs(self):
        self.db().store_certs_desc(FOUR_CERTS, 0)

        certs = {c for c in self.db().scan_certs()}
        self.assertEqual(4, len(certs))
        for i in range(4):
            self.assertTrue("hello-%d" % i in certs)

    def test_scan_certs_honours_limit(self):
        self.db().store_certs_desc(FOUR_CERTS, 0)

        certs = {c for c in self.db().scan_certs(limit=2)}
        self.assertEqual(2, len(certs))

    def test_scan_certs_by_subject_finds_cert(self):
        self.db().store_certs_desc(FOUR_CERTS, 0)

        for i in range(4):
            matches = [c for c in self.db().scan_certs_by_subject(
                    "domain-%d.com" % i)]
            self.assertEqual(1, len(matches))
            self.assertEqual("hello-%d" % i, matches[0])

    def test_scan_certs_by_subject_honours_limit(self):
        for i in range(0, 4):
            cert = certificate_pb2.X509Description()
            cert.der = "hello-%d" % i
            cert.subject.extend([name_attribute("com.domain")])
            cert.sha256_hash = str(i)
            self.db().store_cert_desc(cert, i, 0)

        matches = {c for c in self.db().scan_certs_by_subject("domain.com",
                                                              limit=2)}
        self.assertEqual(2, len(matches))

    def test_scan_certs_by_subject_finds_by_common_name(self):
        for i in range(4):
            cert = certificate_pb2.X509Description()
            cert.der = "hello-%d" % i
            cert.subject.extend([name_attribute("Trusty CA %d" % i)])
            cert.sha256_hash = str(i)
            self.db().store_cert_desc(cert, i, 0)

        for i in range(4):
            matches = [c for c in self.db().scan_certs_by_subject(
                                                            "Trusty CA %d" % i)]
            self.assertEqual(1, len(matches))
            self.assertEqual("hello-%d" % i, matches[0])

    def test_scan_certs_by_subject_finds_by_all_names(self):
        cert = certificate_pb2.X509Description()
        cert.der = "hello"
        cert.subject.extend([name_attribute("com.%d" % i) for i in range(4)])
        cert.sha256_hash = 'asdf'
        self.db().store_cert_desc(cert, 0, 0)

        for i in range(4):
            matches = [c for c in self.db().scan_certs_by_subject("%d.com" % i)]
            self.assertEqual(1, len(matches))
            self.assertEqual("hello", matches[0])

    def test_scan_certs_by_subject_finds_by_prefix(self):
        for i in range(4):
            cert = certificate_pb2.X509Description()
            cert.der = "hello-%d" % i
            cert.subject.extend([name_attribute("com.example" + (".%d" % i)*i)])
            cert.sha256_hash = str(i)
            self.db().store_cert_desc(cert, i, 0)

        matches = {c for c in self.db().scan_certs_by_subject("example.com")}
        self.assertEqual(4, len(matches))
        for i in range(4):
            self.assertTrue("hello-%d" % i in matches)

    def test_scan_certs_by_subject_ignores_longer(self):
        cert = certificate_pb2.X509Description()
        cert.der = "hello"
        cert.subject.extend([name_attribute("example.com")])
        self.db().store_cert_desc(cert, 0, 0)

        matches = {c for c in self.db().scan_certs_by_subject(
                "mail.example.com")}
        self.assertEqual(0, len(matches))

    def test_scan_certs_by_subject_finds_by_literal_wildcard(self):
        cert = certificate_pb2.X509Description()
        cert.der = "hello-wild"
        cert.subject.extend([name_attribute("com.example.*")])
        cert.sha256_hash = 'cert'
        self.db().store_cert_desc(cert, 0, 0)
        cert2 = certificate_pb2.X509Description()
        cert2.der = "hello"
        cert2.subject.extend([name_attribute("com.example.www")])
        cert.sha256_hash = 'cert2'
        self.db().store_cert_desc(cert2, 1, 0)

        matches = [c for c in self.db().scan_certs_by_subject(
                "*.example.com")]
        self.assertEqual(1, len(matches))
        self.assertEqual("hello-wild", matches[0])

    def test_scan_certs_by_subject_ignores_wildcard_cert(self):
        for i in range(4):
            cert = certificate_pb2.X509Description()
            cert.der = "hello-%d" % i
            cert.subject.extend([name_attribute("com.example.%d" % i)])
            cert.sha256_hash = str(i)
            self.db().store_cert_desc(cert, i, 0)
        cert = certificate_pb2.X509Description()
        cert.der = "hello-wild"
        cert.subject.extend([name_attribute("com.example.*")])
        cert.sha256_hash = 'asdf'
        self.db().store_cert_desc(cert, 0, 0)

        matches = {c for c in self.db().scan_certs_by_subject(
                "2.example.com")}
        self.assertEqual(1, len(matches))
        self.assertTrue("hello-2" in matches)
