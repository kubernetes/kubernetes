import abc
import hashlib


class CertDB(object):
    """Database interface for storing X509 certificate information."""
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def sha256_hash(der_cert):
        return hashlib.sha256(der_cert).digest()

    @abc.abstractmethod
    def store_cert_desc(self, cert_desc, index, log_key):
        """Stores a certificate using its description.

        Args:
            cert:          CertificateDescription
            index:         position in log
            log_key:       log id in LogDB"""

    @abc.abstractmethod
    def store_certs_desc(self, certs, log_key):
        """Store certificates using its descriptions.

        Batched version of store_cert_desc.

        Args:
            certs:         iterable of (CertificateDescription, index) tuples
            log_key:       log id in LogDB"""

    @abc.abstractmethod
    def get_cert_by_sha256_hash(self, cert_sha256_hash):
        """Fetch a certificate with a matching SHA256 hash
        Args:
            cert_sha256_hash: the SHA256 hash of the certificate
        Returns:
            A DER-encoded certificate, or None if the cert is not found."""

    @abc.abstractmethod
    def scan_certs(self, limit=0):
        """Scan all certificates.
        Args:
            limit:        maximum number of entries to yield. Default is no
                          limit.
        Yields:
            DER-encoded certificates."""

    @abc.abstractmethod
    def scan_certs_by_subject(self, subject_name, limit=0):
        """Scan certificates matching a subject name.
        Args:
            subject_name: a subject name, usually a domain. A scan for
                          example.com returns certificates for www.example.com,
                          *.example.com, test.mail.example.com, etc. Similarly
                          'com' can be used to look for all .com certificates.
                          Wildcards are treated as literal characters: a search
                          for *.example.com returns certificates for
                          *.example.com but not for mail.example.com and vice
                          versa.
                          Name may also be a common name rather than a DNS name,
                          e.g., "Trustworthy Certificate Authority".
            limit:        maximum number of entries to yield. Default is no
                          limit.
        Yields:
            DER-encoded certificates."""
