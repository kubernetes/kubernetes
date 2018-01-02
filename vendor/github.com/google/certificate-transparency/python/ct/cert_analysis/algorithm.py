import calendar
import time

from ct.crypto import cert
from ct.crypto.asn1 import oid
from ct.cert_analysis.observation import Observation

class AlgorithmObservation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(AlgorithmObservation, self).__init__(
                "Signature algorithm: " + description, *args, **kwargs)


class AlgorithmMismatch(AlgorithmObservation):
    def __init__(self, details):
        super(AlgorithmMismatch, self).__init__(
                "signature algorithm identifier in Certificate is different ",
                details=details)


class SHA1Observation(AlgorithmObservation):
    def _format_details(self):
        # Times are in UTC, so use "UTC" instead of %Z (which gives local TZ)
        return time.strftime("%Y-%m-%dT%H:%M:%SUTC", self.details)


class RsaSHA1(SHA1Observation):
    def __init__(self, details):
        super(RsaSHA1, self).__init__(
                "uses sha-1WithRSAEncryption after 1 Jan 2017", details=details)


class DsaSHA1(SHA1Observation):
    def __init__(self, details):
        super(DsaSHA1, self).__init__("uses id-dsa-with-SHA1 after 1 Jan 2017",
                                      details=details)


class EcdsaSHA1(SHA1Observation):
    def __init__(self, details):
        super(EcdsaSHA1, self).__init__("uses ecdsa-with-SHA1 after 1 Jan 2017",
                                        details=details)


class CheckSignatureAlgorithmsMismatch(object):
    @staticmethod
    def check(certificate):
        """Checks if signatureAlgorithm matches signature.

        Returns:
            array containing AlgorithmMismatch or nothing
        """
        signature = certificate.signature()
        signature_algorithm = certificate.signature_algorithm()
        if signature != signature_algorithm:
            return [AlgorithmMismatch((signature, signature_algorithm))]


def check_sha1_after_2017(not_after, algorithm):
    invalid_after = time.strptime("2017-01-01T00:00:00UTC",
                                  "%Y-%m-%dT%H:%M:%S%Z")
    if calendar.timegm(not_after) - calendar.timegm(invalid_after) >= 0:
        if algorithm == oid.SHA1_WITH_RSA_ENCRYPTION:
            return RsaSHA1(not_after)
        elif algorithm == oid.ID_DSA_WITH_SHA1:
            return DsaSHA1(not_after)
        elif algorithm == oid.ECDSA_WITH_SHA1:
            return EcdsaSHA1(not_after)


# naming of this object and the next is quite awkward, because one refers to
# certificate->tbsCertificate->signatureAlgorithm
# and the next class refers to
# certificate->signatureAlgorithm
class CheckTbsCertificateAlgorithmSHA1Ater2017(object):
    @staticmethod
    def check(certificate):
        """Checks if certificate is valid after 1 January 2017 and it is using
        SHA1 in tbsCertificate->signatureAlgorithm.

        Returns:
            array containing SHA1 observation or empty array
        """
        signature_algorithm = certificate.signature_algorithm()
        # Check for sha-1 ending after 1 January 2017
        try:
            not_after = certificate.not_after()
            check = check_sha1_after_2017(not_after,
                     signature_algorithm["algorithm"])
            if check:
                check.reason = "TbsCertificate"
                return [check]
        # if something goes wrong it should be caught by using validity checks
        # (CheckValidityCorrupt) not algorithm checks.
        except cert.CertificateError:
            pass


class CheckCertificateAlgorithmSHA1After2017(object):
    @staticmethod
    def check(certificate):
        """Checks if certificate is valid after 1 January 2017 and it is using
        SHA1 in signatureAlgorithm.

        Returns:
            array containing SHA1 observation or empty array
        """
        signature = certificate.signature()
        # Check for sha-1 ending after 1 January 2017
        try:
            not_after = certificate.not_after()
            check = check_sha1_after_2017(not_after,
                                 signature["algorithm"])
            if check:
                check.reason = "Certificate"
                return [check]
        # if not_after raises it should be caught using validity checks
        # not algorithm checks.
        except cert.CertificateError:
            pass
