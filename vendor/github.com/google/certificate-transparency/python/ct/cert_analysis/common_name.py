from ct.cert_analysis.observation import Observation
from ct.cert_analysis import tld_check
from ct.crypto import cert


class CommonNameObservation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(CommonNameObservation, self).__init__("Subject common name: " +
                                                    description, *args, **kwargs)

class NoSubjectCommonName(CommonNameObservation):
    def __init__(self):
        super(NoSubjectCommonName, self).__init__("no subject common name")


class TldMatchesBothUnicodeAndIdna(CommonNameObservation):
    def __init__(self, *args, **kwargs):
        super(TldMatchesBothUnicodeAndIdna, self).__init__("unicode and idna"
                "encoding of address matches different top level domain names",
                                                           *args, **kwargs)


class NoTldMatch(CommonNameObservation):
    def __init__(self, *args, **kwargs):
        super(NoTldMatch, self).__init__(
                "no top level domain matches", *args, **kwargs)


class NotAnAddress(CommonNameObservation):
    def __init__(self, *args, **kwargs):
        super(NotAnAddress, self).__init__("not an address", *args, **kwargs)


class NonUnicodeAddress(CommonNameObservation):
    def __init__(self, *args, **kwargs):
        super(NonUnicodeAddress, self).__init__("non unicode address", *args,
                                                **kwargs)


class GenericWildcard(CommonNameObservation):
    def __init__(self, *args, **kwargs):
        super(GenericWildcard, self).__init__("name wildcard matches top level "
                                              "domain name", *args, **kwargs)


class CorruptSubjectCommonNames(CommonNameObservation):
    def __init__(self):
        super(CorruptSubjectCommonNames, self).__init__("corrupt")


class CheckCorruptSubjectCommonName(object):
    """Check if CN attribute is corrupt.

    Returns:
        array containing CorruptSubjectCommonNames or None"""
    @staticmethod
    def check(certificate):
        try:
            certificate.subject_common_names()
        except cert.CertificateError:
            return [CorruptSubjectCommonNames()]


class CheckLackOfSubjectCommonName(object):
    """Checks existence of subject common names.

    Returns:
        array containing NoSubjectCommonName"""
    @staticmethod
    def check(certificate):
        try:
            if len(certificate.subject_common_names()) == 0:
               return [NoSubjectCommonName()]
        except cert.CertificateError:
            pass


class CheckSCNTldMatches(object):
    """"Check whether common names matches some top level domain name.

    This will also return NoTldMatch if common name is not an address.
    Returns:
        array containing TldMatchesBothUnicodeAndIdna, NoTldMatch,
        GenericWildcard or None"""
    @classmethod
    def check(cls, certificate):
        try:
            return tld_check.CheckTldMatches.check(
                    certificate.subject_common_names(), "Subject common name: ")
        except cert.CertificateError:
            pass

