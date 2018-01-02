from ct.cert_analysis.observation import Observation
from ct.crypto import cert

class CrlObservation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(CrlObservation, self).__init__("CRL: " + description, *args,
                                              **kwargs)


class LackOfCrl(CrlObservation):
    def __init__(self):
        super(LackOfCrl, self).__init__("no pointers")


class CorruptCrlExtension(CrlObservation):
    def __init__(self):
        super(CorruptCrlExtension, self).__init__("corrupt extension")


class MultipleCrlExtensions(CrlObservation):
    def __init__(self):
        super(MultipleCrlExtensions, self).__init__("multiple extensions")

#TODO(laiqu) Check whether extension is critical
class CheckCrlExistence(object):
    """According to Baseline Requirements for the Issuance and Managment of
    Publicly-Trusted Certificates, v1.2.3. (13.2.2) certificates should contain
    CRL endpoints"""
    @staticmethod
    def check(certificate):
        try:
            if len(certificate.crl_distribution_points()) == 0:
                return [LackOfCrl()]
        except cert.CertificateError:
            pass


class CheckCorruptOrMultipleCrlExtension(object):
    @staticmethod
    def check(certificate):
        try:
            certificate.crl_distribution_points()
        except cert.CertificateError as e:
            if "multiple" in str(e).lower():
                return [MultipleCrlExtensions()]
            else:
                return [CorruptCrlExtension()]
