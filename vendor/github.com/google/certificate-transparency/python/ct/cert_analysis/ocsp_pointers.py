from ct.cert_analysis.observation import Observation
from ct.crypto import cert

class OcspObservation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(OcspObservation, self).__init__("OCSP: " + description, *args,
                                              **kwargs)


class LackOfOcsp(OcspObservation):
    def __init__(self):
        super(LackOfOcsp, self).__init__("no pointers")


class CorruptAiaExtension(OcspObservation):
    def __init__(self):
        super(CorruptAiaExtension, self).__init__("corrupt extension")


class MultipleOcspExtensions(OcspObservation):
    def __init__(self):
        super(MultipleOcspExtensions, self).__init__("mutlitple extensions")

#TODO(laiqu) Check whether extension is critical.
class CheckOcspExistence(object):
    """According to Baseline Requirements for the Issuance and Managment of
    Publicly-Trusted Certificates, v1.2.3. (13.2.5) certificates should contain
    OCSP endpoints"""
    @staticmethod
    def check(certificate):
        try:
            if len(certificate.ocsp_responders()) == 0:
                return [LackOfOcsp()]
        except cert.CertificateError:
            pass


class CheckCorruptOrMultipleAiaExtension(object):
    @staticmethod
    def check(certificate):
        try:
            certificate.ocsp_responders()
        except cert.CertificateError as e:
            if "multiple" in str(e).lower():
                return [MultipleOcspExtensions()]
            else:
                return [CorruptAiaExtension()]
