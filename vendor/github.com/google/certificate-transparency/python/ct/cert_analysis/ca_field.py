import re
from ct.crypto import cert
from ct.cert_analysis.observation import Observation

class ConstraintObservation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(ConstraintObservation, self).__init__(
                "Basic constraint: " + description, *args, **kwargs)


class CaTrue(ConstraintObservation):
    def __init__(self):
        super(CaTrue, self).__init__("CA is TRUE")


class CorruptOrMultiple(ConstraintObservation):
    def __init__(self):
        super(CorruptOrMultiple, self).__init__(
                "multiple extensions or corrupt")


class CheckCATrue(object):
    NOT_DOMAIN_NAME_REGEX = re.compile('[^a-zA-z0-9\-.*]')
    @staticmethod
    def check(certificate):
        """Checks if certificate CA field is set to TRUE and there is domain
        name in CN or certificate has SAN.

        Returns:
            array containing CaTrue or CorruptOrMultiple in case of
            problem with extension or empty array
        """
        try:
            bc = certificate.basic_constraint_ca()
            if bc and bc.value == True:
                try:
                    if certificate.subject_alternative_names():
                        return [CaTrue()]
                except cert.CertificateError():
                    pass
                try:
                    for name in certificate.subject_common_names():
                        if not CheckCATrue.NOT_DOMAIN_NAME_REGEX.search(name.value):
                            return [CaTrue()]
                except cert.CertificateError:
                    pass
        except cert.CertificateError:
            pass


class CheckCorruptCAField(object):
    """Checks if extension is corrupt or there are multiple extensions.

    Returns:
        array containing CorruptOrMultiple or None
    """
    @staticmethod
    def check(certificate):
        try:
            certificate.basic_constraint_ca()
        except cert.CertificateError:
            return [CorruptOrMultiple()]

