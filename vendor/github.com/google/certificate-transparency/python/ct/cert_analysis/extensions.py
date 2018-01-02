from ct.crypto import cert
from ct.crypto.asn1 import oid
from ct.cert_analysis.observation import Observation

# From BR v1.2.3
# (extension_oid, critical?), if crtical is None then it means it wasn't
# specified in BR.
_CA_EXTENSIONS = [(oid.ID_CE_BASIC_CONSTRAINTS, True),
                  (oid.ID_CE_KEY_USAGE, True),]


_CA_PROHIBITED_EXTENSIONS = [oid.ID_CE_CERTIFICATE_POLICIES,
                             oid.ID_CE_EXT_KEY_USAGE,]


_SUBORDINATE_EXTENSIONS = [(oid.ID_CE_CERTIFICATE_POLICIES, False),
                           (oid.ID_CE_CRL_DISTRIBUTION_POINTS, False),
                           (oid.ID_PE_AUTHORITY_INFO_ACCESS, False),
                           (oid.ID_CE_BASIC_CONSTRAINTS, True),
                           (oid.ID_CE_KEY_USAGE, True),]


_SUBORDINATE_OPTIONAL = [(oid.ID_CE_NAME_CONSTRAINTS, True),
                         (oid.ID_CE_EXT_KEY_USAGE, False),]


_SUBSCRIBER_EXTENSIONS = [(oid.ID_CE_CERTIFICATE_POLICIES, False),
                          (oid.ID_PE_AUTHORITY_INFO_ACCESS, False),
                          (oid.ID_CE_EXT_KEY_USAGE, None),]


_SUBSCRIBER_OPTIONAL = [(oid.ID_CE_CRL_DISTRIBUTION_POINTS, False),
                        (oid.ID_CE_BASIC_CONSTRAINTS, None),
                        (oid.ID_CE_KEY_USAGE, None),]


_SUBSCRIBER = 0
_SUBORDINATE = 1
_ROOT = 2


class ExtensionObservation(Observation):
    def __init__(self, description, cert_type, *args, **kwargs):
        super(ExtensionObservation, self).__init__(
                "Extensions: %s - %s" % (self._type_to_string(cert_type),
                                         description) , *args, **kwargs)

    @staticmethod
    def _type_to_string(type_):
        if type_ == _SUBSCRIBER:
            return "subscriber certificate"
        elif type_ == _SUBORDINATE:
            return "subordinate certificate"
        elif type_ == _ROOT:
            return "root certificate"


class LackOfRequiredExtension(ExtensionObservation):
    def __init__(self, cert_type, reason):
        """Required extension is missing.

        Args:
            cert_type: certificate type, one of _ROOT, _SUBORDINATE, _SUBSCRIBER
            reason:    string representing extension OID
        """
        super(LackOfRequiredExtension, self).__init__(
                "required extension is missing", cert_type, reason)

class ContainsProhibitedExtension(ExtensionObservation):
    def __init__(self, cert_type, reason):
        """Contains prohibited extension.

        Args:
            cert_type: certificate type, one of _ROOT, _SUBORDINATE, _SUBSCRIBER
            reason:    string representing extension OID
        """
        super(ContainsProhibitedExtension, self).__init__(
                "contains prohibited extension", cert_type, reason)


class WrongCritical(ExtensionObservation):
    def __init__(self, cert_type, reason, details):
        """Critical field is set to wrong value.

        Args:
            cert_type: certificate type, one of _ROOT, _SUBORDINATE, _SUBSCRIBER
            reason:    string representing extension OID
            details:   boolean representing extension critical field
        """
        super(WrongCritical, self).__init__(
                "certificate extension 'critical' field is set to wrong value",
                cert_type, reason, details)

        def _format_details():
            return "%s instead of %s" % (self.details, not self.details)

def _oid_to_string(oid_):
    return '.'.join([str(val) for val in oid_.value])


class CheckCorrectExtensions(object):
    @classmethod
    def check(cls, certificate):
        required = _SUBSCRIBER_EXTENSIONS
        optional = _SUBSCRIBER_OPTIONAL
        prohibited = []
        cert_type = _SUBSCRIBER
        # Is it root certificate?
        if certificate.is_self_signed():
            required = _CA_EXTENSIONS
            optional = []
            prohibited = _CA_PROHIBITED_EXTENSIONS
            cert_type = _ROOT
        else:
            # Is it subordinate certificate?
            try:
                ca = certificate.basic_constraint_ca()
            except cert.CertificateError:
                # if it fails let's assume that it is leaf cert
                pass
            else:
                if ca and ca.value == True:
                    required = _SUBORDINATE_EXTENSIONS
                    optional = _SUBORDINATE_OPTIONAL
                    cert_type = _SUBORDINATE

        observations = []
        extensions = certificate.get_extensions()
        oid_crit = {ex['extnID']: ex['critical'] for ex in extensions}
        for req in required:
            crit = oid_crit.get(req[0], None)
            if crit == None:
                observations += [LackOfRequiredExtension(cert_type,
                                                         _oid_to_string(req[0]))]
            elif crit != req[1] and req[1] != None:
                observations += [WrongCritical(cert_type, _oid_to_string(req[0]),
                                               crit)]
        for opt in optional:
            if opt[1] == None:
                continue
            crit = oid_crit.get(opt[0], None)
            # if certificate contains optional extension, check if it has
            # correct critical.
            if crit != None and crit != opt[1]:
                observations += [WrongCritical(cert_type, _oid_to_string(opt[0]), crit)]
        for proh in prohibited:
            crit = oid_crit.get(proh, None)
            if crit != None:
                observations += [ContainsProhibitedExtension(cert_type,
                                                             _oid_to_string(proh))]
        return observations
