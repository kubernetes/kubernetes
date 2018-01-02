import calendar
import datetime
import time

from ct.crypto import cert
from observation import Observation

class ValidityObservation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(ValidityObservation, self).__init__(
                "Validity: " + description, *args, **kwargs)


class NotBeforeInFuture(ValidityObservation):
    def __init__(self, details):
        super(NotBeforeInFuture, self).__init__("not before is in future",
                                                details=details)

    def _format_details(self):
        return time.strftime("%Y-%m-%dT%H:%M:%S%Z", self.details)


class NotAfterNotWellDefined(ValidityObservation):
    def __init__(self):
        super(NotAfterNotWellDefined, self).__init__(
                "not after not well defined")

class NotBeforeCorrupt(ValidityObservation):
    def __init__(self):
        super(NotBeforeCorrupt, self).__init__("notBefore value corrupt")


class NotAfterCorrupt(ValidityObservation):
    def __init__(self):
        super(NotAfterCorrupt, self).__init__("notAfter value corrupt")


class CheckValidityNotBeforeFuture(object):
    @staticmethod
    def check(certificate):
        """Checks validity field for notBefore in the future.

        Returns:
            array containing NotBeforeInFuture or None
        """
        try:
            not_before = certificate.not_before()
            now = datetime.datetime.utcnow()

            if calendar.timegm(not_before) - calendar.timegm(now.utctimetuple()) > 0:
                return [NotBeforeInFuture(details=not_before)]
        except cert.CertificateError:
            pass


class CheckValidityCorrupt(object):
    @staticmethod
    def check(certificate):
        """Checks if notBefore or notAfter field is corrupt.

        Returns:
            array containing NotBeforeCorrupt, NotAfterCorrupt or empty array
        """
        ret = []
        try:
            certificate.not_before()
        except cert.CertificateError:
            ret.append(NotBeforeCorrupt())
        try:
            certificate.not_after()
        except cert.CertificateError:
            ret.append(NotAfterCorrupt())
        return ret


class CheckIsExpirationDateWellDefined(object):
    @staticmethod
    def check(certificate):
        """Checks if notAfter fields is 9999123123595Z GeneralizedTime, which
        means that expiration time is not well-defined.

        Returns:
            array containing NotAfterNotWellDefined or None
        """
        if not certificate.is_not_after_well_defined():
            return [NotAfterNotWellDefined()]
