from ct.crypto import cert
from observation import Observation

class IpAddressObservation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(IpAddressObservation, self).__init__(
                "IPAddres: " + description, *args, **kwargs)


class IPv6(IpAddressObservation):
    def __init__(self):
        super(IPv6, self).__init__("IPv6")


class Private(IpAddressObservation):
    def __init__(self, details):
        super(Private, self).__init__(
                "IP address in IANA-reserved private range",
                details=details)


class CorruptIPAddress(IpAddressObservation):
    def __init__(self):
        super(CorruptIPAddress, self).__init__("corrupt extension")


class CheckPrivateIpAddresses(object):
    @staticmethod
    def check(certificate):
        """Checks if certificate contains IPv4 address in private range.

        Returns:
            array containing Private for each private address,
            or empty array there are no observations.
        """
        observations = []
        try:
            for address in certificate.subject_ip_addresses():
                octets = address.as_octets()
                if len(octets) == 16:
                    if octets[0] == ord('\xfd'):
                        observations += [Private(str(address))]
                else:
                    private = False
                    # check 10.0.0.0 to 10.255.255.255
                    if octets[0] == 10:
                        private = True
                    # check 172.16.0.0 to 172.31.255.255
                    elif (octets[0] == 172 and 16 <= octets[1] <= 31):
                        private = True
                    # check 192.168.0.0 to 192.168.255.255
                    elif (octets[0] == 192 and octets[1] == 168):
                        private = True
                    if private:
                        observations += [Private(str(address))]
        except cert.CertificateError:
            pass
        return observations


class CheckCorruptIpAddresses(object):
    @staticmethod
    def check(certificate):
        """Checks if ip addresses are corrupt in certificate.

        Returns:
            array containing CorruptIPAddress or None
        """
        try:
            certificate.subject_ip_addresses()
        except cert.CertificateError:
            return [CorruptIPAddress()]

