from observation import Observation

class SerialNumberObservation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(SerialNumberObservation, self).__init__(
                "Serial number: " + description, *args, **kwargs)


class Negative(SerialNumberObservation):
    def __init__(self, details):
        super(Negative, self).__init__("is negative", details=details)


class CheckNegativeSerialNumber(object):
    @staticmethod
    def check(certificate):
        """Checks whether serial number is negative.

        Returns:
            array containing Negative or None.
        """
        serial_number = certificate.serial_number()
        if serial_number.value < 0:
            return [Negative(details=serial_number)]
