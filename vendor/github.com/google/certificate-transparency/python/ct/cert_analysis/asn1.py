from ct.cert_analysis.observation import Observation


class ASN1Observation(Observation):
    def __init__(self, description, *args, **kwargs):
        super(ASN1Observation, self).__init__("ASN.1: " + description, *args,
                                              **kwargs)


class All(ASN1Observation):
    def __init__(self):
        super(All, self).__init__("fails non strict test")


class Strict(ASN1Observation):
    def __init__(self, reason, details=None):
        super(Strict, self).__init__("fails strict test", reason=reason,
                                     details=details)


class MultipleExtensions(ASN1Observation):
    def __init__(self):
        super(MultipleExtensions, self).__init__("multiple extensions")
