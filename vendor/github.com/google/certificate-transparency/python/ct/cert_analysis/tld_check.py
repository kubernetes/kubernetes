from ct.cert_analysis.observation import Observation
from ct.cert_analysis import tld_list

class TldCheckObservation(Observation):
    def __init__(self, description, prefix=None, *args, **kwargs):
        super(TldCheckObservation, self).__init__(
                ("Top level domain: " if not prefix else prefix) + description,
                *args, **kwargs)

class TldMatchesBothUnicodeAndIdna(TldCheckObservation):
    def __init__(self, *args, **kwargs):
        super(TldMatchesBothUnicodeAndIdna, self).__init__("unicode and idna"
                "encoding of address matches different top level domain names",
                                                           *args, **kwargs)


class NoTldMatch(TldCheckObservation):
    def __init__(self, *args, **kwargs):
        super(NoTldMatch, self).__init__(
                "no top level domain matches", *args, **kwargs)


class NotAnAddress(TldCheckObservation):
    def __init__(self, *args, **kwargs):
        super(NotAnAddress, self).__init__("not an address", *args, **kwargs)


class NonUnicodeAddress(TldCheckObservation):
    def __init__(self, *args, **kwargs):
        super(NonUnicodeAddress, self).__init__("non unicode address", *args,
                                                **kwargs)


class GenericWildcard(TldCheckObservation):
    def __init__(self, *args, **kwargs):
        super(GenericWildcard, self).__init__("name wildcard matches top level "
                                              "domain name", *args, **kwargs)


class CheckTldMatches(object):
    TLD_LIST_ = None
    @classmethod
    def get_tld_list(cls):
        if not cls.TLD_LIST_:
            cls.TLD_LIST_ = tld_list.TLDList()
        return cls.TLD_LIST_

    @classmethod
    def check(cls, names, prefix=None):
        # This check is different from others, because it's supposed to be used
        # by other checks (common_name and dnsnames). The code for this check
        # would be the same in common_name and dnsnames, but resulting
        # observations should have different descriptions. This check still can
        # live on it's own if list of addresses is passed instead of
        # certificate. If prefix is provided, it's attached to descriptions of
        # observations.
        observations = []
        for name in names:
            name = name.value
            try:
                tld_match, idna_match, unicode_fail = (
                        cls.get_tld_list().match_certificate_name(name))
            except ValueError:
                observations += [NotAnAddress(details=name, prefix=prefix)]
                continue
            if unicode_fail:
                observations += [NonUnicodeAddress(details=name, prefix=prefix)]
            if tld_match and idna_match and tld_match != idna_match:
                observations += [TldMatchesBothUnicodeAndIdna(
                                    details=(name, tld_match, idna_match),
                                    prefix=prefix)]
            if not (tld_match or idna_match):
                observations += [NoTldMatch(details=(name), prefix=prefix)]
            # Check for generic wildcard
            if name.startswith('*.'):
                name_without_wildcard = name[2:]
                tld_match, idna_match, _ = cls.get_tld_list().match_certificate_name(
                                               name_without_wildcard)
                if (tld_match == name_without_wildcard or
                    idna_match == name_without_wildcard):
                    observations += [GenericWildcard(details=(name,
                                    tld_match if tld_match else idna_match),
                                                     prefix=prefix)]
        return observations
