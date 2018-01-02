import calendar
import hashlib
import re
import unicodedata

from ct.crypto import cert
from ct.crypto.asn1 import x509_common
from ct.proto import certificate_pb2


def from_cert(certificate, observations=[]):
    """Pulls out interesting fields from certificate, so format of data will
    be similar in every database implementation."""
    proto = certificate_pb2.X509Description()
    proto.der = certificate.to_der()
    try:
        for sub in [(type_.short_name,
                     to_unicode('.'.join(process_name(value.human_readable()))))
                    for type_, value in certificate.subject()]:
            proto_sub = proto.subject.add()
            proto_sub.type, proto_sub.value = sub
    except cert.CertificateError:
        pass

    try:
        for iss in [(type_.short_name,
                     to_unicode('.'.join(process_name(value.human_readable()))))
                    for type_, value in certificate.issuer()]:
            proto_iss = proto.issuer.add()
            proto_iss.type, proto_iss.value = iss
    except cert.CertificateError:
        pass

    try:
        for alt in certificate.subject_alternative_names():
            proto_alt = proto.subject_alternative_names.add()
            proto_alt.type, proto_alt.value = (alt.component_key(),
                                               to_unicode('.'.join(process_name(
                                      alt.component_value().human_readable()))))
    except cert.CertificateError:
        pass

    try:
        proto.version = str(certificate.version())
    except cert.CertificateError:
        pass

    try:
        proto.serial_number = str(certificate.serial_number().human_readable()
                                  .upper().replace(':', ''))
    except cert.CertificateError:
        pass

    try:
        tbs_alg = certificate.signature()["algorithm"]
        if tbs_alg:
            proto.tbs_signature.algorithm_id = tbs_alg.long_name

        tbs_params = certificate.signature()["parameters"]
        if tbs_params:
            proto.tbs_signature.parameters = tbs_params.value

        cert_alg = certificate.signature_algorithm()["algorithm"]
        if cert_alg:
            proto.cert_signature.algorithm_id = cert_alg.long_name

        cert_params = certificate.signature_algorithm()["parameters"]
        if cert_params:
            proto.cert_signature.parameters = cert_params.value
    except cert.CertificateError:
        pass

    try:
        proto.basic_constraint_ca = bool(certificate.basic_constraint_ca())
    except cert.CertificateError:
        pass

    try:
        proto.validity.not_before, proto.validity.not_after = (
            1000 * int(calendar.timegm(certificate.not_before())),
            1000 * int(calendar.timegm(certificate.not_after())))
    except cert.CertificateError:
        pass

    proto.sha256_hash = hashlib.sha256(proto.der).digest()

    for observation in observations:
        proto_obs = proto.observations.add()
        if observation.description:
            proto_obs.description = observation.description
        if observation.reason:
            proto_obs.reason = observation.reason
        proto_obs.details = observation.details_to_proto()

    return proto


def to_unicode(value):
    encoded = unicode(value, 'utf-8', 'replace')
    for ch in encoded:
        try:
            _ = unicodedata.name(ch)
        except ValueError:
            # Mangled Unicode code-point. Perhaps this is just
            # plain ISO-8859-1 data incorrectly reported as UTF-8.
            return unicode(value, 'iso-8859-1', 'replace')
    return encoded


def process_name(subject, reverse=True):
    # RFCs for DNS names: RFC 1034 (sect. 3.5), RFC 1123 (sect. 2.1);
    # for common names: RFC 5280.
    # However we probably do not care about full RFC compliance here
    # (e.g. we ignore that a compliant label cannot begin with a hyphen,
    # we accept multi-wildcard names, etc.).
    #
    # For now, make indexing work for the common case:
    # allow letter-digit-hyphen, as well as wildcards (RFC 2818).
    forbidden = re.compile(r"[^a-z\d\-\*]")
    labels = subject.lower().split(".")
    valid_dns_name = len(labels) > 1 and all(
        map(lambda x: len(x) and not forbidden.search(x), labels))

    if valid_dns_name:
        # ["com", "example", "*"], ["com", "example", "mail"],
        # ["localhost"], etc.
        return list(reversed(labels)) if reverse else labels

    else:
        # ["John Smith"], ["Trustworthy Certificate Authority"],
        # ["google.com\x00"], etc.
        # TODO(ekasper): figure out what to do (use stringprep as specified
        # by RFC 5280?) to properly handle non-letter-digit-hyphen names.
        return [subject]
