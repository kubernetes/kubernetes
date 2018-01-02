"""ASN.1 X509 specification."""

from ct.crypto.asn1 import types
from ct.crypto.asn1 import x509_common
from ct.crypto.asn1 import x509_extension
from ct.crypto.asn1 import x509_name
from ct.crypto.asn1 import x509_time


class TBSCertificate(types.Sequence):
    components = (
        (types.Component("version", x509_common.Version.explicit(0),
                         default=0)),
        (types.Component("serialNumber", x509_common.CertificateSerialNumber)),
        (types.Component("signature", x509_common.AlgorithmIdentifier)),
        (types.Component("issuer", x509_name.Name)),
        (types.Component("validity", x509_time.Validity)),
        (types.Component("subject", x509_name.Name)),
        (types.Component("subjectPublicKeyInfo",
                         x509_common.SubjectPublicKeyInfo)),
        (types.Component("issuerUniqueID",
                         x509_common.UniqueIdentifier.implicit(1),
                         optional=True)),
        (types.Component("subjectUniqueID",
                         x509_common.UniqueIdentifier.implicit(2),
                         optional=True)),
        (types.Component("extensions",
                         x509_extension.Extensions.explicit(3), optional=True))
        )


class Certificate(types.Sequence):
    components = (
        (types.Component("tbsCertificate", TBSCertificate)),
        (types.Component("signatureAlgorithm", x509_common.AlgorithmIdentifier)),
        (types.Component("signatureValue", types.BitString))
        )
