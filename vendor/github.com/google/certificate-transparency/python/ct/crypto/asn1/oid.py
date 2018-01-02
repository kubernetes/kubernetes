"""ASN.1 object identifiers. This module contains a dictionary of known OIDs.

See http://luca.ntop.org/Teaching/Appunti/asn1.html for a good introduction
to ASN.1.
"""

from ct.crypto import error
from ct.crypto.asn1 import tag
from ct.crypto.asn1 import types


@types.Universal(6, tag.PRIMITIVE)
class ObjectIdentifier(types.Simple):
    """Object identifier."""

    def _name(self, dict_idx):
        try:
            return _OID_NAME_DICT[self][dict_idx]
        except KeyError:
            # fall back to OID
            return ".".join(map(str, self._value))

    @property
    def short_name(self):
        """Return the short name representation of an OID."""
        return self._name(1)

    @property
    def long_name(self):
        """Return the long name representation of an OID."""
        return self._name(0)

    def __str__(self):
        return self.short_name

    @property
    def value(self):
        """The value of an OID is a tuple of integers."""
        return self._value

    @staticmethod
    def _encode_component(value):
        """Encode an OID component as a bytearray.

        Args:
            value: an integer component value

        Returns:
            a bytearray representing the encoded component.
        """
        int_bytes = bytearray()
        # Encode in base-128.
        # All bytes apart from the lsb have the high bit set.
        int_bytes.append(value & 0x7f)
        value >>= 7
        while value:
            int_bytes.append(value & 0x7f | 0x80)
            value >>= 7
        int_bytes.reverse()
        return int_bytes

    @classmethod
    def _read_component(cls, int_bytes):
        """Parse a single component from a non-empty bytearray.

        Args:
            int_bytes: a non-empty bytearray.

        Returns:
            a (component, rest) tuple with the decoded integer and the
                remaining bytes of the bytearray.
        """
        ret = 0
        i = 0
        while int_bytes[i] & 0x80:
            num = int_bytes[i] & 0x7f
            if not ret and not num:
                # The component must be encoded with as few digits as possible,
                # i.e., leading zeroes are not allowed. Since ASN.1 libraries
                # interpret leading 0x80-octets differently, this may be
                # indicative of an attempt to trick a browser into accepting a
                # certificate it shouldn't. See page 7 of
                # www.cosic.esat.kuleuven.be/publications/article-1432.pdf
                raise error.ASN1Error("Leading 0x80 octets in the base-128 "
                                      "encoding of  OID component")
            ret |= num
            ret <<= 7
            i += 1

        ret |= int_bytes[i]
        return ret, int_bytes[i+1:]

    def _encode_value(self):
        int_bytes = bytearray()
        # ASN.1 specifies that the first two components are encoded together
        # as c0*40 + c1.
        int_bytes += self._encode_component(self._value[0]*40 + self._value[1])
        for v in self._value[2:]:
            int_bytes += self._encode_component(v)
        return str(int_bytes)

    @classmethod
    def _convert_value(cls, value):
        if isinstance(value, ObjectIdentifier):
            return value.value
        else:
            if isinstance(value, str):
                value = [int(v) for v in value.split(".")]
            if len(value) < 2:
                raise ValueError("OID must have at least 2 components")
            if not all([v >= 0 for v in value]):
                raise ValueError("OID cannot have negative components")
            if value[0] > 2:
                raise ValueError("First OID component must be 0, 1 or 2, "
                                 "got %d" % value[0])
            if value[0] <= 1 and value[1] > 39:
                raise ValueError("Second OID component must be <= 39 if "
                                 "first component is <= 1; got %d, %d" %
                                 (value[0], value[1]))
            return tuple(value)

    @classmethod
    def _decode_value(cls, buf, strict=True):
        """Decode from a string or string buffer."""
        if buf in _OID_DECODING_DICT:
            return _OID_DECODING_DICT[buf]
        if not buf:
            raise error.ASN1Error("Invalid encoding")
        int_bytes = bytearray(buf)

        # Last byte can't have the high bit set.
        if int_bytes[-1] & 0x80:
            raise error.ASN1Error("Invalid encoding")

        components = []

        first, int_bytes = cls._read_component(int_bytes)
        if first < 40:
            components += [0, first]
        elif first < 80:
            components += [1, first - 40]
        else:
            components += [2, first - 80]

        while int_bytes:
            component, int_bytes = cls._read_component(int_bytes)
            components.append(component)

        return tuple(components)

###############################################################################
#                           Known object identifiers                          #
#                                                                             #
# If you add a new OID, make sure to also add its commonly known alias to     #
# _OID_NAME_DICT.                                                             #
###############################################################################

# Signature and public key algorithms
# RFC 3279
RSA_ENCRYPTION = ObjectIdentifier(value="1.2.840.113549.1.1.1")
MD2_WITH_RSA_ENCRYPTION = ObjectIdentifier(value="1.2.840.113549.1.1.2")
MD5_WITH_RSA_ENCRYPTION = ObjectIdentifier(value="1.2.840.113549.1.1.4")
SHA1_WITH_RSA_ENCRYPTION = ObjectIdentifier(value="1.2.840.113549.1.1.5")
ID_DSA = ObjectIdentifier(value="1.2.840.10040.4.1")
ID_DSA_WITH_SHA1 = ObjectIdentifier(value="1.2.840.10040.4.3")
ID_EC_PUBLICKEY = ObjectIdentifier(value="1.2.840.10045.2.1")
ECDSA_WITH_SHA1 = ObjectIdentifier(value="1.2.840.10045.4.1")
# RFC 5758
ECDSA_WITH_SHA224 = ObjectIdentifier(value="1.2.840.10045.4.3.1")
ECDSA_WITH_SHA256 = ObjectIdentifier(value="1.2.840.10045.4.3.2")
ECDSA_WITH_SHA384 = ObjectIdentifier(value="1.2.840.10045.4.3.3")
ECDSA_WITH_SHA512 = ObjectIdentifier(value="1.2.840.10045.4.3.4")
# RFC 4055
ID_RSASSA_PSS = ObjectIdentifier(value="1.2.840.113549.1.1.10")
ID_SHA256_WITH_RSA_ENCRYPTION = ObjectIdentifier(value="1.2.840.113549.1.1.11")
ID_SHA384_WITH_RSA_ENCRYPTION = ObjectIdentifier(value="1.2.840.113549.1.1.12")
ID_SHA512_WITH_RSA_ENCRYPTION = ObjectIdentifier(value="1.2.840.113549.1.1.13")
ID_SHA224_WITH_RSA_ENCRYPTION = ObjectIdentifier(value="1.2.840.113549.1.1.14")
# RFC 4491
ID_GOSTR3411_94_WITH_GOSTR3410_94 = ObjectIdentifier(value="1.2.643.2.2.4")
ID_GOSTR3411_94_WITH_GOSTR3410_2001 = ObjectIdentifier(value="1.2.643.2.2.3")
# RFC 5758
ID_DSA_WITH_SHA224 = ObjectIdentifier(value="2.16.840.1.101.3.4.3.1")
ID_DSA_WITH_SHA256 = ObjectIdentifier(value="2.16.840.1.101.3.4.3.2")
# Naming attributes (RFC 5280)
ID_AT_NAME = ObjectIdentifier(value="2.5.4.41")
ID_AT_SURNAME = ObjectIdentifier(value="2.5.4.4")
ID_AT_GIVEN_NAME = ObjectIdentifier(value="2.5.4.42")
ID_AT_INITIALS = ObjectIdentifier(value="2.5.4.43")
ID_AT_GENERATION_QUALIFIER = ObjectIdentifier(value="2.5.4.44")
ID_AT_COMMON_NAME = ObjectIdentifier(value="2.5.4.3")
ID_AT_LOCALITY_NAME = ObjectIdentifier(value="2.5.4.7")
ID_AT_STATE_OR_PROVINCE_NAME = ObjectIdentifier(value="2.5.4.8")
ID_AT_ORGANIZATION_NAME = ObjectIdentifier(value="2.5.4.10")
ID_AT_ORGANIZATIONAL_UNIT_NAME = ObjectIdentifier(value="2.5.4.11")
ID_AT_TITLE = ObjectIdentifier(value="2.5.4.12")
ID_AT_DN_QUALIFIER = ObjectIdentifier(value="2.5.4.46")
ID_AT_COUNTRY_NAME = ObjectIdentifier(value="2.5.4.6")
ID_AT_SERIAL_NUMBER = ObjectIdentifier(value="2.5.4.5")
ID_AT_PSEUDONYM = ObjectIdentifier(value="2.5.4.65")
ID_DOMAIN_COMPONENT = ObjectIdentifier(value="0.9.2342.19200300.100.1.25")
ID_EMAIL_ADDRESS = ObjectIdentifier(value="1.2.840.113549.1.9.1")

# Other naming attributes commonly found in certs
ID_AT_STREET_ADDRESS = ObjectIdentifier(value="2.5.4.9")
ID_AT_DESCRIPTION = ObjectIdentifier(value="2.5.4.13")
ID_AT_BUSINESS_CATEGORY = ObjectIdentifier(value="2.5.4.15")
ID_AT_POSTAL_CODE = ObjectIdentifier(value="2.5.4.17")
ID_AT_POST_OFFICE_BOX = ObjectIdentifier(value="2.5.4.18")

# Standard X509v3 certificate extensions
ID_CE_AUTHORITY_KEY_IDENTIFIER = ObjectIdentifier(value="2.5.29.35")
ID_CE_SUBJECT_KEY_IDENTIFIER = ObjectIdentifier(value="2.5.29.14")
ID_CE_KEY_USAGE = ObjectIdentifier(value="2.5.29.15")
ID_CE_PRIVATE_KEY_USAGE_PERIOD = ObjectIdentifier(value="2.5.29.16")
ID_CE_CERTIFICATE_POLICIES = ObjectIdentifier(value="2.5.29.32")
ID_CE_SUBJECT_ALT_NAME = ObjectIdentifier(value="2.5.29.17")
ID_CE_ISSUER_ALT_NAME = ObjectIdentifier(value="2.5.29.18")
ID_CE_SUBJECT_DIRECTORY_ATTRIBUTES = ObjectIdentifier(value="2.5.29.9")
ID_CE_BASIC_CONSTRAINTS = ObjectIdentifier(value="2.5.29.19")
ID_CE_NAME_CONSTRAINTS = ObjectIdentifier(value="2.5.29.30")
ID_CE_POLICY_CONSTRAINTS = ObjectIdentifier(value="2.5.29.30")
ID_CE_EXT_KEY_USAGE = ObjectIdentifier(value="2.5.29.37")
ID_CE_CRL_DISTRIBUTION_POINTS = ObjectIdentifier(value="2.5.29.31")
ID_CE_INHIBIT_ANY_POLICY = ObjectIdentifier(value="2.5.29.54")
ID_PE_AUTHORITY_INFO_ACCESS = ObjectIdentifier(value="1.3.6.1.5.5.7.1.1")
ID_PE_SUBJECT_INFO_ACCESS = ObjectIdentifier(value="1.3.6.1.5.5.7.1.11")

# RFC 3280 - Used in ExtendedKeyUsage extension
ID_KP_SERVER_AUTH = ObjectIdentifier(value="1.3.6.1.5.5.7.3.1")
ID_KP_CLIENT_AUTH = ObjectIdentifier(value="1.3.6.1.5.5.7.3.2")
ID_KP_CODE_SIGNING = ObjectIdentifier(value="1.3.6.1.5.5.7.3.3")
ID_KP_EMAIL_PROTECTION = ObjectIdentifier(value="1.3.6.1.5.5.7.3.4")
ID_KP_TIME_STAMPING = ObjectIdentifier(value="1.3.6.1.5.5.7.3.8")
ID_KP_OCSP_SIGNING = ObjectIdentifier(value="1.3.6.1.5.5.7.3.9")

# RFC 3280 - Used in Authority Info Access extension
ID_AD_OCSP = ObjectIdentifier(value="1.3.6.1.5.5.7.48.1")
ID_AD_CA_ISSUERS = ObjectIdentifier(value="1.3.6.1.5.5.7.48.2")

# Certificate Policy related OIDs
ID_QT_CPS = ObjectIdentifier(value="1.3.6.1.5.5.7.2.1")
ID_QT_UNOTICE = ObjectIdentifier(value="1.3.6.1.5.5.7.2.2")
ANY_POLICY = ObjectIdentifier(value="2.5.29.32.0")

# CT Specific
CT_EMBEDDED_SCT_LIST = ObjectIdentifier(value="1.3.6.1.4.1.11129.2.4.2")
CT_POISON = ObjectIdentifier(value="1.3.6.1.4.1.11129.2.4.3")
CT_PRECERTIFICATE_SIGNING = ObjectIdentifier(value="1.3.6.1.4.1.11129.2.4.4")

_OID_NAME_DICT = {
    # Object identifier long names taken verbatim from the RFCs.
    # Short names are colloquial.
    RSA_ENCRYPTION: ("rsaEncryption", "RSA"),
    MD2_WITH_RSA_ENCRYPTION: ("md2WithRSAEncryption", "RSA-MD2"),
    MD5_WITH_RSA_ENCRYPTION: ("md5WithRSAEncryption", "RSA-MD5"),
    SHA1_WITH_RSA_ENCRYPTION: ("sha-1WithRSAEncryption", "RSA-SHA1"),
    ID_DSA: ("id-dsa", "DSA"),
    ID_DSA_WITH_SHA1: ("id-dsa-with-sha1", "DSA-SHA1"),
    ID_EC_PUBLICKEY: ("id-ecPublicKey", "EC-PUBKEY"),
    ECDSA_WITH_SHA1: ("ecdsa-with-SHA1", "ECDSA-SHA1"),
    ECDSA_WITH_SHA224: ("ecdsa-with-SHA224", "ECDSA-SHA224"),
    ECDSA_WITH_SHA256: ("ecdsa-with-SHA256", "ECDSA-SHA256"),
    ECDSA_WITH_SHA384: ("ecdsa-with-SHA384", "ECDSA-SHA384"),
    ECDSA_WITH_SHA512: ("ecdsa-with-SHA512", "ECDSA-SHA512"),
    ID_RSASSA_PSS: ("id-RSASSA-PSS", "RSASSA-PSS"),
    ID_GOSTR3411_94_WITH_GOSTR3410_94: ("id-GostR3411-94-with-GostR3410-94",
                                        "GOST94"),
    ID_GOSTR3411_94_WITH_GOSTR3410_2001: ("id-GostR3411-94-with-GostR3410-2001",
                                          "GOST2001"),
    ID_SHA256_WITH_RSA_ENCRYPTION: ("sha256WithRSAEncryption", "RSA-SHA256"),
    ID_SHA384_WITH_RSA_ENCRYPTION: ("sha384WithRSAEncryption", "RSA-SHA384"),
    ID_SHA512_WITH_RSA_ENCRYPTION: ("sha512WithRSAEncryption", "RSA-SHA512"),
    ID_SHA224_WITH_RSA_ENCRYPTION: ("sha224WithRSAEncryption", "RSA-SHA224"),
    ID_DSA_WITH_SHA224: ("id-dsa-with-sha224", "DSA-SHA224"),
    ID_DSA_WITH_SHA256: ("id-dsa-with-sha256", "DSA-SHA256"),
    ID_AT_NAME: ("id-at-name", "name"),
    ID_AT_SURNAME: ("id-at-surname", "surname"),
    ID_AT_GIVEN_NAME: ("id-at-givenName", "givenName"),
    ID_AT_INITIALS: ("id-at-initials", "initials"),
    ID_AT_GENERATION_QUALIFIER: ("id-at-generationQualifier", "genQualifier"),
    ID_AT_COMMON_NAME: ("id-at-commonName", "CN"),
    ID_AT_LOCALITY_NAME: ("id-at-localityName", "L"),
    ID_AT_STATE_OR_PROVINCE_NAME: ("id-at-stateOrProvinceName", "ST"),
    ID_AT_ORGANIZATION_NAME: ("id-at-organizationName", "O"),
    ID_AT_ORGANIZATIONAL_UNIT_NAME: ("id-at-organizationalUnitName", "OU"),
    ID_AT_TITLE: ("id-at-title", "title"),
    ID_AT_DN_QUALIFIER: ("id-at-dnQualifier", "dnQualifier"),
    ID_AT_COUNTRY_NAME: ("id-at-countryName", "C"),
    ID_AT_SERIAL_NUMBER: ("id-at-serialNumber", "serialNumber"),
    ID_AT_PSEUDONYM: ("id-at-pseudonym", "pseudonym"),
    ID_DOMAIN_COMPONENT: ("id-domainComponent", "domainComponent"),
    ID_EMAIL_ADDRESS: ("id-emailAddress", "email"),
    ID_AT_STREET_ADDRESS: ("id-at-streetAddress", "streetAddress"),
    ID_AT_DESCRIPTION: ("id-at-description", "description"),
    ID_AT_BUSINESS_CATEGORY: ("id-at-businessCategory", "businessCategory"),
    ID_AT_POSTAL_CODE: ("id-at-postalCode", "postalCode"),
    ID_AT_POST_OFFICE_BOX: ("id-at-postOfficeBox", "postOfficeBox"),
    ID_CE_AUTHORITY_KEY_IDENTIFIER: ("id-ce-authorityKeyIdentifier",
                                     "authorityKeyIdentifier"),
    ID_CE_SUBJECT_KEY_IDENTIFIER: ("id-ce-subjectKeyIdentifier",
                                   "subjectKeyIdentifier"),
    ID_CE_KEY_USAGE: ("id-ce-keyUsage", "keyUsage"),
    ID_CE_PRIVATE_KEY_USAGE_PERIOD: ("id-ce-privateKeyUsagePeriod",
                                     "privateKeyUsagePeriod"),
    ID_CE_CERTIFICATE_POLICIES: ("id-ce-certificatePolicies",
                                 "certificatePolicies"),
    ID_CE_SUBJECT_ALT_NAME: ("id-ce-subjectAltName", "subjectAltName"),
    ID_CE_ISSUER_ALT_NAME: ("id-ce-issuerAltName", "issuerAltName"),
    ID_CE_SUBJECT_DIRECTORY_ATTRIBUTES: ("id-ce-subjectDirectoryAttributes",
                                         "subjectDirectoryAttributes"),
    ID_CE_BASIC_CONSTRAINTS: ("id-ce-basicConstraints", "basicConstraints"),
    ID_CE_NAME_CONSTRAINTS: ("id-ce-nameConstraints", "nameConstraints"),
    ID_CE_POLICY_CONSTRAINTS: ("id-ce-policyConstraints", "policyConstraints"),
    ID_CE_EXT_KEY_USAGE: ("id-ce-extKeyUsage", "extendedKeyUsage"),
    ID_CE_CRL_DISTRIBUTION_POINTS: ("id-ce-cRLDistributionPoints",
                                    "CRLDistributionPoints"),
    ID_CE_INHIBIT_ANY_POLICY: ("id-ce-inhibitAnyPolicy", "inhibitAnyPolicy"),
    ID_PE_AUTHORITY_INFO_ACCESS: ("id-pe-authorityInfoAccess",
                                  "authorityInformationAccess"),
    ID_PE_SUBJECT_INFO_ACCESS: ("id-pe-subjectInfoAccess",
                                "subjectInformationAccess"),

    ID_KP_SERVER_AUTH: ("id-kp-serverAuth", "serverAuth"),
    ID_KP_CLIENT_AUTH: ("id-kp-clientAuth", "clientAuth"),
    ID_KP_CODE_SIGNING: ("id-kp-codeSigning", "codeSigning"),
    ID_KP_EMAIL_PROTECTION: ("id-kp-emailProtection", "emailProtection"),
    ID_KP_TIME_STAMPING: ("id-kp-timeStamping", "timeStamping"),
    ID_KP_OCSP_SIGNING: ("id-kp-OCSPSigning", "OCSPSigning"),

    ID_AD_OCSP: ("id-ad-ocsp", "OCSP"),
    ID_AD_CA_ISSUERS: ("id-ad-caIssuers", "caIssuers"),
    ID_QT_CPS: ("id-qt-cps", "CPS"),
    ID_QT_UNOTICE: ("id-qt-unotice", "UserNotice"),
    ANY_POLICY: ("anyPolicy", "anyPolicy"),

    CT_EMBEDDED_SCT_LIST: ("ctEmbeddedSCT", "ctEmbeddedSCT"),
    CT_POISON: ("ctPoison", "ctPoison"),
    CT_PRECERTIFICATE_SIGNING: ("ctPrecertificateSigningCert", "ctPrecertificateSigningCert")
    }

_OID_DECODING_DICT = {}

for oid in _OID_NAME_DICT:
  _OID_DECODING_DICT[oid._encode_value()] = oid._value
