
"""X509 Certificate API."""

import collections
import hashlib
import time

from ct.crypto import error
from ct.crypto import pem
from ct.crypto.asn1 import oid
from ct.crypto.asn1 import x509
from ct.crypto.asn1 import x509_extension as x509_ext
from ct.crypto.asn1 import x509_name


class CertificateError(error.Error):
    """Certificate has errors."""
    pass


class Certificate(object):
    """X509 certificates."""
    PEM_MARKERS = ("CERTIFICATE",)

    def __init__(self, der_string, strict_der=True):
        """Initialize from a DER string.

        Args:
            der_string: a binary string containing the DER-encoded
                certificate.
            strict_der: if False, tolerate some non-fatal DER errors.

        Raises:
            error.ASN1Error: invalid encoding.
        """
        # ASN.1 errors fall through.
        self._asn1_cert = x509.Certificate.decode(der_string,
                                                  strict=strict_der)
        # The general philosophy here is that a certificate decoded in
        # strict mode should never raise CertificateErrors later on in the
        # code. Strict mode already catches corrupt extensions, to the extent
        # that their IDs are recognized; in addition, we have to ensure that
        # no extension appears more than once.
        # TODO(ekasper): move this check to the Extensions class.
        if strict_der and self._has_multiple_extension_values():
            raise error.ASN1Error("Multiple extensions")
        # Certs are primarily used as read-only objects so we cache the DER
        # encoding. If any public setters or other methods modifying the
        # contents of the certificate are ever added to this class, they must
        # invalidate the cached encoding.

    def __repr__(self):
        # This prints the full ASN1 representation. Useful for debugging.
        return repr(self._asn1_cert)

    def __str__(self):
        return self._asn1_cert.human_readable(label=self.__class__.__name__)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.is_identical_to(other)
        else:
            return NotImplemented

    def __ne__(self, other):
        are_equal = self.__eq__(other)

        if are_equal is NotImplemented:
            return NotImplemented
        else:
            return not are_equal

    def __hash__(self):
        return hash(self.fingerprint())

    @classmethod
    def from_pem(cls, pem_string, strict_der=True):
        """Read a single PEM-encoded certificate from a string.

        Args:
            pem_string: the certificate string.
            strict_der: if False, tolerate some non-fatal DER errors.

        Returns:
            a Certificate object.

        Raises:
            ct.crypto.pem.PemError, ct.crypto.error.ASN1Error: the string
            does not contain a valid PEM certificate.
        """
        der_cert, _ = pem.from_pem(pem_string, cls.PEM_MARKERS)
        return cls.from_der(der_cert, strict_der=strict_der)

    def _has_multiple_extension_values(self):
        """Returns true if any extension appears more than once."""
        extns = self._asn1_cert["tbsCertificate"]["extensions"] or []
        extn_value_count = collections.Counter([e["extnID"] for e in extns])
        return any([c > 1 for c in extn_value_count.values()])

    def _get_decoded_extension_value(self, extn_id):
        """Get the decoded value of an extension.

        Args:
            extn_id: extension OID.

        Returns:
            the decoded extension value matching an extn_id, or None.

        Raises:
            CertificateError: corrupt extension, or multiple extensions
                matching the given OID.
        """
        decoded_extn_values = self._get_decoded_extension_values(extn_id)
        if not decoded_extn_values:
            return None
        if len(decoded_extn_values) > 1:
            # TODO(ekasper): could refine this to only raise when the multiple
            # extension values are conflicting.
            raise CertificateError("Multiple extension values")
        return decoded_extn_values[0]

    def _get_decoded_extension_values(self, extn_id):
        """Get all decoded values matching an extension ID.
        Args:
            extn_id: extension OID.

        Returns:
            a list of decoded extension values matching an extn_id
            (i.e., tolerates multiple extensions).

        Raises:
            CertificateError: corrupt extension.
        """
        extns = self._asn1_cert["tbsCertificate"]["extensions"] or []
        extn_values = [e["extnValue"] for e in extns if e["extnID"] == extn_id]
        if not extn_values:
            return []
        decoded_values = [e.decoded_value for e in extn_values]
        if any([val is None for val in decoded_values]):
            raise CertificateError("Corrupt or unrecognized extension: %s"
                                   % extn_id)
        return decoded_values

    @classmethod
    def from_der(cls, der_string, strict_der=True):
        """Read a single DER-encoded certificate from a string.

        This is just an alias to __init__ to match from_pem().

        Args:
            der_string: the certificate string.
            strict_der: if False, tolerate some non-fatal DER errors.

        Returns:
            a Certificate object.

        Raises:
            ct.crypto.error.ASN1Error: the string does not contain a valid
            DER certificate.
        """
        return cls(der_string, strict_der=strict_der)

    @classmethod
    def from_pem_file(cls, pem_file, strict_der=True):
        """Read a single PEM-encoded certificate from a file.

        Args:
            pem_file: the certificate file.
            strict_der: if False, tolerate some non-fatal DER errors.

        Returns:
            a Certificate object
        Raises:
            ct.crypto.pem.PemError, ct.crypto.error.ASN1Error: the file does not
            contain a valid PEM certificate
            IOError: the file could not be read,
        """
        der_cert, _ = pem.from_pem_file(pem_file, cls.PEM_MARKERS)
        return cls.from_der(der_cert, strict_der=strict_der)

    @classmethod
    def from_der_file(cls, der_file, strict_der=True):
        """Read a single DER-encoded certificate from a file.

        Args:
            der_file: the certificate file.
            strict_der: if False, tolerate some non-fatal DER errors.

        Returns:
            a Certificate object.

        Raises:
            ct.crypto.error.ASN1Error: the file does not contain a valid DER
            certificate
            IOError: the file could not be read.
        """
        with open(der_file, "rb") as der_cert_file:
            return cls.from_der(der_cert_file.read(), strict_der=strict_der)

    def to_der(self):
        """Get the DER-encoding of the certificate."""
        return self._asn1_cert.encode()

    def to_pem(self):
        """Get the PEM-encoding of the certificate."""
        return pem.to_pem(self._asn1_cert.encode(), self.PEM_MARKERS[0])

    def is_identical_to(self, other_cert):
        """Returns True if this certificate is identical to |other_cert|."""
        return self.to_der() == other_cert.to_der()

    def to_asn1(self):
        """Get a copy of the ASN.1 representation of the certificate."""
        return x509.Certificate.decode(self._asn1_cert.encode())

    def get_extensions(self):
        """Get all extensions.

        Returns:
            a list of extensions.
        """
        return self._asn1_cert["tbsCertificate"]["extensions"] or []

    def version(self):
        """Get the version.

        Returns:
            an integral value of the version (i.e., V1 is 0).
        """
        return self._asn1_cert["tbsCertificate"]["version"]

    def issuer_common_name(self):
        """Get the common names of the issuer.

        Returns:
            an list of common name strings

        Raises:
            Corrupt issuer common name attribute.
        """
        try:
            return self._asn1_cert["tbsCertificate"]["issuer"].attributes(
                    oid.ID_AT_COMMON_NAME)
        except error.ASN1Error:
            raise CertificateError("Corrupt issuer common name attribute")

    def issuer_country_name(self):
        """Get issuer country name.

        Returns:
            an list of issuer country names.

        Raises:
            Corrupt issuer country name attribute.
        """
        try:
            return self._asn1_cert["tbsCertificate"]["issuer"].attributes(
                    oid.ID_AT_COUNTRY_NAME)
        except error.ASN1Error:
            raise CertificateError("Corrupt issuer country name attribute.")

    def subject_common_names(self):
        """Get the common names of the subject.

        Returns:
            a list of common name strings ("CN" attribute values in the
            certificate's "subject" field).

        Raises:
            CertificateError: corrupt CN attribute.
        """
        try:
            return self._asn1_cert["tbsCertificate"]["subject"].attributes(
                oid.ID_AT_COMMON_NAME)
        except error.ASN1Error:
            raise CertificateError("Corrupt common name attribute")

    def subject_organization_name(self):
        """Get subject organization name.

        Returns:
            a list of subject organization names.

        Raises:
            CertificateError: corrupt subject organization name attribute.
        """
        try:
            return self._asn1_cert["tbsCertificate"]["subject"].attributes(
                    oid.ID_AT_ORGANIZATION_NAME)
        except error.ASN1Error:
            raise CertificateError("Corrupt subject organization name "
                                   "attribute.")

    def subject_street_address(self):
        """Get subject street address.

        Returns:
            a list of subject street addresses.

        Raises:
            CertificateError: corrupt subject street adress attribute.
        """
        try:
            return self._asn1_cert["tbsCertificate"]["subject"].attributes(
                    oid.ID_AT_STREET_ADDRESS)
        except error.ASN1Error:
            raise CertificateError("Corrupt subject street address attribute.")

    def subject_locality_name(self):
        """Get subject locality name.

        Returns:
            a list of subject locality names.

        Raises:
            CertificateError: corrupt subject locality names attribute.
        """
        try:
            return self._asn1_cert["tbsCertificate"]["subject"].attributes(
                    oid.ID_AT_LOCALITY_NAME)
        except error.ASN1Error:
            raise CertificateError("Corrupt subject locality name attribute")

    def subject_state_or_province_name(self):
        """Get subject state or province name.

        Returns:
            a list of subject state or province names.

        Raises:
            CertificateError: corrupt subject state or province names attribute.
        """
        try:
            return self._asn1_cert["tbsCertificate"]["subject"].attributes(
                    oid.ID_AT_STATE_OR_PROVINCE_NAME)
        except error.ASN1Error:
            raise CertificateError("Corrupt subject state or province name "
                                   "attribute.")

    def subject_postal_code(self):
        """Get subject postal code.

        Returns:
            a list of subject postal codes.

        Raises:
            CertificateError: corrupt subject postal code attribute.
        """
        try:
            return self._asn1_cert["tbsCertificate"]["subject"].attributes(
                    oid.ID_AT_POSTAL_CODE)
        except error.ASN1Error:
            raise CertificateError("Corrupt subject postal code attribute.")

    def subject_organizational_unit_name(self):
        """Get subject organizational unit name.

        Returns:
            a list of subject organizational unit names.

        Raises:
            CertificateError: corrupt subject organizational unit name attribute
        """
        try:
            return self._asn1_cert["tbsCertificate"]["subject"].attributes(
                    oid.ID_AT_ORGANIZATIONAL_UNIT_NAME)
        except error.ASN1Error:
            raise CertificateError("Corrupt subject organizational unit name "
                                   "attribute.")

    def subject_alternative_names(self):
        """Get a list of subjectAlternativeNames extension values.

        Returns:
            a list of subject alternative names. Each element in the list
            is an ASN.1 GeneralName. If multiple SAN extensions are present
            (non-strict mode), returns a list of all names found (possibly
            with duplicates).

            Since each GeneralName is a choice, its contents can be inspected
            as follows:

            sans = cert.subject_alternative_names()
            for san in sans:
                # The type of the alternative name; one of
                # x509_name.OTHER_NAME
                # x509_name.RFC822_NAME
                # x509_name.X400_ADDRESS_NAME
                # x509_name.DIRECTORY_NAME
                # x509_name.EDI_PARTY_NAME
                # x509_name.URI_NAME
                # x509_name.IP_ADDRESS_NAME
                # x509_name.REGISTERED_ID_NAME
                san_type = san.component_key()

                # The corresponding ASN.1 value:
                san_value = san.component_value()
                ...

        Raises:
            CertificateError: corrupt extension.
        """
        sans = self._get_decoded_extension_values(oid.ID_CE_SUBJECT_ALT_NAME)
        return sum([list(san) for san in sans], [])

    def subject(self):
        """Returns list of subject field values in fashion similar to
        subject_alternative_names fashion.
        """
        subject = self._asn1_cert["tbsCertificate"]["subject"]
        return [(sub['type'], sub['value'])
                for sub in subject.flatten()]

    def issuer(self):
        """Returns list of issuer field values in fashion similar to
        subject method fashion.
        """
        issuer = self._asn1_cert["tbsCertificate"]["issuer"]
        return [(iss['type'], iss['value'])
                for iss in issuer.flatten()]

    def _get_subject_alt_names_by_type(self, san_type):
        """Returns the SAN extension values corresponding to |san_type|"""
        # A certificate should only have one SAN extension but we can't rely on
        # this (in non-strict mode), so we return everything we find.
        sans = self._get_decoded_extension_values(oid.ID_CE_SUBJECT_ALT_NAME)
        ret = []
        for san in sans:
            ret += [c.component_value() for c in san
                    if c.component_key() == san_type]
        return ret

    # DNS names and IP addresses are relevant to HTTPS server identification.
    # Other types of alternative names *should* normally be ignored by web
    # browsers but may be relevant for other services.
    # See http://tools.ietf.org/html/rfc6125
    def subject_dns_names(self):
        """Get the dnsNames in the subjectAlternativeNames extension.

        Returns:
            a list of DNS names in the subjectAlternativeNames (SAN) extension.
            If multiple SAN extensions are present (non-strict mode), returns
            a list of all DNS names found (possibly with duplicates).

        Raises:
            CertificateError: corrupt extension.
        """
        return self._get_subject_alt_names_by_type(x509_name.DNS_NAME)

    def subject_ip_addresses(self):
        """Get the ipAddresses in the subjectAlternativeNames extension.

        Returns:
            a list of IPAddress entries in the subjectAlternativeNames (SAN)
            extension. If multiple SAN extensions are present (non-strict mode),
            returns a list of all IP addresses found (possibly with duplicates).

        Raises:
            CertificateError: corrupt extension.
        """
        return self._get_subject_alt_names_by_type(x509_name.IP_ADDRESS_NAME)

    def print_subject_name(self):
        """Get a human readable string of the subject name attributes."""
        return (self._asn1_cert["tbsCertificate"]["subject"].
                human_readable(wrap=0))

    def print_issuer_name(self):
        """Get a human readable string of the issuer name attributes."""
        return (self._asn1_cert["tbsCertificate"]["issuer"].
                human_readable(wrap=0))

    def serial_number(self):
        """Get the serial number.

        While the serial number is an integer, it could be very large.
        RFC 5280 specification states is should not be longer than 20 octets,
        and also states users SHOULD be prepared to gracefully
        handle non-conforming certificates.

        Returns:
           An ASN.1 Integer.

        """
        return self._asn1_cert["tbsCertificate"]["serialNumber"]

    def signature(self):
        """Get TBSCertificate signature.

        Returns:
            an ASN.1 BitString.
        """
        return self._asn1_cert["tbsCertificate"]["signature"]

    def signature_algorithm(self):
        """Get Certificate signature algorithm.

        Returns:
            an AlgorithmIdentifier
        """
        return self._asn1_cert["signatureAlgorithm"]

    def basic_constraint_ca(self):
        """Get the BasicConstraints CA value.

        Returns:
            an ASN.1 Boolean, or None.

        Raises:
            CertificateError: corrupt extension, or multiple extensions.
        """
        # CertificateErrors fall through.
        bc = self._get_decoded_extension_value(oid.ID_CE_BASIC_CONSTRAINTS)
        if bc is None:
            return None

        return bc["cA"]

    def basic_constraint_path_length(self):
        """Get the BasicConstraints pathLenConstraint value.

        Returns:
            an ASN.1 Integer, or None.

        Raises:
            CertificateError: corrupt extension, or multiple extensions.
        """
        bc = self._get_decoded_extension_value(oid.ID_CE_BASIC_CONSTRAINTS)
        if bc is None:
            return None

        return bc["pathLenConstraint"]

    def not_before(self):
        """Get a time.struct_time representing the notBefore in UTC time.

        Returns:
            a time.struct_time object.

        Raises:
            CertificateError: corrupt notBefore value.
        """
        return (self._asn1_cert["tbsCertificate"]["validity"]["notBefore"].
                component_value().gmtime())

    def not_after(self):
        """Get a time.struct_time representing the notAfter in UTC time.

        Returns:
            a time.struct_time object.

        Raises:
            CertificateError: corrupt notAfter value.
        """
        return (self._asn1_cert["tbsCertificate"]["validity"]["notAfter"].
                component_value().gmtime())

    def is_not_after_well_defined(self):
        """Checks if notAfter field is well defined.

        RFC5280 specifies that to indicate that certificate has no well-defined
        expiration date notAfter should be set to 99991231235959Z
        generalized time.

        Returns:
            True if not after is well defined, False otherwise
        """
        not_after = self._asn1_cert["tbsCertificate"]["validity"]["notAfter"]
        if "generalTime" in not_after.value:
            not_after = not_after.value["generalTime"]
            if not_after.value == '99991231235959Z':
                return False
        return True

    def is_temporally_valid_now(self):
        """Determine whether notBefore <= now <= notAfter.

        Returns:
            True or False.

        Raises:
            CertificateError: corrupt time.
        """
        return self.is_temporally_valid_at(time.gmtime())

    def is_expired(self):
        """Is certificate notAfter in the past?

        Returns:
            True or False.

        Raises:
            CertificateError: corrupt time.
        """
        now = time.gmtime()
        return now > self.not_after()

    def is_not_yet_valid(self):
        """Is certificate notBefore in the future?

        Returns:
            True or False.

        Raises:
            CertificateError: corrupt time.
        """
        now = time.gmtime()
        return now < self.not_before()

    def is_temporally_valid_at(self, gmtime):
        """Is certificate valid at the given moment?

        Args:
            gmtime: a struct_time GMT time.

        Returns:
            True or False.

        Raises:
            CertificateError: corrupt time.
        """
        return self.not_before() <= gmtime <= self.not_after()

    def is_self_signed(self):
        """Is self signed?

        Returns:
            True or False.
        """
        return (self._asn1_cert["tbsCertificate"]["issuer"] ==
                self._asn1_cert["tbsCertificate"]["subject"])

    def fingerprint(self, hashfunc="sha1"):
        """Get the certificate fingerprint.

        Args:
            hashfunc: name of a hash function. Algorithms always present are
                'md5', 'sha1', 'sha224', 'sha256', 'sha384', and 'sha512'.
        Returns:
            a (binary) hash digest of the DER encoding.
        """
        h = hashlib.new(hashfunc)
        h.update(self._asn1_cert.encode())
        return h.digest()

    def key_hash(self, hashfunc="sha1"):
        """Get the certificate's public key hash.

        Args:
            hashfunc: name of a hash function. Algorithms always present are
                'md5', 'sha1', 'sha224', 'sha256', 'sha384', and 'sha512'.
        Returns:
            a (binary) hash digest of the public key.
        """
        h = hashlib.new(hashfunc)
        h.update(
            self._asn1_cert["tbsCertificate"]["subjectPublicKeyInfo"].encode())
        return h.digest()

    def key_usage(self, key_usage):
        """Whether the certificate has the given key usage asserted.

        Args:
            key_usage: the usage; one of
            x509_extension.KeyUsage.DIGITAL_SIGNATURE
            x509_extension.KeyUsage.NON_REPUDIATION
            x509_extension.KeyUsage.KEY_ENCIPHERMENT
            x509_extension.KeyUsage.DATA_ENCIPHERMENT
            x509_extension.KeyUsage.KEY_AGREEMENT
            x509_extension.KeyUsage.KEY_CERT_SIGN
            x509_extension.KeyUsage.CRL_SIGN
            x509_extension.KeyUsage.ENCIPHER_ONLY
            x509_extension.KeyUsage.DECIPHER_ONLY

        Returns:
            True: the key usage is asserted.
            False: the key usage extension is present but the specified
                usage is not asserted.
            None: the key usage extension is not present.

        Raises:
            CertificateError: corrupt key usage extension, or multiple
                extension values.
        """
        ku = self._get_decoded_extension_value(oid.ID_CE_KEY_USAGE)
        if ku is None:
            return None
        return ku.has_bit_set(key_usage.value)

    def key_usages(self):
        """List the asserted key usages.

        Returns:
            A list of key usages asserted in the certificate (or an empty
                list if the extension is not present).

        Raises:
            CertificateError: corrupt key usage extension, or multiple
                extension values.
        """
        ku = self._get_decoded_extension_value(oid.ID_CE_KEY_USAGE)
        return ku.bits_set() if ku else []

    def extended_key_usage(self, ext_key_usage):
        """Whether the certificate has the given extended key usage asserted.

        Args:
            ext_key_usage: the object identifier of the usage, e.g.,
                ct.crypto.asn1.oid.ID_KP_SERVER_AUTH.

        Returns:
            True: the extended key usage is asserted.
            False: the extended key usage extension is present but the
                specified usage is not asserted.
            None: the extended key usage extension is not present.

        Raises:
            CertificateError: corrupt key usage extension, or multiple
                extension values.
        """
        eku = self._get_decoded_extension_value(oid.ID_CE_EXT_KEY_USAGE)
        if eku is None:
            return None
        return ext_key_usage in eku

    def extended_key_usages(self):
        """List the asserted extended key usages.

        Returns:
            A sequence of extended key usage identifiers asserted in the
            certificate (or an empty list if the extension is not present).

        Raises:
            CertificateError: corrupt key usage extension, or multiple
                extension values.
        """
        return self._get_decoded_extension_value(oid.ID_CE_EXT_KEY_USAGE) or []

    def subject_key_identifier(self):
        """Get the subject key identifier.

        Returns:
            An x509_extension.KeyIdentifier (ASN.1 OctetString) holding the
            value of the subject key identifier, or None if the subject key
            identifier extension is not present.
        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        return self._get_decoded_extension_value(
            oid.ID_CE_SUBJECT_KEY_IDENTIFIER)

    def authority_key_identifier(self, identifier_type=x509_ext.KEY_IDENTIFIER):
        """Get the authority key identifier of the given type.

        Args:
            identifier_type: the identifier component to fetch, one of
              x509_extension.KEY_IDENTIFIER,
              x509_extension.AUTHORITY_CERT_ISSUER,
              x509_extension.AUTHORITY_CERT_SERIAL_NUMBER.

        Returns:
            the identifier component of the appropriate type, or None if the
            component/extension is not present. The types are
              x509_extension.KEY_IDENTIFIER: x509_extension.KeyIdentifier
                  (an OCTET STRING),
              x509_extension.AUTHORITY_CERT_ISSUER: x509_name.GeneralNames,
              x509_extension.AUTHORITY_CERT_SERIAL_NUMBER:
                  x509_common.CertificateSerialNumber.

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        akid = self._get_decoded_extension_value(
            oid.ID_CE_AUTHORITY_KEY_IDENTIFIER)

        return akid[identifier_type] if akid else None

    def policies(self):
        """List certificate policies.

        Returns:
            a list of certificate policies, or an empty list if the extension is
            not present.

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        policies = self._get_decoded_extension_value(
            oid.ID_CE_CERTIFICATE_POLICIES)
        return policies or []

    def policy(self, policy_oid):
        """Find a policy with the given OID.

        Returns:
            The matching policy, or None.

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        policies = self.policies()

        if not policies:
            return None

        matching = [p for p in policies
                    if p[x509_ext.POLICY_IDENTIFIER] == policy_oid]
        if not matching:
            return None

        # TODO(ekasper): make this exception fire earlier in strict mode.
        if len(matching) > 1:
            raise CertificateError("Policy %s asserted more than once" %
                                   policy_oid)
        return matching[0]

    def has_policy(self, policy_oid):
        """Whether the certificate has a given policy.

        Returns:
            True if the given policy is asserted, False otherwise.

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        return self.policy(policy_oid) is not None

    def crl_distribution_points(self):
        """List CRL distribution points.

        Returns:
            a list of DistributionPoint entries.

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        return (self._get_decoded_extension_value(
            oid.ID_CE_CRL_DISTRIBUTION_POINTS) or [])

    def ca_issuers(self):
        """List CA issuers from the Authority Information Access extension.

        Returns:
            a list of CA issuers (as ASN.1 GeneralNames).

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        aia = self._get_decoded_extension_value(oid.ID_PE_AUTHORITY_INFO_ACCESS)
        if aia is None:
            return []
        return [a[x509_ext.ACCESS_LOCATION] for a in aia
                if a[x509_ext.ACCESS_METHOD] == oid.ID_AD_CA_ISSUERS]

    def ocsp_responders(self):
        """List OCSP responders from the Authority Information Access extension.

        Returns:
            a list of OCSP responders (as ASN.1 GeneralNames).

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        aia = self._get_decoded_extension_value(oid.ID_PE_AUTHORITY_INFO_ACCESS)
        if aia is None:
            return []
        return [a[x509_ext.ACCESS_LOCATION] for a in aia
                if a[x509_ext.ACCESS_METHOD] == oid.ID_AD_OCSP]

    def embedded_sct_list(self):
        """Get the encoded list of embedded timestamps

        Returns:
            bytes representing a TLS encoded SignedCertificateTimestampList,
            or None if not present

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        signed_certificate_timestamp_list = (
                self._get_decoded_extension_value(oid.CT_EMBEDDED_SCT_LIST))
        if signed_certificate_timestamp_list is None:
            return None
        return signed_certificate_timestamp_list.value

    def has_extension(self, extn_id):
        """Check if certificate contains a given extnsion.

        Args:
            extn_id: extension OID.

        Returns:
            True or False

        Raises:
            CertificateError: corrupt extension, or multiple extension values.
        """
        return self._get_decoded_extension_value(extn_id) is not None

def certs_from_pem(pem_string, skip_invalid_blobs=False, strict_der=True):
    """Read multiple PEM-encoded certificates from a string.

    Args:
        pem_string: the certificate string
        skip_invalid_blobs: if False, invalid PEM blobs cause a PemError.
            If True, invalid blobs are skipped. In non-skip mode, an
            immediate StopIteration before any valid blocks are found also
            causes a PemError exception.
        strict_der: if False, tolerate some non-fatal DER errors.

    Yields:
        Certificate objects.

    Raises:
        ct.crypto.pem.PemError, ct.crypto.ASN1Error: a block was invalid
        IOError: the file could not be read.
    """
    for der_cert, _ in pem.pem_blocks(pem_string, Certificate.PEM_MARKERS,
                                      skip_invalid_blobs=skip_invalid_blobs):
        try:
            yield Certificate.from_der(der_cert, strict_der=strict_der)
        except error.ASN1Error:
            if not skip_invalid_blobs:
                raise


def certs_from_pem_file(pem_file, skip_invalid_blobs=False, strict_der=True):
    """Read multiple PEM-encoded certificates from a file.

    Args:
        pem_file: the certificate file.
        skip_invalid_blobs: if False, invalid PEM blobs cause a PemError.
            If True, invalid blobs are skipped. In non-skip mode, an
            immediate StopIteration before any valid blocks are found also
            causes a PemError exception.
        strict_der: if False, tolerate some non-fatal DER errors.

    Yields:
        Certificate objects.

    Raises:
        ct.crypto.pem.PemError, ct.crypto.error.ASN1Error:
            a block was invalid
        IOError: the file could not be read.
    """
    for der_cert, _ in pem.pem_blocks_from_file(
        pem_file, Certificate.PEM_MARKERS,
        skip_invalid_blobs=skip_invalid_blobs):
        try:
            yield Certificate.from_der(der_cert, strict_der=strict_der)
        except error.ASN1Error:
            if not skip_invalid_blobs:
                raise
