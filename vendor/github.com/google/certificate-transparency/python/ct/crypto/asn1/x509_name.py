"""ASN.1 specification for X509 name types."""

from ct.crypto import error
from ct.crypto.asn1 import oid
from ct.crypto.asn1 import types


class AttributeType(oid.ObjectIdentifier):
    pass


class AttributeValue(types.Any):
    pass


class DirectoryString(types.Choice):
    components = {
        "teletexString": types.TeletexString,
        "printableString": types.PrintableString,
        "universalString": types.UniversalString,
        "utf8String": types.UTF8String,
        "bmpString": types.BMPString,
        # Does not really belong here.
        "ia5String": types.IA5String
        }


_ATTRIBUTE_DICT = {
    # Note: this mapping does not conform to the RFCs, as some of the OIDs
    # have more restricted values. But real certificates do not conform either,
    # so we try to be lenient and accept all strings that we can recognize.
    oid.ID_AT_NAME: DirectoryString,
    oid.ID_AT_SURNAME: DirectoryString,
    oid.ID_AT_GIVEN_NAME: DirectoryString,
    oid.ID_AT_INITIALS: DirectoryString,
    oid.ID_AT_GENERATION_QUALIFIER: DirectoryString,
    oid.ID_AT_COMMON_NAME: DirectoryString,
    oid.ID_AT_LOCALITY_NAME: DirectoryString,
    oid.ID_AT_STATE_OR_PROVINCE_NAME: DirectoryString,
    oid.ID_AT_ORGANIZATION_NAME: DirectoryString,
    oid.ID_AT_ORGANIZATIONAL_UNIT_NAME: DirectoryString,
    oid.ID_AT_TITLE: DirectoryString,
    oid.ID_AT_DN_QUALIFIER: DirectoryString,  # PrintableString
    oid.ID_AT_COUNTRY_NAME: DirectoryString,  # PrintableString
    oid.ID_AT_SERIAL_NUMBER: DirectoryString,  # PrintableString
    oid.ID_AT_PSEUDONYM: DirectoryString,
    oid.ID_DOMAIN_COMPONENT: DirectoryString,  # IA5String
    oid.ID_EMAIL_ADDRESS: DirectoryString,  # IA5String
    oid.ID_AT_STREET_ADDRESS: DirectoryString,
    oid.ID_AT_DESCRIPTION: DirectoryString,
    oid.ID_AT_BUSINESS_CATEGORY: DirectoryString,
    oid.ID_AT_POSTAL_CODE: DirectoryString,
    oid.ID_AT_POST_OFFICE_BOX: DirectoryString,
    }


class AttributeTypeAndValue(types.Sequence):
    print_labels = False
    print_delimiter = "="
    components = (
        (types.Component("type", AttributeType)),
        (types.Component("value", AttributeValue, defined_by="type",
                         lookup=_ATTRIBUTE_DICT))
        )


class RelativeDistinguishedName(types.SetOf):
    print_labels = False
    print_delimiter = ", "
    component = AttributeTypeAndValue


class RDNSequence(types.SequenceOf):
    print_labels = False
    print_delimiter = "/"
    component = RelativeDistinguishedName
    # See http://tools.ietf.org/html/rfc6125 for context.
    def flatten(self):
        """Get a flat list of AttributeTypeAndValue pairs in an RDNSequence.

        The hierarchical (Relative) information is not used in all contexts,
        so we provide a way of discarding that information and flattening
        the structure.
        """
        return sum([list(rdn) for rdn in self], [])

    def attributes(self, attr_type):
        """Get a flat list of attribute values of the given type.

        Returns:
            a list of attributes.

        Raises:
            error.ASN1Error: corrupt attribute value.
        """
        attrs = self.flatten()
        decoded_values = [attr["value"].decoded_value for attr in attrs
                          if attr["type"] == attr_type]
        if any([val is None for val in decoded_values]):
            raise error.ASN1Error("Corrupt name attribute")
        # A subject name attribute is always a DirectoryString (a Choice),
        # so we need to take its value.
        return [d.component_value() for d in decoded_values]


# Bypass the CHOICE indirection since exactly one option is specified.
# class Name(types.Choice):
#     components = {"rdnSequence": RDNSequence}
class Name(RDNSequence):
    pass


class OtherName(types.Sequence):
    print_delimiter = ", "
    components = (
        (types.Component("type-id", oid.ObjectIdentifier)),
        (types.Component("value", types.Any.explicit(0)))
        )


class EDIPartyName(types.Sequence):
    print_delimiter = ", "
    components = (
        # Definition here: http://tools.ietf.org/html/rfc5280#section-4.2.1.6
        # Note: this definition suggests that the tagging is implicit.
        # However, implicit tagging of a CHOICE type is ambiguous, so this is
        # in practice interpreted as an explicit tag.
        (types.Component("nameAssigner", DirectoryString.explicit(0),
                         optional=True)),
        (types.Component("partyName", DirectoryString.explicit(1)))
        )


# Partially defined ORAddress: we've not come across any certs that contain it
# but this should be enough to allow the decoder to continue without blowing up.
class BuiltInDomainDefinedAttributes(types.SequenceOf):
    component = types.Any


class ExtensionAttributes(types.SetOf):
    component = types.Any


class ORAddress(types.Sequence):
    components = (
        (types.Component("builtInStandardAttributes", types.Any)),
        (types.Component("builtInDomainDefinedAttributes",
                         BuiltInDomainDefinedAttributes, optional=True)),
        (types.Component("extensionAttributes",
                         ExtensionAttributes, optional=True))
        )


OTHER_NAME = "otherName"
RFC822_NAME = "rfc822Name"
DNS_NAME = "dNSName"
X400_ADDRESS_NAME = "x400Address"
DIRECTORY_NAME = "directoryName"
EDI_PARTY_NAME = "ediPartyName"
URI_NAME = "uniformResourceIdentifier"
IP_ADDRESS_NAME = "iPAddress"
REGISTERED_ID_NAME = "registeredID"


class IPAddress(types.OctetString):
    def __init__(self, value=None, serialized_value=None, strict=True):
        super(IPAddress, self).__init__(value=value,
                                        serialized_value=serialized_value,
                                        strict=strict)
        if strict and len(self._value) != 4 and len(self._value) != 16:
            raise error.ASN1Error("%s is not a valid IP address" %
                                  self.value.encode("hex"))

    def as_octets(self):
        return tuple([ord(b) for b  in self._value])

    def __str__(self):
        if len(self._value) == 4:
            return ".".join([str(ord(c)) for c in self._value])
        if len(self._value) == 16:
            return ":".join([self._value[i:i+2].encode("hex")
                             for i in range(0, len(self._value), 2)])
        return super(IPAddress, self).__str__()


class GeneralName(types.Choice):
    print_labels = True  # Print component type.
    # Definition here: http://tools.ietf.org/html/rfc5280#section-4.2.1.6
    components = {
        OTHER_NAME: OtherName.implicit(0),
        RFC822_NAME: types.IA5String.implicit(1),
        DNS_NAME: types.IA5String.implicit(2),
        X400_ADDRESS_NAME: ORAddress.implicit(3),
        # Implicit CHOICE tag is converted to an explicit one.
        DIRECTORY_NAME: Name.explicit(4),
        EDI_PARTY_NAME: EDIPartyName.implicit(5),
        URI_NAME: types.IA5String.implicit(6),
        IP_ADDRESS_NAME: IPAddress.implicit(7),
        REGISTERED_ID_NAME: oid.ObjectIdentifier.implicit(8)
        }


class GeneralNames(types.SequenceOf):
  component = GeneralName
