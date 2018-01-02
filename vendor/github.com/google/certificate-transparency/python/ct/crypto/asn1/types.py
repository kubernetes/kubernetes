"""ASN.1 types.

Spec: http://www.itu.int/ITU-T/studygroups/com17/languages/X.690-0207.pdf
See also http://luca.ntop.org/Teaching/Appunti/asn1.html for a good introduction
to ASN.1.

This module implements a restricted encoder/decoder for a subset of ASN.1 types.

The decoder has a strict and non-strict mode. Non-strict mode tolerates selected
non-fatal DER decoding errors. The encoder is DER-only.

Generic decoding is not supported: objects can only be decoded against a
predefined ASN.1 type. However, applications can derive arbitrary custom ASN.1
type specifications from the supported base types.

Constraints (e.g., on the length of an ASN.1 string value) are not supported,
and should be checked at application level, where necessary.
"""

import abc
import collections
import functools
import re

from ct.crypto import error
from ct.crypto.asn1 import print_util
from ct.crypto.asn1 import tag


_ZERO = "\x00"
_MINUS_ONE = "\xff"
_EOC = "\x00\x00"


def encode_int(value, signed=True):
    """Encode an integer.

    Args:
        value: an integral value.
        signed: if True, encode in two's complement form. If False, encode as
            an unsigned integer.

    Raises:
        ValueError: attempt to encode a negative integer as unsigned.

    Returns:
        a variable-length string representing the encoded integer.
    """
    if not signed and value < 0:
        raise ValueError("Unsigned integer cannot be negative")

    if not value:
        return _ZERO
    if value == -1:
        return _MINUS_ONE

    int_bytes = bytearray()

    while value != 0 and value != -1:
        int_bytes.append(value & 0xff)
        value >>= 8

    if signed:
        # In two's complement form, negative values have the most significant
        # bit set, thus we:
        if value == -1 and int_bytes[-1] <= 127:
            # Add a "-1"-byte for indicating a negative value.
            int_bytes.append(0xff)
        elif value == 0 and int_bytes[-1] > 127:
            # Add a "0"-byte for indicating a positive value.
            int_bytes.append(0)

    int_bytes.reverse()
    return str(int_bytes)


def decode_int(buf, signed=True, strict=True):
    """Decode an integer.

    Args:
        buf: a string or string buffer.
        signed: if True, decode in two's complement form. If False, decode as
            an unsigned integer.

    Raises:
        ASN1Error.

    Returns:
        an integer.
    """
    if not buf:
        raise error.ASN1Error("Invalid integer encoding: empty value")

    leading = ord(buf[0])
    int_bytes = bytearray(buf[1:])

    if int_bytes:
      if strict and leading == 0 and int_bytes[0] < 128:
        # 0x00 0x42 == 0x42
        raise error.ASN1Error("Extra leading 0-bytes in integer "
                              "encoding")
      elif strict and signed and leading == 0xff and int_bytes[0] >= 128:
        # 0xff 0x82 == 0x82
        raise error.ASN1Error("Extra leading 0xff-bytes in negative "
                              "integer encoding")

    if signed and leading > 127:
            leading -= 256

    for b in int_bytes:
        leading <<= 8
        leading += b

    return leading


# Lengths between 0 and 127 are encoded as a single byte.
# Lengths greater than 127 are encoded as follows:
#   * MSB of first byte is 1 and remaining bits encode the number of
#     additional bytes.
#   * Remaining bytes encode the length.
_MULTIBYTE_LENGTH = 0x80
_MULTIBYTE_LENGTH_MASK = 0x7f


def encode_length(length):
    """Encode an integer.

    Args:
        length: a non-negative integral value.

    Returns:
        a string.
    """
    if length <= 127:
        return chr(length)
    encoded_length = encode_int(length, signed=False)
    return chr(_MULTIBYTE_LENGTH | len(encoded_length)) + encoded_length


def read_length(buf, strict=True):
    """Read an ASN.1 object length from the beginning of the buffer.

    Args:
        buf: a string or string buffer.
        strict: if false, accept indefinite length encoding.

    Raises:
        ASN1Error.

    Returns:
        a (length, rest) tuple consisting of a non-negative integer representing
        the length of an ASN.1 object, and the remaining bytes. For indefinite
        length, returns (-1, rest).
    """
    if not buf:
        raise error.ASN1Error("Invalid length encoding: empty value")
    length, rest = ord(buf[0]), buf[1:]
    if length <= 127:
        return length, rest
    # 0x80 == ASN.1 indefinite length
    if length == 128:
        if strict:
            raise error.ASN1Error("Indefinite length encoding")
        return -1, rest

    length &= _MULTIBYTE_LENGTH_MASK
    if len(rest) < length:
        raise error.ASN1Error("Invalid length encoding")
    # strict=True: let's hope that at least ASN.1 lengths are properly encoded.
    return (decode_int(rest[:length], signed=False, strict=True), rest[length:])


class Universal(object):
    """Apply a universal tag to the class.

    Can be used as a callable, or a decorator:

    Integer = Universal(2, tag.PRIMITIVE)(Abstract)

    is the same as

    @Universal(2, tag.PRIMITIVE)
    class Integer(Abstract):
        pass

    and defines a type with an ASN.1 integer tag.
    """

    def __init__(self, number, encoding):
        """Setup the tag.

        Args:
            number: the tag number.
            encoding: the encoding. One of tag.PRIMITIVE or tag.CONSTRUCTED.
        """
        self.tag = tag.Tag(number, tag.UNIVERSAL, encoding)

    def __call__(self, cls):
        """Apply the universal tag.

        Args:
            cls: class to modify. The class must have an empty 'tags'
                attribute.

        Returns:
            the class with a modified 'tags' attribute.

        Raises:
            TypeError: invalid application of the tag.
        """
        if cls.tags:
            raise TypeError("Cannot apply a UNIVERSAL tag to a tagged type.")
        cls.tags = (self.tag,)
        return cls


class Explicit(object):
    """Apply an explicit tag to the class.

    Can be used as a callable, or a decorator:

    MyInteger = Explicit(0, tag.APPLICATION)(Integer)

    is the same as

    @Explicit(0, tag.APPLICATION)
    class MyInteger(Integer):
        pass

    and results in a MyInteger type that is explicitly tagged with an
    application-class 0-tag.
    """

    def __init__(self, number, tag_class=tag.CONTEXT_SPECIFIC):
        """Setup the tag.

        Args:
            number: the tag number.
            tag_class: the tag class. One of tag.CONTEXT_SPECIFIC,
                tag.APPLICATION or tag.PRIVATE.

        Raises:
            TypeError: invalid application of the tag.
        """
        if tag_class == tag.UNIVERSAL:
            raise TypeError("Cannot tag with a UNIVERSAL tag")
        # Explicit tagging always results in constructed encoding.
        self._tag = tag.Tag(number, tag_class, tag.CONSTRUCTED)

    def __call__(self, cls):
        """Apply the explicit tag.

        Args:
            cls: class to modify. The class must have an iterable 'tags'
                attribute.
        Returns:
            the class with a modified 'tags' attribute.
        """
        tags = list(cls.tags)
        tags.append(self._tag)
        cls.tags = tuple(tags)
        return cls


class Implicit(object):
    """Apply an implicit tag to the class.

    Can be used as a callable, or a decorator:

    MyInteger = Implicit(0, tag.APPLICATION)(Integer)

    is the same as

    @Implicit(0, tag.APPLICATION)
    class MyInteger(Integer):
        pass

    and results in a MyInteger type whose tag is implicitly replaced with an
    application-class 0-tag.
    """

    def __init__(self, number, tag_class=tag.CONTEXT_SPECIFIC):
        """Setup the tag.

        Args:
            number: the tag number.
            tag_class: the tag class. One of tag.CONTEXT_SPECIFIC,
                tag.APPLICATION or tag.PRIVATE.

        Raises:
            TypeError: invalid application of the tag.
        """
        if tag_class == tag.UNIVERSAL:
            raise TypeError("Cannot tag with a UNIVERSAL tag")
        # We cannot precompute the tag because the encoding depends
        # on the existing tags.
        self._number = number
        self._tag_class = tag_class

    def __call__(self, cls):
        """Apply the implicit tag.

        Args:
            cls: class to modify. The class must have an iterable 'tags'
                attribute.

        Returns:
            the class with a modified 'tags' attribute.

        Raises:
            TypeError: invalid application of the tag.
        """
        if not cls.tags:
            raise TypeError("Cannot implicitly tag an untagged type")
        tags = list(cls.tags)
        # Only simple types and simple types derived via implicit tagging have a
        # primitive encoding, so the last tag determines the encoding type.
        tags[-1] = (tag.Tag(self._number, self._tag_class,
                            cls.tags[-1].encoding))
        cls.tags = tuple(tags)
        return cls


class Abstract(object):
    """Abstract base class."""
    __metaclass__ = abc.ABCMeta

    tags = ()

    @classmethod
    def explicit(cls, number, tag_class=tag.CONTEXT_SPECIFIC):
        """Dynamically create a new tagged type.

        Args:
            number: tag number.
            tag_class: tag class.

        Returns:
            a subtype of cls with the given explicit tag.
        """
        name = "%s.explicit(%d, %d)" % (cls.__name__, number, tag_class)

        # TODO(ekasper): the metaclass could register created types so we
        # return the _same_ type when called more than once with the same
        # arguments.
        mcs = cls.__metaclass__
        return_class = mcs(name, (cls,), {})
        return Explicit(number, tag_class)(return_class)

    @classmethod
    def implicit(cls, number, tag_class=tag.CONTEXT_SPECIFIC):
        """Dynamically create a new tagged type.

        Args:
            number: tag number.
            tag_class: tag class.

        Returns:
            a subtype of cls with the given implicit tag.
        """
        name = "%s.implicit(%d, %d)" % (cls.__name__, number, tag_class)
        mcs = cls.__metaclass__
        return_class = mcs(name, (cls,), {})
        return Implicit(number, tag_class)(return_class)

    def __init__(self, value=None, serialized_value=None, strict=True):
        """Initialize from a value or serialized buffer.

        Args:
            value: initializing value of an appropriate type. If the
                serialized_value is not set, the initializing value must be set.
            serialized_value: serialized inner value (with tags and lengths
                stripped).
            strict: if False, tolerate some non-fatal decoding errors.

        Raises:
            error.ASN1Error: decoding the serialized value failed.
            TypeError: invalid initializer.
        """
        if serialized_value is not None:
            self._value = self._decode_value(serialized_value, strict=strict)
        elif value is not None:
            self._value = self._convert_value(value)
        else:
            raise TypeError("Cannot initialize from None")
        self._serialized_value = serialized_value

    @classmethod
    def _convert_value(cls, value):
        """Convert initializer to an appropriate value."""
        raise NotImplementedError

    @abc.abstractmethod
    def _decode_value(self, buf, strict=True):
        """Decode the initializer value from a buffer.

        Args:
            buf: a string or string buffer.
            strict: if False, tolerate some non-fatal decoding errors.

        Returns:
           the value of the object.
        """
        pass

    @property
    def value(self):
        """Get the value of the object.

        An ASN.1 object can always be reconstructed from its value.
        """
        # Usually either the immutable value, or a shallow copy of
        # the mutable value.
        raise NotImplementedError

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.value)

    def __str__(self):
        return str(self.value)

    @abc.abstractmethod
    def _encode_value(self):
        """Encode the contents, excluding length and tags.

        Returns:
            a string representing the encoded value.
        """
        pass

    # Implemented by Choice and Any.
    # Used when the type is untagged so that read() does not reveal a length.
    @classmethod
    def _read(cls, buf, strict=True):
        """Read the value from the beginning of a string or buffer."""
        raise NotImplementedError

    # Only applicable where indefinite length encoding is possible, i.e., for
    # simple string types using constructed encoding and structured types
    # (Sequence, Set, SequenceOf, SetOf) only. Currently, it's only
    # implemented for structured types; constructed encoding of simple string
    # types is not supported. Only applicable in non-strict mode.
    @classmethod
    def _read_indefinite_value(cls, buf):
        """Read the inner value from the beginning of a string or buffer."""
        raise NotImplementedError

    def encode(self):
        """Encode oneself.

        Returns:
            a string representing the encoded object.
        """
        # If we have a read-only object that we created from a serialized value
        # and never modified since, use the original cached value.
        #
        # This ensures that objects decoded in non-strict mode will retain their
        # original encoding.
        #
        # BUG: we do not cache tag and length encoding, so reencoding is broken
        # for objects that use indefinite length encoding.
        if self._serialized_value and not self.modified():
            encoded_value = self._serialized_value
        else:
            # We can only use the cached value if the object has never been
            # modified after birth. Since mutable objects cannot track when
            # their recursive subcomponents are modified, the modified flag,
            # once set, can never be unset.
            self._serialized_value = None
            encoded_value = self._encode_value()
        for t in self.tags:
            encoded_length = encode_length(len(encoded_value))
            encoded_value = t.value + encoded_length + encoded_value
        return encoded_value

    @classmethod
    def read(cls, buf, strict=True):
        """Read from a string or buffer.

        Args:
            buf: a string or string buffer.
            strict: if False, tolerate some non-fatal decoding errors.

        Returns:
            a tuple consisting of an instance of the class and the remaining
            bytes.
        """
        if cls.tags:
            # Each indefinite length must be closed with the EOC (\x00\x00)
            # octet.
            # If we have multiple tags (i.e., explicit tagging is used) and the
            # outer tags use indefinite length, each such encoding adds an EOC
            # to the end (while a regular tag adds nothing). Therefore, we first
            # read all tags, then the value, and finally strip the EOC octets of
            # the explicit tags.
            indefinite = 0
            for t in reversed(cls.tags):
                if buf[:len(t)] != t.value:
                    raise error.ASN1TagError(
                        "Invalid tag: expected %s, got %s while decoding %s" %
                        (t, buf[:len(t.value)], cls.__name__))
                # Logging statements are really expensive in the recursion even
                # if debug-level logging itself is disabled.
                # logging.debug("%s: read tag %s", cls.__name__, t)
                buf = buf[len(t):]
                # Only permit indefinite length for constructed types.
                decoded_length, buf = read_length(buf, strict=(
                    strict or t.encoding != tag.CONSTRUCTED))
                if decoded_length == -1:
                    indefinite += 1
                # logging.debug("%s: read length %d", cls.__name__,
                #               decoded_length)
                elif len(buf) < decoded_length:
                    raise error.ASN1Error("Invalid length encoding in %s: "
                                          "read length %d, remaining bytes %d" %
                                          (cls.__name__, decoded_length,
                                           len(buf)))

            # The last tag had definite length.
            if decoded_length != -1:
                value, rest = (cls(serialized_value=buf[:decoded_length],
                                   strict=strict), buf[decoded_length:])
            else:
                decoded, rest = cls._read_indefinite_value(buf)
                value = cls(value=decoded)
                # _read_indefinite_value will strip the inner EOC.
                indefinite -= 1
            # Remove EOC octets corresponding to outer explicit tags.
            if indefinite:
                if rest[:indefinite*2] != _EOC*indefinite:
                    raise error.ASN1Error("Missing EOC octets")
                rest = rest[indefinite*2:]

        else:
            # Untagged CHOICE and ANY; no outer tags to determine the length.
            value, rest = cls._read(buf, strict=strict)

        # logging.debug("%s: decoded value %s", cls.__name__, value)
        # logging.debug("Remaining bytes: %d", len(rest))
        return value, rest

    @classmethod
    def decode(cls, buf, strict=True):
        """Decode from a string or buffer.

        Args:
            buf: a string or string buffer.
            strict: if False, tolerate some non-fatal decoding errors.

        Returns:
            an instance of the class.
        """
        value, rest = cls.read(buf, strict=strict)
        if rest:
            raise error.ASN1Error("Invalid encoding: leftover bytes when "
                                  "decoding %s" % cls.__name__)
        return value

    # Compare by value.
    # Note this means objects with equal values do not necessarily have
    # equal encodings.
    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    @abc.abstractmethod
    def human_readable_lines(self, wrap=80, label=""):
        """A pretty human readable representation of the object.

        Args:
            wrap: maximum number of characters per line. 0 or negative wrap
                means no limit. Should be chosen long enough to comfortably fit
                formatted data; otherwise it is simply ignored and output may
                look funny.
            label: a label prefix.

        Returns:
            a list of line strings of at most |wrap| characters each.
        """
        pass

    def human_readable(self, wrap=80, label=""):
        """A pretty human readable representation of the object.

        Args:
            wrap: maximum number of characters per line. 0 or negative wrap
               means no limit. Should be chosen long enough to comfortably fit
               formatted data; otherwise it is simply ignored and output may
               look funny.
            label: a label prefix.

        Returns:
            a multi-line string of at most |wrap| characters per line.
        """
        return ("\n").join(self.human_readable_lines(wrap=wrap, label=label))


# Boilerplate code for some simple types whose value directly corresponds to a
# basic immutable type.
@functools.total_ordering
class Simple(Abstract):
    """Base class for Boolean, Integer, and string types."""
    # Pretty-printed character length.
    # OctetString and BitString use this to nicely format hex bytes.
    char_wrap = 1

    @property
    def value(self):
        return self._value

    def __hash__(self):
        return hash(self.value)

    def __lt__(self, other):
        return self.value < other

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return int(self.value)

    def __nonzero__(self):
        return bool(self.value)

    def modified(self):
        """Returns True if the object has been modified after creation."""
        return False

    @classmethod
    def wrap_lines(cls, long_string, wrap):
        """Split long lines into multiple chunks according to the wrap limit.

        Derived classes can override char_wrap if they wish to, e.g., not split
        hex bytes.

        Args:
            long_string: a string_value() representation of the object
            wrap: maximum number of characters per line. 0 or negative wrap
                means no limit. Should be chosen long enough to comfortably fit
                formatted data; otherwise it is simply ignored and output may
                look funny.

        Returns:
           long_string split into lines of at most |wrap| characters each.
        """
        wrap -= wrap % cls.char_wrap
        return print_util.wrap_lines(long_string, wrap)

    def human_readable_lines(self, wrap=80, label=""):
        """A pretty human readable representation of the object.

        Args:
            wrap: maximum number of characters per line. 0 or negative wrap
                means no limit. Should be chosen long enough to comfortably fit
                formatted data; otherwise it is simply ignored and output may
                look funny.
            label: a label prefix.

        Returns:
            a list of line strings of at most |wrap| characters each.
        """
        to_print = str(self)
        formatted_label = label + ": " if label else ""
        if (to_print.find("\n") == -1 and
            (wrap <= 0 or len(to_print) + len(formatted_label) <= wrap)):
            # Fits on one line, like this:
            # label: value
            return [formatted_label + to_print]

        else:
            # Multiline output:
            # label:
            #   firstlongvalueline
            #   secondvalueline
            ret = []
            indent = 2
            if label:
                ret += print_util.wrap_lines(label + ":", wrap)
            return ret + [" " * indent + x for x in
                          self.wrap_lines(to_print, wrap-indent)]


@Universal(1, tag.PRIMITIVE)
class Boolean(Simple):
    """Boolean."""
    _TRUE = "\xff"
    _FALSE = "\x00"

    def _encode_value(self):
        return self._TRUE if self._value else self._FALSE

    @classmethod
    def _convert_value(cls, value):
        return bool(value)

    @classmethod
    def _decode_value(cls, buf, strict=True):
        if len(buf) != 1:
            raise error.ASN1Error("Invalid encoding")

        # Continuing here breaks re-encoding.
        if strict and buf[0] != cls._TRUE and buf[0] != cls._FALSE:
                raise error.ASN1Error("BER encoding of Boolean value: %s" %
                                      buf[0])
        value = False if buf[0] == cls._FALSE else True
        return value


@Universal(2, tag.PRIMITIVE)
class Integer(Simple):
    """Integer."""

    def _encode_value(self):
        return encode_int(self._value)

    @classmethod
    def _convert_value(cls, value):
        return int(value)

    @classmethod
    def _decode_value(cls, buf, strict=True):
        return decode_int(buf, strict=strict)

@Universal(5, tag.PRIMITIVE)
class Null(Simple):
    """Null."""

    def _encode_value(self):
        return ""

    @classmethod
    def _convert_value(cls, value):
        return None

    @classmethod
    def _decode_value(cls, buf, strict=True):
        return None

class ASN1String(Simple):
    """Base class for string types."""

    def _encode_value(self):
        return self._value

    @classmethod
    def _convert_value(cls, value):
        if isinstance(value, str) or isinstance(value, buffer):
            value = str(value)
        elif isinstance(value, ASN1String):
            value = value.value
        else:
            raise TypeError("Cannot convert %s to %s" %
                            (type(value), cls.__name__))
        cls._check_for_illegal_characters(value)
        return value

    @classmethod
    def _check_for_illegal_characters(cls, buf):
        """Raises if there are any illegal characters in string.

        Args:
            buf: string which will be checked for illegal characters

        Raises:
            ASN1Error.
        """
        pass

    @classmethod
    def _decode_value(cls, buf, strict=True):
        if strict:
            cls._check_for_illegal_characters(buf)
        return buf


# Based on https://www.itu.int/rec/T-REC-X.208-198811-W/en
# and http://kikaku.itscj.ipsj.or.jp/ISO-IR/overview.htm
@Universal(19, tag.PRIMITIVE)
class PrintableString(ASN1String):
    """PrintableString."""
    NOT_ACCEPTABLE = re.compile("[^a-zA-Z0-9 '()+,\-./:=?]")
    @classmethod
    def _check_for_illegal_characters(cls, buf):
        search_result = PrintableString.NOT_ACCEPTABLE.search(buf)
        if search_result:
            index = search_result.start()
            raise error.ASN1IllegalCharacter(
                    "Illegal character in PrintableString", buf, index)


@Universal(20, tag.PRIMITIVE)
class TeletexString(ASN1String):
    """TeletexString (aka T61String)."""
    pass


@Universal(22, tag.PRIMITIVE)
class IA5String(ASN1String):
    """IA5String."""
    @classmethod
    def _check_for_illegal_characters(self, buf):
        for index, character in enumerate(buf):
            if ord(character) > 127:
                raise error.ASN1IllegalCharacter(
                        "Illegal character in IA5String", buf, index)


@Universal(26, tag.PRIMITIVE)
class VisibleString(ASN1String):
    """VisibleString (aka ISO646String)."""
    @classmethod
    def _check_for_illegal_characters(self, buf):
        for index, character in enumerate(buf):
            if ord(character) < 32 or ord(character) > 126:
                raise error.ASN1IllegalCharacter(
                        "Illegal character in VisibleString", buf, index)


@Universal(30, tag.PRIMITIVE)
class BMPString(ASN1String):
    """BMPString."""
    pass


@Universal(12, tag.PRIMITIVE)
class UTF8String(ASN1String):
    """UTF8String."""
    pass


@Universal(28, tag.PRIMITIVE)
class UniversalString(ASN1String):
    """UniversalString."""
    pass


@Universal(4, tag.PRIMITIVE)
class OctetString(ASN1String):
    """Octet string."""
    char_wrap = 3

    def __str__(self):
        return print_util.bytes_to_hex(self._value)


@Universal(3, tag.PRIMITIVE)
class BitString(Simple):
    """Bit string."""
    char_wrap = 3

    def __str__(self):
        return print_util.bits_to_hex(self._value)

    def _encode_value(self):
        pad = (8 - len(self._value) % 8) % 8
        padded_bits = self._value + pad*"0"
        ret = bytearray([pad])
        for i in range(0, len(padded_bits), 8):
            ret.append(int(padded_bits[i:i+8], 2))
        return str(ret)

    def _convert_value(self, value):
        """The value of a BitString is a string of '0's and '1's."""
        if isinstance(value, BitString):
            return value.value
        elif isinstance(value, str):
            # Must be a string of '0's and '1's.
            if not all(c == "0" or c == "1" for c in value):
                raise ValueError("Cannot initialize a BitString from %s:"
                                 "string must consist of 0s and 1s" % value)
            return value
        else:
            raise TypeError("Cannot initialize a BitString from %s"
                            % type(value))

    @classmethod
    def _decode_value(cls, buf, strict=True):
        if not buf:
            raise error.ASN1Error("Invalid encoding: empty %s value" %
                                  cls.__name__)
        int_bytes = bytearray(buf)
        pad = int_bytes[0]
        if pad > 7:
            raise error.ASN1Error("Invalid padding %d in %s" %
                                  (pad, cls.__name__))
        ret = "".join(format(b, "08b") for b in int_bytes[1:])
        if pad:
            if not ret or any([c == "1" for c in ret[-1*pad:]]):
                raise error.ASN1Error("Invalid padding")
            ret = ret[:-1*pad]
        return ret

class NamedBitList(BitString):
    """A bit string with named bits."""
    # To use the NamedBitList ASN.1 construct, set named_bit_list
    # to a tuple of NamedValue instances, where the name of each NamedValue
    # corresponds to the identifier and the value to the number of the
    # distinguished bit, defined by "number" or "DefinedValue" in ASN.1,
    # see http://www.itu.int/ITU-T/studygroups/com17/languages/X.680-0207.pdf
    named_bit_list = None
    char_wrap = 1

    def __str__(self):
            return ", ".join(["%s" % n.name for n in self.bits_set()])

    def has_bit_set(self, number):
        """Test if the given bit is set.

        Args:
            number: the number of the ASN.1 bit. Bit numbering follows ASN.1
                conventions, i.e., bit number 0 is the "leading bit".

        Returns:
            True: the bit is 1.
            False: the bit is 0, or the BitString is not long enough.
        """
        # According to
        # http://www.itu.int/ITU-T/studygroups/com17/languages/X.680-0207.pdf
        # we must not assume that the presence of named bits constrains the
        # contents of the bit string:
        # "21.6 The presence of a "NamedBitList" has no effect on the set of
        # abstract values of this type. Values containing 1 bits other than the
        # named bits are permitted.
        # 21.7 When a "NamedBitList" is used in defining a bitstring type ASN.1
        # encoding rules are free to add (or remove) arbitrarily any trailing 0
        # bits to (or from) values that are being encoded or decoded.
        # Application designers should therefore ensure that different semantics
        # are not associated with such values which differ only in the number of
        # trailing 0 bits.
        return len(self._value) > number and self._value[number] == "1"

    def bits_set(self):
        """List the named_bit_list elements whose bit is set."""
        return [n for n in self.named_bit_list if self.has_bit_set(n.value)]


class Any(ASN1String):
    """Any.

    Any is a container for an arbitrary value. An Any type can be tagged with
    explicit tags like any other type: those tags will be applied to the
    underlying value. Implicit tagging of Any types is not supported.

    The value of an Any is an undecoded raw string. In addition, Any can hold
    the decoded value of the object.
    """
    char_wrap = 3

    def __init__(self, value=None, serialized_value=None, strict=True):
        if isinstance(value, str):
            super(Any, self).__init__(value=None, serialized_value=value,
                                      strict=strict)
            self._decoded_value = None
        else:
            super(Any, self).__init__(value=value,
                                      serialized_value=serialized_value,
                                      strict=strict)
            self._decoded_value = value

    def __repr__(self):
        if self._decoded_value is not None:
            return "%s(%r)" % (self.__class__.__name__, self._decoded_value)
        return "%s(%r)" % (self.__class__.__name__, self._value)

    def __str__(self):
        if self._decoded_value is not None:
            return str(self._decoded_value)
        return print_util.bytes_to_hex(self._value)

    def human_readable_lines(self, wrap=80, label=""):
        """A pretty human readable representation of the object.

        Args:
            wrap: maximum number of characters per line. 0 or negative wrap
                means no limit. Should be chosen long enough to comfortably fit
                formatted data; otherwise it is simply ignored and output may
                look funny.
            label: a label prefix.

        Returns:
            a list of line strings of at most |wrap| characters each.
        """
        if self._decoded_value is not None:
            return self._decoded_value.human_readable_lines(wrap=wrap,
                                                            label=label)
        return super(Any, self).human_readable_lines(wrap=wrap, label=label)

    def modified(self):
        if self._decoded_value is not None:
            return self._decoded_value.modified()
        return False

    def _encode_value(self):
        if self._decoded_value is not None and self._decoded_value.modified():
            return self._decoded_value.encode()
        return self._value

    @property
    def decoded(self):
        return self._decoded_value is not None

    @property
    def decoded_value(self):
        return self._decoded_value

    @classmethod
    def _read(cls, buf, strict=True):
       readahead_tag, rest = tag.Tag.read(buf)
       length, rest = read_length(rest, strict=(
           strict or readahead_tag.encoding != tag.CONSTRUCTED))
       if length == -1:
           # Not sure if this even makes any sense.
           raise NotImplementedError("Indefinite length encoding of ANY types "
                                     "is not supported")
       if len(rest) < length:
           raise error.ASN1Error("Invalid length encoding")
       decoded_length = len(buf) - len(rest) + length
       return cls(serialized_value=buf[:decoded_length],
                  strict=strict), buf[decoded_length:]

    @classmethod
    def _convert_value(cls, value):
        """The value of an Any is the undecoded value."""
        # Always return the undecoded value for consistency; the
        # decoded/decoded_value properties can be used to retrieve the
        # decoded contents.
        if isinstance(value, Any):
            # This gets ambiguous real fast (do we keep the original tags or
            # replace with our own tags?) so we ban it.
            raise TypeError("Instantiating Any from another Any is illegal")
        elif isinstance(value, Abstract):
            return value.encode()
        else:
            raise TypeError("Cannot convert %s to %s" % (type(value),
                                                         cls.__name__))

    @classmethod
    def _decode_value(cls, buf, strict=True):
        return buf

    def decode_inner(self, value_type, strict=True):
        """Decode the undecoded contents according to a given specification.

        Args:
            value_type: an ASN.1 type.
            strict: if False, tolerate some non-fatal decoding errors.

        Raises:
            ASN1Error: decoding failed.
            RuntimeError: value already decoded.
        """
        self._decoded_value = value_type.decode(self._value, strict=strict)


class Constructed(Abstract):
    """Constructed types."""
    print_labels = True
    print_delimiter = "\n"

    def __init__(self, value=None, serialized_value=None, strict=True):
        """Initialize from a value or serialized buffer.

        Args:
            value: initializing value of an appropriate type. If the
                serialized_value is not set, the initializing value must be set.
            serialized_value: serialized inner value (with tags and lengths
                stripped).
            strict: if False, tolerate some non-fatal decoding errors.

        Raises:
            error.ASN1Error: decoding the serialized value failed.
            TypeError: invalid initializer.
        """
        super(Constructed, self).__init__(value=value,
                                          serialized_value=serialized_value,
                                          strict=strict)
        # All methods that mutate the object must set this to True.
        self._modified = False

    def modified(self):
        return self._modified or any([v and v.modified()
                                      for _, v in self.iteritems()])

    def human_readable_lines(self, wrap=80, label=""):
        """A pretty human readable representation of the object.

        Args:
            wrap: maximum number of characters per line. 0 or negative wrap
                means no limit. Should be chosen long enough to comfortably fit
                formatted data; otherwise it is simply ignored and output may
                look funny.
            label: a label prefix.

        Returns:
            a list of line strings of at most |wrap| characters each.
        """
        # A "\n" becomes ["", ""] which magically starts a new line when we call
        # append_lines() on it. Things like "\n-----\n" work, too.
        delimiter = (print_util.wrap_lines(self.print_delimiter, wrap=wrap))
        lines = []

        # Component count. Needed so we can print "<no components>" when none
        # are found.
        count = 0
        # Whether the next component should start on a new line. Set to true
        # when the previous component was multiline. For example, a mix of short
        # and long components with a ", " delimiter is thus printed as
        # short1, short2, short3,
        # myextremelylongcomponentth
        # atspansmultiplelines
        # short4, short5
        newline = False

        if label:
            lines += print_util.wrap_lines(label + ":", wrap)
            # If the delimiter is multiline, then output looks prettier if the
            # label is also on a separate line.
            if len(delimiter) > 1:
                newline = True
            elif len(lines[-1]) < wrap:
                # Else add a whitespace so we get "label: value"
                lines[-1] += " "

        indent = 2
        for key, value in self.iteritems():
            if value is None:
                continue
            label = str(key) if self.print_labels else ""
            print_component = value.human_readable_lines(wrap=wrap-indent,
                                                         label=label)
            if not print_component:
                continue

            if count:
                print_util.append_lines(delimiter, wrap, lines)
            count += 1
            # Make multiline components a separate block on a new line, unless
            # we already are on a new line.
            if (newline or len(print_component) > 1) and lines and lines[-1]:
                lines += print_component
            else:
                print_util.append_lines(print_component, wrap, lines)

            newline = len(print_component) > 1

        if not count:
            print_util.append_lines(["<no components>"], wrap, lines)

        # Indent everything apart from the first line.
        return [lines[0]] + ["  " + x for x in lines[1:]]


class MetaChoice(abc.ABCMeta):
    """Metaclass for building a Choice type."""

    def __new__(mcs, name, bases, dic):
        # Build a tag -> component_name map for the decoder.
        components = dic.get("components", {})
        if components:
            tag_map = {}
            keys_seen = set()
            for key, spec in components.iteritems():
                if key in keys_seen:
                    raise TypeError("Duplicate name in Choice specification")
                keys_seen.add(key)

                if not spec.tags:
                    raise TypeError("Choice type cannot have untagged "
                                    "components")
                if spec.tags[-1] in tag_map:
                    raise TypeError("Duplicate outer tag in a Choice "
                                    "specification")
                tag_map[spec.tags[-1]] = key
            dic["tag_map"] = tag_map
        return super(MetaChoice, mcs).__new__(mcs, name, bases, dic)


class Choice(Constructed, collections.MutableMapping):
    """Choice."""
    __metaclass__ = MetaChoice

    # There is only ever one component anyway.
    print_delimiter = ""
    print_labels = False

    def __init__(self, value=None, serialized_value=None,
                 readahead_tag=None, readahead_value=None, strict=True):
        """Initialize fully or partially.

        Args:
            value: if present, should be a dictionary with one entry
                representing the chosen key and value.
            serialized_value: if present, the serialized contents (with tags
                and lengths stripped).
            readahead_tag: if present, the first tag in serialized_value
            readahead_value: if present, the value wrapped by the first tag in
                serialized value.
            strict: if False, tolerate some non-fatal decoding errors.

        Raises:
            ValueError: invalid initializer value.
        """
        if readahead_tag is not None:
            self._value = self._decode_readahead_value(
                serialized_value, readahead_tag, readahead_value,
                strict=strict)
            self._serialized_value = serialized_value
            self._modified = False
        else:
            super(Choice, self).__init__(value=value,
                                         serialized_value=serialized_value,
                                         strict=strict)

    def __getitem__(self, key):
        value = self._value.get(key, None)
        if value is not None:
            return value
        elif key in self.components:
            return None
        raise KeyError("Invalid key %s for %s" % (key, self.__class__.__name__))

    def __setitem__(self, key, value):
        spec = self.components[key]
        if value is None:
            self._value = {}
        elif type(value) is spec:
            self._value = {key: value}
        # If the supplied value is not of the exact same type then we try to
        # construct one.
        else:
            self._value = {key: spec(value)}
        self._modified = True

    def __delitem__(self, key):
        if key in self._value:
            self._value = {}
        # Raise if the key is invalid; else do nothing.
        elif key not in self.components:
            raise KeyError("Invalid key %s" % key)
        self._modified = True

    def __iter__(self):
        return iter(self._value)

    def __len__(self):
        return len(self._value)

    @property
    def value(self):
        return dict(self._value)

    def component_key(self):
        if not self._value:
            return None
        return self._value.keys()[0]

    # A slightly unfortunate overload of the term "value"...
    def component_value(self):
        if not self._value:
            return None
        return self._value.values()[0]

    def _encode_value(self):
        if not self._value:
            raise error.ASN1Error("Choice component not set")
        # Encode the single component.
        return self._value.values()[0].encode()

    @classmethod
    def _read(cls, buf, strict=True):
        readahead_tag, rest = tag.Tag.read(buf)
        length, rest = read_length(rest, strict=(
            strict or readahead_tag.encoding != tag.CONSTRUCTED))
        if length == -1:
            raise NotImplementedError("Indefinite length encoding of CHOICE "
                                      "type is not supported")
        if len(rest) < length:
            raise error.ASN1Error("Invalid length encoding")
        decoded_length = len(buf) - len(rest) + length
        return (cls(serialized_value=buf[:decoded_length],
                    readahead_tag=readahead_tag, readahead_value=rest[:length],
                    strict=strict),
                buf[decoded_length:])

    @classmethod
    def _convert_value(cls, value):
        if not value:
            return dict()
        if len(value) != 1:
            raise ValueError("Choice must have at most one component set")

        key, value = value.iteritems().next()
        if value is None:
            return {}

        try:
            spec = cls.components[key]
        except KeyError:
            raise ValueError("Invalid Choice key %s" % key)
        if type(value) is spec:
            return {key: value}
        # If the supplied value is not of the exact same type then we try to
        # construct one.
        else:
            return {key: spec(value)}

    @classmethod
    def _decode_readahead_value(cls, buf, readahead_tag, readahead_value,
                                strict=True):
        """Decode using additional information about the outermost tag."""
        try:
            key = cls.tag_map[readahead_tag]
        except KeyError:
            raise error.ASN1TagError("Tag %s is not a valid tag for a "
                                     "component of %s" %
                                     (readahead_tag, cls.__name__))

        if len(cls.components[key].tags) == 1:
            # Shortcut: we already know the tag and length, so directly get
            # the value.
            value = cls.components[key](serialized_value=readahead_value,
                                        strict=strict)
        else:
            # Component has multiple tags but the readahead only read the
            # outermost tag, so read everything again.
            value, rest = cls.components[key].read(buf, strict=strict)
            if rest:
                raise error.ASN1Error("Invalid encoding: leftover bytes when "
                                      "decoding %s" % cls.__name__)
        return {key: value}

    @classmethod
    def _decode_value(cls, buf, strict=True):
        readahead_tag, rest = tag.Tag.read(buf)
        length, rest = read_length(rest, strict=strict)
        if length == -1:
            if readahead_tag.encoding != tag.CONSTRUCTED:
                raise error.ASN1Error("Indefinite length encoding in primitive "
                                      "type")
            raise NotImplementedError("Indefinite length encoding of CHOICE "
                                      "type is not supported")

        if len(rest) != length:
            raise error.ASN1Error("Invalid length encoding")
        return cls._decode_readahead_value(buf, readahead_tag, rest,
                                           strict=strict)


class Repeated(Constructed, collections.MutableSequence):
    """Base class for SetOf and SequenceOf."""

    def __getitem__(self, index):
        return self._value[index]

    def __setitem__(self, index, value):
        # We are required to support both single-value as well as slice
        # assignment.
        if isinstance(index, slice):
            self._value[index] = self._convert_value(value)
        else:
            self._value[index] = (value if type(value) is self.component
                                  else self.component(value))
        self._modified = True

    def __delitem__(self, index):
        del self._value[index]
        self._modified = True

    def __len__(self):
        return len(self._value)

    def iteritems(self):
        return enumerate(self._value)

    def insert(self, index, value):
        if type(value) is not self.component:
            value = self.component(value)
        self._value.insert(index, value)
        self._modified = True

    @property
    def value(self):
        return list(self._value)

    @classmethod
    def _convert_value(cls, value):
        return [x if type(x) is cls.component else cls.component(x)
                for x in value]


@Universal(16, tag.CONSTRUCTED)
class SequenceOf(Repeated):
    """Sequence Of."""

    def _encode_value(self):
        ret = [x.encode() for x in self._value]
        return "".join(ret)

    @classmethod
    def _decode_value(cls, buf, strict=True):
        ret = []
        while buf:
            value, buf = cls.component.read(buf, strict=strict)
            ret.append(value)
        return ret

    @classmethod
    def _read_indefinite_value(cls, buf):
        ret = []
        while len(buf) >= 2:
            if buf[:2] == _EOC:
                return ret, buf[2:]
            value, buf = cls.component.read(buf, strict=False)
            ret.append(value)
        raise error.ASN1Error("Missing EOC octets")

# We cannot use a real set to represent SetOf because
# (a) our components are mutable and thus not hashable and
# (b) ASN.1 allows duplicates: {1} and {1, 1} are distinct sets.
# Note that this means that eq-comparison is order-dependent.
@Universal(17, tag.CONSTRUCTED)
class SetOf(Repeated):
    """Set Of."""

    def _encode_value(self):
        ret = [x.encode() for x in self._value]
        ret.sort()
        return "".join(ret)

    @classmethod
    def _decode_value(cls, buf, strict=True):
        ret = []
        while buf:
            value, buf = cls.component.read(buf, strict=strict)
            ret.append(value)
        # TODO(ekasper): reject BER encodings in strict mode, i.e.,
        # verify sort order.
        return ret

    @classmethod
    def _read_indefinite_value(cls, buf):
        ret = []
        while len(buf) >= 2:
            if buf[:2] == _EOC:
                # TODO(ekasper): reject BER encodings in strict mode, i.e.,
                # verify sort order.
                return ret, buf[2:]
            value, buf = cls.component.read(buf, strict=False)
            ret.append(value)
        raise error.ASN1Error("Missing EOC octets")


class Component(object):
    """Sequence component specification."""

    def __init__(self, name, value_type, optional=False, default=None,
                 defined_by=None, lookup=None):
        """Define a sequence component.

        Args:
            name: component name. Must be unique within a sequence.
            value_type: the ASN.1 type.
            optional: if True, the component is optional.
            default: default value of the component.
            defined_by: for Any types, this specifies the component
                that defines the type.
            lookup: the lookup dictionary for Any types.
        """
        self.name = name
        self.value_type = value_type
        if default is None or type(default) is value_type:
            self.default = default
        else:
            self.default = value_type(default)
        if self.default is not None:
            self.encoded_default = self.default.encode()
        else:
            self.encoded_default = None
        self.optional = optional or (self.default is not None)
        self.defined_by = defined_by
        self.lookup = lookup


class MetaSequence(abc.ABCMeta):
    """Metaclass for building Sequence types."""

    def __new__(mcs, name, bases, dic):
        # Build a key -> component map for setting values.
        components = dic.get("components", ())
        if components:
            key_map = {}
            for component in components:
                if component.name in key_map:
                    raise TypeError("Duplicate name in Sequence specification")
                key_map[component.name] = component
            dic["key_map"] = key_map
        return super(MetaSequence, mcs).__new__(mcs, name, bases, dic)


@Universal(16, tag.CONSTRUCTED)
class Sequence(Constructed, collections.MutableMapping):
    """Sequence."""
    __metaclass__ = MetaSequence

    def __getitem__(self, key):
        return self._value[key]

    def __setitem__(self, key, value):
        component = self.key_map[key]
        value = self._convert_single_value(component, value)
        self._value[key] = value
        self._modified = True

    def __delitem__(self, key):
        if key not in self.key_map:
            raise KeyError("Invalid key %s" % key)
        self[key] = None
        self._modified = True

    def __iter__(self):
        """Iterate component names in order."""
        for component in self.components:
            yield component.name

    def __len__(self):
        """Missing optional components are counted in the length."""
        return len(self.components)

    @property
    def value(self):
        # Note that this does not preserve the component order.
        # However an order is encoded in the type spec, so we can still
        # recreate the original object from this value.
        return dict(self._value)

    def _encode_value(self):
        ret = []
        for component in self.components:
            value = self._value[component.name]
            if value is None:
                if not component.optional:
                    raise error.ASN1Error("Missing %s value in %s" %
                                          (component.name,
                                           self.__class__.__name__))
            else:
                # Value is not None.
                # We could compare by value for most types, but for "set" types
                # different values may yield the same encoding, so we compare
                # directly by encoding.
                # (Even though I haven't seen a defaulted set type in practice.)
                encoded_value = value.encode()
                if component.encoded_default != encoded_value:
                    ret.append(encoded_value)
        return "".join(ret)

    @classmethod
    def _convert_single_value(cls, component, value):
        # If value is None, we store the default if it is different from None.
        if value is None:
            return component.default
        elif type(value) is component.value_type:
            return value
        # If the supplied value is not of the exact same type then we discard
        # the tag information and try to construct from scratch.
        else:
            # TODO(ekasper): verify defined_by constraints here.
            return component.value_type(value)

    @classmethod
    def _convert_value(cls, value):
        ret = {}
        value = value or {}
        if not all([key in cls.key_map for key in value]):
            raise ValueError("Invalid keys in initializer")
        for component in cls.components:
            ret[component.name] = cls._convert_single_value(
                component, value.get(component.name, None))
        return ret

    @classmethod
    def _read_value(cls, buf, strict=True):
        ret = dict()
        for component in cls.components:
            try:
                value, buf = component.value_type.read(buf, strict=strict)
            except error.ASN1TagError:
                # If the component was optional and we got a tag mismatch,
                # assume decoding failed because the component was missing,
                # and carry on.
                # TODO(ekasper): since we let errors fall through recursively,
                # not all of the tag errors can be reasonably explained by
                # missing optional components. We could tighten this to match by
                # outermost tag only, and have metaclass verify the uniqueness
                # of component tags. Meanwhile, the worst that can happen is
                # that we retry in vain and don't return the most helpful error
                # message when we do finally fail.
                if not component.optional:
                    raise
                else:
                    ret[component.name] = component.default
            else:
                ret[component.name] = value

        # Second pass for decoding ANY.
        for component in cls.components:
            if component.defined_by is not None:
                value_type = component.lookup.get(
                    ret[component.defined_by], None)
                if value_type is not None:
                    try:
                        ret[component.name].decode_inner(value_type,
                                                         strict=strict)
                    except error.ASN1Error:
                        if strict:
                            raise
        return ret, buf

    @classmethod
    def _decode_value(cls, buf, strict=True):
        ret, buf = cls._read_value(buf, strict=strict)
        if buf:
            raise error.ASN1Error("Invalid encoding")
        return ret

    @classmethod
    def _read_indefinite_value(cls, buf):
        # We must be in strict=False mode by definition.
        ret, buf = cls._read_value(buf, strict=False)
        if buf[:2] != _EOC:
            raise error.ASN1Error("Missing EOC octets")
        return ret, buf[2:]
