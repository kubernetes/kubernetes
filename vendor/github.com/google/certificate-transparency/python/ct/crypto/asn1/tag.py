"""ASN.1 tagging."""

from ct.crypto import error


UNIVERSAL = 0x00
APPLICATION = 0x40
CONTEXT_SPECIFIC = 0x80
PRIVATE = 0xc0
PRIMITIVE = 0x00
CONSTRUCTED = 0x20

# Constants for better readability.
IMPLICIT, EXPLICIT = range(2)


class Tag(object):
    """An ASN.1 tag."""
    _CLASS_MASK = 0xc0
    _ENCODING_MASK = 0x20
    _NUMBER_MASK = 0x1f
    _HIGH = 0x1f
    _FULL_SUB_OCTET = 0x7f
    _LAST_OCTET = 0x80

    def __init__(self, number, tag_class, encoding):
        """ASN.1 tag.

        Initialize a tag from its number, class and encoding.

        Args:
            number: the numeric value of the tag.
            tag_class: must be one of UNIVERSAL, APPLICATION, CONTEXT_SPECIFIC
                or PRIVATE.
            encoding: must be one of PRIMITIVE or CONSTRUCTED.

        Raises:
            ValueError: invalid initializers.
        """
        if tag_class not in (UNIVERSAL, APPLICATION, CONTEXT_SPECIFIC, PRIVATE):
            raise ValueError("Invalid tag class %s" % tag_class)
        if encoding not in (PRIMITIVE, CONSTRUCTED):
            raise ValueError("Invalid encoding %s" % encoding)

        # Public just for lightweight access. Do not modify directly.
        self.number = number
        self.tag_class = tag_class
        self.encoding = encoding
        if number <= 30:
            self.value = chr(tag_class | encoding | number)
        else:
            res = [tag_class | encoding | self._HIGH]
            tmp = []
            while number > 0:
                tmp.append((number & self._FULL_SUB_OCTET) | self._LAST_OCTET)
                number >>= 7
            tmp[0] -= self._LAST_OCTET
            tmp.reverse()
            res += tmp
            self.value = ''.join([chr(byte) for byte in res])

    def __repr__(self):
        return ("%s(%r, %r, %r)" % (self.__class__.__name__, self.number,
                                    self.tag_class, self.encoding))

    def __str__(self):
        return "[%s %d]" % (self.class_name(), self.number)

    def __len__(self):
        return len(self.value)

    def class_name(self):
        if self.tag_class == UNIVERSAL:
            return "UNIVERSAL"
        elif self.tag_class == APPLICATION:
            return "APPLICATION"
        elif self.tag_class == CONTEXT_SPECIFIC:
            return "CONTEXT-SPECIFIC"
        elif self.tag_class == PRIVATE:
            return "PRIVATE"
        else:
            raise ValueError("Invalid tag class %x" % self.tag_class)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Tag):
            return NotImplemented
        return self.value == other.value

    def __ne__(self, other):
        if not isinstance(other, Tag):
            return NotImplemented
        return self.value != other.value

    @classmethod
    def read(cls, buf):
        """Read from the beginning of a string or buffer.

        Args:
            buf: a binary string or string buffer containing an ASN.1 object.

        Returns:
            an tuple consisting of an instance of the class and the remaining
            buffer/string.
        """

        if not buf:
            raise error.ASN1TagError("Ran out of bytes while decoding")
        tag_bytes = 0
        id_byte = ord(buf[tag_bytes])
        tag_class = id_byte & cls._CLASS_MASK
        encoding = id_byte & cls._ENCODING_MASK
        number = id_byte & cls._NUMBER_MASK
        if number == cls._HIGH:
            number = 0
            tag_bytes += 1
            success = False
            for i in range(1, len(buf)):
                number <<= 7
                id_byte = ord(buf[i])
                number |= (id_byte & cls._FULL_SUB_OCTET)
                tag_bytes += 1
                if id_byte & cls._LAST_OCTET == 0:
                    success = True
                    break
            if not success:
                raise error.ASN1TagError("Ran out of bytes while decoding")
            if tag_bytes - 1 > 5:
                raise error.ASN1TagError("Base 128 integer too large")
            tag_bytes -= 1
        tag = cls(number, tag_class, encoding)
        return tag, buf[tag_bytes + 1:]
