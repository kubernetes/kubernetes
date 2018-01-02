"""ASN.1 UTCTime and GeneralizedTime, as understood by RFC 5280."""
import abc
import time

from ct.crypto import error
from ct.crypto.asn1 import tag
from ct.crypto.asn1 import types


class BaseTime(types.ASN1String):
    """Base class for time types."""
    def __init__(self, value=None, serialized_value=None, strict=True):
        super(BaseTime, self).__init__(value=value,
                                       serialized_value=serialized_value,
                                       strict=strict)
        self._gmtime = self._decode_gmtime(strict=strict)
        # This is a lenient "strict": if we were able to decode the time,
        # even if it didn't fully conform to the standard, then we'll allow it.
        # If the time string is garbage then we raise.
        if strict and self._gmtime is None:
            raise error.ASN1Error("Corrupt time: %s" % self._value)

    def gmtime(self):
        """GMT time.

        Returns:
            a time.struct_time struct.

        Raises:
            error.ASN1Error: the ASN.1 string does not represent a valid time.
        """
        if self._gmtime is None:
            raise error.ASN1Error("Corrupt time: %s" % self._value)
        return self._gmtime

    @abc.abstractmethod
    def _decode_gmtime(self, strict):
        pass

    def __str__(self):
        try:
            return time.strftime("%c GMT", self.gmtime())
        except error.ASN1Error:
            return str(self.value)


@types.Universal(23, tag.PRIMITIVE)
class UTCTime(BaseTime):
    """UTCTime, as understood by RFC 5280."""
    # YYMMDDHHMMSSZ
    _ASN1_LENGTH = 13

    # YYMMDDHHMMZ
    _UTC_NO_SECONDS_LENGTH = 11

    # YYMMDDHHMMSS+HHMM
    _UTC_TZ_OFFSET_LENGTH = 17

    # YYMMDDHHMMSS
    _UTC_NO_Z_LENGTH = 12

    def _decode_gmtime(self, strict):
        """GMT time.

        Returns:
            a time.struct_time struct, or None if the string does not represent
                a valid time.
        """
        # From RFC 5280:
        # For the purposes of this profile, UTCTime values MUST be expressed in
        # Greenwich Mean Time (Zulu) and MUST include seconds (i.e., times are
        # YYMMDDHHMMSSZ), even where the number of seconds is zero.  Conforming
        # systems MUST interpret the year field (YY) as follows:
        #
        # Where YY is greater than or equal to 50, the year SHALL be
        # interpreted as 19YY; and
        #
        # Where YY is less than 50, the year SHALL be interpreted as 20YY.
        #
        # In addition, there are a number of older certificates
        # that exclude the seconds, e.g. 0001010000Z and others than use
        # an alternative timezone format 360526194526+0000
        string_time = self.value

        if len(string_time) == self._ASN1_LENGTH and string_time[-1] == "Z":
            format = "%Y%m%d%H%M%S%Z"
        elif (len(string_time) == self._UTC_NO_SECONDS_LENGTH and
              string_time[-1] == "Z"):
            format = "%Y%m%d%H%M%Z"
        elif (len(string_time) == self._UTC_TZ_OFFSET_LENGTH and
              string_time[self._UTC_NO_Z_LENGTH] in ('+','-')):
            # note according to http://docs.python.org/2/library/time.html
            # "%z" is not supported on all platforms.
            #
            # TBD: in next patch, parse this correctly
            #
            # Given that it's very infrequent and non-standard,
            # we'll ignore time zone for now.
            #
            # convert the +HHMM to a timedelta and add to timestruct
            # One could also special case the "+0000" which should be the same
            # as GMT (without DST).
            #
            format = "%Y%m%d%H%M%S%Z"
            string_time = string_time[0:self._ASN1_LENGTH]
        elif (len(string_time) == self._UTC_NO_Z_LENGTH) and not strict:
            string_time += "Z"
            format = "%Y%m%d%H%M%S%Z"
        else:
            return None

        try:
            year = int(string_time[:2])
        except ValueError:
            return None

        if 0 <= year < 50:
            century = "20"
        elif 50 <= year <= 99:
            century = "19"
        else:
            return None

        try:
            # Adding GMT clears the daylight saving flag.
            return time.strptime(century + string_time[:-1] + "GMT", format)
        except ValueError:
            return None


@types.Universal(24, tag.PRIMITIVE)
class GeneralizedTime(BaseTime):
    """Generalized time, as understood by RFC 5280."""
    # YYYYMMDDHHMMSSZ
    _ASN1_LENGTH = 15

    def _decode_gmtime(self, strict):
        """GMT time.

        Returns:
            a time.struct_time struct, or None if the string does not represent
                a valid time.
        """
        # From RFC 5280:
        # For the purposes of this profile, GeneralizedTime values MUST be
        # expressed in Greenwich Mean Time (Zulu) and MUST include seconds
        # (i.e., times are YYYYMMDDHHMMSSZ), even where the number of seconds
        # is zero.  GeneralizedTime values MUST NOT include fractional seconds.
        if len(self._value) != self._ASN1_LENGTH or self._value[-1] != "Z":
            return None
        try:
            # Adding GMT clears the daylight saving flag.
            return time.strptime(self._value[:-1] + "GMT", "%Y%m%d%H%M%S%Z")
        except ValueError:
            return None


class Time(types.Choice):
    print_labels = False
    components = {"utcTime": UTCTime,
                  "generalTime": GeneralizedTime}


class Validity(types.Sequence):
    components = (
        (types.Component("notBefore", Time)),
        (types.Component("notAfter", Time))
        )
