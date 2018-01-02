"""Status codes are bad, but True/False is not expressive enough.

Consider a cryptographic signature verification method verify(data, sig) that
returns 1 for valid signatures, 0 for invalid signatures, and -1 to signal some
specific error. This can easily lead to insecure usage such as
if verify(data, sig):
    # do stuff on success

Or, here's another example, borrowed from real code:
r, s = asn1_decode(sig)  # raises ASN1Error
return verify_sig(data, r, s) # returns True/False

A caller may obviously be interested in distinguishing a decoding error from a
signature computation error - but why is a computation error False while a
decoding error is an exception? What other exceptions might this code raise?
This is a nightmare for the caller to handle.

Therefore, methods in the crypto package that verify a property return True
when verification succeeds and raise an exception on any error. This minimises
the risk of uncaught errors, allows to provide information for callers that care
about the specific failure reason, and makes failure handling easy for callers
that do not care:

try:
    verify(myargs)
except MyError:
    # handle specific error here
    return
except VerifyError:
    # verify failed, we don't care why
    return
# do more stuff on success here

Returning True is strictly speaking not needed but simplifies testing.
We provide a defensive returns_true_or_raises wrapper for ensuring this
behaviour: callers of methods decorated with @returns_true_or_raises can be
certain that the _only_ value the method returns is True - it never returns
None, or False, or [], or anything else.
"""

import functools


class Error(Exception):
    """Exceptions raised by the crypto subpackage."""
    pass


class UnsupportedAlgorithmError(Error):
    """An algorithm is not implemented or supported."""
    pass


class VerifyError(Error):
    """Some expected property of the input cannot be verified.

    The property either verifiably does not hold, or cannot be conclusively
    verified. Domain-specific verification errors inherit from this class.
    """
    pass


class ConsistencyError(VerifyError):
    """There is a (cryptographic) inconsistency in the data."""
    pass


class ProofError(VerifyError):
    """A cryptographic proof is not valid.

    This error does not necessarily indicate that the sought property does not
    hold but rather that the given data is insufficient for verifying the
    desired property.
    """
    pass


# TODO(ekasper): TBD if this hierarchy is appropriate.
class EncodingError(Error):
    """Encoding/decoding error.

    Inputs cannot be serialized, or serialized data cannot be parsed.
    """
    pass


class ASN1Error(EncodingError):
    """An ASN1 object cannot be encoded or decoded."""
    pass


class ASN1TagError(ASN1Error):
    """ASN1 tag mismatch."""
    pass


class UnknownASN1TypeError(ASN1Error):
    """An OID does not map to a known ASN.1 type."""
    pass

class ASN1IllegalCharacter(ASN1Error):
    """String contains illegal character."""
    def __init__(self, message, string, index, *args):
        self.message = message
        self.string = string
        self.index = index
        super(ASN1Error, self).__init__(message, *args)

    def __str__(self):
        return "%s (string: %s, character: %s, index: %d)" % (self.message,
                                                              self.string,
                                                              self.string[
                                                                    self.index],
                                                              self.index)

class IncompleteChainError(VerifyError):
    """A certificate is missing from the chain"""
    pass

class SignatureError(VerifyError):
    """A public-key signature does not verify."""
    pass


class UnsupportedVersionError(Error):
    """The version of the data structure is unknown."""
    pass


def returns_true_or_raises(f):
    """A safety net.

    Decorator for functions that are only allowed to return True or raise
    an exception.

    Args:
        f: A function whose only expected return value is True.

    Returns:
        A wrapped functions whose guaranteed only return value is True.
    """
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        ret = f(*args, **kwargs)
        if ret is not True:
            raise RuntimeError("Unexpected return value %r" % ret)
        return True
    return wrapped
