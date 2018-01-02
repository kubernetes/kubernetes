from ct.crypto import error
from ct.crypto import pem
from ct.crypto.asn1 import types
from ct.proto import client_pb2

import hashlib
import ecdsa

class _ECDSASignature(types.Sequence):
    components = (
            (types.Component("r", types.Integer)),
            (types.Component("s", types.Integer))
    )

class EcdsaVerifier(object):
    """Verifies ECDSA signatures."""

    # The signature algorithm used for this public key."""
    SIGNATURE_ALGORITHM = client_pb2.DigitallySigned.ECDSA
    # The hash algorithm used for this public key."""
    HASH_ALGORITHM = client_pb2.DigitallySigned.SHA256

    # Markers to look for when reading a PEM-encoded ECDSA public key."""
    __READ_MARKERS = ("PUBLIC KEY", "ECDSA PUBLIC KEY")
    # A marker to write when writing a PEM-encoded ECDSA public key."""
    __WRITE_MARKER = "ECDSA PUBLIC KEY"

    def __init__(self, key_info):
        """Creates a verifier that uses a PEM-encoded ECDSA public key.

        Args:
        - key_info: KeyInfo protobuf message

        Raises:
        - PemError: If the key has an invalid encoding
        """
        if (key_info.type != client_pb2.KeyInfo.ECDSA):
            raise error.UnsupportedAlgorithmError(
                "Expected ECDSA key, but got key type %d" % key_info.type)

        # Will raise a PemError on invalid encoding
        self.__der, _ = pem.from_pem(key_info.pem_key, self.__READ_MARKERS)
        try:
            self.__key = ecdsa.VerifyingKey.from_der(self.__der)
        except ecdsa.der.UnexpectedDER as e:
            raise error.EncodingError(e)

    def __repr__(self):
        return "%s(public key: %r)" % (self.__class__.__name__,
                                       pem.to_pem(self.__der,
                                                  self.__WRITE_MARKER))

    @error.returns_true_or_raises
    def verify(self, signature_input, signature):
        """Verifies the signature was created by the owner of the public key.

        Args:
        - signature_input: The data that was originally signed.
        - signature: An ECDSA SHA256 signature.

        Returns:
        - True if the signature verifies.

        Raises:
        - error.EncodingError: If the signature encoding is invalid.
        - error.SignatureError: If the signature fails verification.
        """
        try:
            _ECDSASignature.decode(signature)
            return self.__key.verify(signature, signature_input,
                                     hashfunc=hashlib.sha256,
                                     sigdecode=ecdsa.util.sigdecode_der)
        except (ecdsa.der.UnexpectedDER, error.ASN1Error) as e:
            raise error.EncodingError("Invalid DER encoding for signature %s",
                                      signature.encode("hex"), e)
        except ecdsa.keys.BadSignatureError:
            raise error.SignatureError("Signature did not verify: %s",
                                       signature.encode("hex"))

