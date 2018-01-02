from ct.crypto import error
from ct.crypto import pem
from ct.proto import client_pb2

import Crypto.Hash.SHA256
import Crypto.PublicKey.RSA
import Crypto.Signature.PKCS1_v1_5

class RsaVerifier(object):
    """Verifies RSA signatures."""

    # The signature algorithm used for this public key.
    SIGNATURE_ALGORITHM = client_pb2.DigitallySigned.RSA
    # The hash algorithm used for this public key.
    HASH_ALGORITHM = client_pb2.DigitallySigned.SHA256

    # Markers to look for when reading a PEM-encoded RSA public key.
    __READ_MARKERS = ("PUBLIC KEY", "RSA PUBLIC KEY")
    # A marker to write when writing a PEM-encoded RSA public key.
    __WRITE_MARKER = "RSA PUBLIC KEY"

    def __init__(self, key_info):
        """Creates a verifier that uses a PEM-encoded RSA public key.

        Args:
        - key_info: KeyInfo protobuf message

        Raises:
        - PemError: If the key has an invalid encoding
        """
        if (key_info.type != client_pb2.KeyInfo.RSA):
            raise error.UnsupportedAlgorithmError(
                "Expected RSA key, but got key type %d" % key_info.type)

        # Will raise a PemError on invalid encoding
        self.__der, _ = pem.from_pem(key_info.pem_key, self.__READ_MARKERS)
        try:
            self.__key = Crypto.PublicKey.RSA.importKey(self.__der)
        except (ValueError, IndexError, TypeError) as e:
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
        - signature: An RSA SHA256 signature.

        Returns:
        - True if the signature verifies.

        Raises:
        - error.SignatureError: If the signature fails verification.
        """
        verifier = Crypto.Signature.PKCS1_v1_5.new(self.__key)
        sha256_hash = Crypto.Hash.SHA256.new(signature_input)

        if verifier.verify(sha256_hash, signature):
            return True
        else:
            raise error.SignatureError("Signature did not verify: %s",
                                       signature.encode("hex"))

