"""Verify CT log statements."""

import io
import struct

from ct.crypto import error
from ct.crypto import pem
from ct.crypto import merkle
from ct.crypto import verify_ecdsa
from ct.crypto import verify_rsa
from ct.crypto.asn1 import oid
from ct.crypto.asn1 import x509_common
from ct.crypto.asn1 import x509_extension as x509_ext
from ct.crypto.asn1 import x509_name
from ct.proto import client_pb2
from ct.proto import ct_pb2
from ct.serialization import tls_message

SUPPORTED_SIGNATURE_ALGORITHMS = (
    ct_pb2.DigitallySigned.ECDSA,
    ct_pb2.DigitallySigned.RSA
)

def decode_signature(signature):
    """Decode the TLS-encoded serialized signature.

    Args:
        signature: TLS-encoded signature.

    Returns:
        a tuple of (hash algorithm, signature algorithm, signature data)

    Raises:
        ct.crypto.error.EncodingError: invalid TLS encoding.
    """

    sig_stream = io.BytesIO(signature)

    sig_prefix = sig_stream.read(2)
    if len(sig_prefix) != 2:
        raise error.EncodingError("Invalid algorithm prefix %s" %
                                      sig_prefix.encode("hex"))
    hash_algo, sig_algo = struct.unpack(">BB", sig_prefix)
    if (hash_algo != ct_pb2.DigitallySigned.SHA256 or
        sig_algo not in SUPPORTED_SIGNATURE_ALGORITHMS):
        raise error.EncodingError("Invalid algorithm(s) %d, %d" %
                                  (hash_algo, sig_algo))

    length_prefix = sig_stream.read(2)
    if len(length_prefix) != 2:
        raise error.EncodingError("Invalid signature length prefix %s" %
                                  length_prefix.encode("hex"))
    sig_length, = struct.unpack(">H", length_prefix)
    remaining = sig_stream.read()
    if len(remaining) != sig_length:
        raise error.EncodingError("Invalid signature length %d for "
                                  "signature %s with length %d" %
                                  (sig_length, remaining.encode("hex"),
                                   len(remaining)))
    return (hash_algo, sig_algo, remaining)

def _get_precertificate_issuer(chain):
    try:
        issuer = chain[1]
    except IndexError:
        raise error.IncompleteChainError(
                "Chain with PreCertificate must contain issuer.")

    if not issuer.extended_key_usage(oid.CT_PRECERTIFICATE_SIGNING):
        return issuer
    else:
        try:
            return chain[2]
        except IndexError:
            raise error.IncompleteChainError(
                "Chain with PreCertificate signed by PreCertificate "
                "Signing Cert must contain issuer.")

def _find_extension(asn1, extn_id):
    """Find an extension from a certificate's ASN.1 representation

    Args:
        asn1: x509.Certificate instance.
        extn_id: OID of the extension to look for.

    Returns:
        The decoded value of the extension, or None if not found.
        This is a reference and can be modified.
    """
    for e in asn1["tbsCertificate"]["extensions"]:
        if e["extnID"] == extn_id:
            return e["extnValue"].decoded_value

    return None

def _remove_extension(asn1, extn_id):
    """Remove an extension from a certificate's ASN.1 representation

    Args:
        asn1: x509.Certificate instance.
        extn_id: OID of the extension to be removed.
    """
    asn1["tbsCertificate"]["extensions"] = (
        filter(lambda e: e["extnID"] != extn_id,
               asn1["tbsCertificate"]["extensions"])
    )

def _encode_tbs_certificate_for_validation(cert, issuer):
    """Normalize a Certificate for CT Signing / Verification
    The poison and embedded sct extensions are removed if present,
    and the issuer information is changed to match the one given in
    argument. The resulting TBS certificate is encoded.

    Args:
        cert: Certificate instance to be normalized
        issuer: Issuer certificate used to fix issuer information in TBS
                certificate.

    Returns:
        DER encoding of the normalized TBS Certificate
    """
    asn1 = cert.to_asn1()
    issuer_asn1 = issuer.to_asn1()

    _remove_extension(asn1, oid.CT_POISON)
    _remove_extension(asn1, oid.CT_EMBEDDED_SCT_LIST)
    asn1["tbsCertificate"]["issuer"] = issuer_asn1["tbsCertificate"]["subject"]

    akid = _find_extension(asn1, oid.ID_CE_AUTHORITY_KEY_IDENTIFIER)
    if akid is not None:
        akid[x509_ext.KEY_IDENTIFIER] = issuer.subject_key_identifier()
        akid[x509_ext.AUTHORITY_CERT_SERIAL_NUMBER] = issuer.serial_number()
        akid[x509_ext.AUTHORITY_CERT_ISSUER] = [
            x509_name.GeneralName({
                x509_name.DIRECTORY_NAME: issuer_asn1["tbsCertificate"]["issuer"]
            })
        ]

    return asn1["tbsCertificate"].encode()

def _is_precertificate(cert):
    return (cert.has_extension(oid.CT_POISON) or
            cert.has_extension(oid.CT_EMBEDDED_SCT_LIST))

def _create_dst_entry(sct, chain):
    """Create a Digitally Signed Timestamped Entry to be validated

    Args:
        sct: client_pb2.SignedCertificateTimestamp instance.
        chain: list of Certificate instances.

    Returns:
        client_pb2.DigitallySignedTimestampedEntry instance with all
        fields set.

    Raises:
        ct.crypto.error.IncompleteChainError: a certificate is missing
            from the chain.
    """

    try:
        leaf_cert = chain[0]
    except IndexError:
        raise error.IncompleteChainError(
                "Chain must contain leaf certificate.")

    entry = client_pb2.DigitallySignedTimestampedEntry()
    entry.sct_version = ct_pb2.V1
    entry.signature_type = client_pb2.CERTIFICATE_TIMESTAMP
    entry.timestamp = sct.timestamp
    entry.ct_extensions = sct.extensions

    if _is_precertificate(leaf_cert):
        issuer = _get_precertificate_issuer(chain)

        entry.entry_type = client_pb2.PRECERT_ENTRY
        entry.pre_cert.issuer_key_hash = issuer.key_hash('sha256')
        entry.pre_cert.tbs_certificate = (
            _encode_tbs_certificate_for_validation(leaf_cert, issuer)
        )
    else:
        entry.entry_type = client_pb2.X509_ENTRY
        entry.asn1_cert = leaf_cert.to_der()

    return entry


class LogVerifier(object):
    """CT log verifier."""

    def __init__(self, key_info, merkle_verifier=merkle.MerkleVerifier()):
        """Initialize from KeyInfo protocol buffer and a MerkleVerifier."""
        self.__merkle_verifier = merkle_verifier
        if (key_info.type == client_pb2.KeyInfo.ECDSA):
            self.__sig_verifier = verify_ecdsa.EcdsaVerifier(key_info)
        elif (key_info.type == client_pb2.KeyInfo.RSA):
            self.__sig_verifier = verify_rsa.RsaVerifier(key_info)
        else:
            raise error.UnsupportedAlgorithmError("Key type %d not supported" %
                                                  key_info.type)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.__sig_verifier)

    def _encode_sth_input(self, sth_response):
        if len(sth_response.sha256_root_hash) != 32:
            raise error.EncodingError("Wrong hash length: expected 32, got %d" %
                                      len(sth_response.sha256_root_hash))
        return struct.pack(">BBQQ32s", ct_pb2.V1, ct_pb2.TREE_HEAD,
                           sth_response.timestamp, sth_response.tree_size,
                           sth_response.sha256_root_hash)

    @error.returns_true_or_raises
    def _assert_correct_signature_algorithms(self, hash_algo, sig_algo):
        if (hash_algo != self.__sig_verifier.HASH_ALGORITHM):
            raise error.SignatureError(
                "Hash algorithm used for the signature (%d) does not match the "
                "one used for the public key (%d)" %
                (hash_algo, self.__sig_verifier.HASH_ALGORITHM))

        if (sig_algo != self.__sig_verifier.SIGNATURE_ALGORITHM):
            raise error.SignatureError(
                "Signing algorithm used (%d) does not match the one used for "
                "the public key (%d)" %
                (sig_algo, self.__sig_verifier.SIGNATURE_ALGORITHM))

        return True

    @error.returns_true_or_raises
    def verify_sth(self, sth_response):
        """Verify the STH Response.

        Args:
            sth_response: client_pb2.SthResponse proto. The response must have
                all fields present.

        Returns:
            True. The return value is enforced by a decorator and need not be
                checked by the caller.

        Raises:
            ct.crypto.error.EncodingError: failed to encode signature input,
                or decode the signature.
            ct.crypto.error.SignatureError: invalid signature.
        """
        signature_input = self._encode_sth_input(sth_response)

        (hash_algo, sig_algo, signature) = decode_signature(
            sth_response.tree_head_signature)

        self._assert_correct_signature_algorithms(hash_algo, sig_algo)

        return self.__sig_verifier.verify(signature_input, signature)

    @staticmethod
    @error.returns_true_or_raises
    def verify_sth_temporal_consistency(old_sth, new_sth):
        """Verify the temporal consistency for two STH responses.

        For two STHs, verify that the newer STH has bigger tree size.
        Does not verify STH signatures or consistency of hashes.

        Args:
            old_sth: client_pb2.SthResponse proto. The STH with the older
                timestamp must be supplied first.
            new_sth: client_pb2.SthResponse proto.

        Returns:
            True. The return value is enforced by a decorator and need not be
                checked by the caller.

        Raises:
            ct.crypto.error.ConsistencyError: STHs are inconsistent
            ValueError: "Older" STH is not older.
        """
        if old_sth.timestamp > new_sth.timestamp:
            raise ValueError("Older STH has newer timestamp (%d vs %d), did "
                             "you supply inputs in the wrong order?" %
                             (old_sth.timestamp, new_sth.timestamp))

        if (old_sth.timestamp == new_sth.timestamp and
            old_sth.tree_size != new_sth.tree_size):
            # Issuing two different STHs for the same timestamp is illegal,
            # even if they are otherwise consistent.
            raise error.ConsistencyError("Inconsistency: different tree sizes "
                                         "for the same timestamp")
        if (old_sth.timestamp < new_sth.timestamp and
            old_sth.tree_size > new_sth.tree_size):
            raise error.ConsistencyError("Inconsistency: older tree has bigger "
                                         "size")
        return True

    @error.returns_true_or_raises
    def verify_sth_consistency(self, old_sth, new_sth, proof):
        """Verify consistency of two STHs.

        Verify the temporal consistency and consistency proof for two STH
        responses. Does not verify STH signatures.

        Args:
            old_sth: client_pb2.SthResponse() proto. The STH with the older
                timestamp must be supplied first.
            new_sth: client_pb2.SthResponse() proto.
            proof: a list of SHA256 audit nodes.

        Returns:
            True. The return value is enforced by a decorator and need not be
                checked by the caller.

        Raises:
            ConsistencyError: STHs are inconsistent
            ProofError: proof is invalid
            ValueError: "Older" STH is not older.
        """
        self.verify_sth_temporal_consistency(old_sth, new_sth)
        self.__merkle_verifier.verify_tree_consistency(
            old_sth.tree_size, new_sth.tree_size, old_sth.sha256_root_hash,
            new_sth.sha256_root_hash, proof)
        return True

    @error.returns_true_or_raises
    def verify_sct(self, sct, chain):
        """Verify the SCT over the X.509 certificate provided

        Args:
            sct: client_pb2.SignedCertificateTimestamp proto. Must have
                all fields present.
            chain: list of cert.Certificate instances. Begins with the
                certificate to be checked.

        Returns:
            True. The return value is enforced by a decorator and need not be
                checked by the caller.

        Raises:
            ct.crypto.error.EncodingError: failed to encode signature input,
                or decode the signature.
            ct.crypto.error.SignatureError: invalid signature.
            ct.crypto.error.IncompleteChainError: a certificate is missing
                from the chain.
        """

        if sct.version != ct_pb2.V1:
            raise error.UnsupportedVersionError("Cannot handle version: %s" %
                                                sct.version)
        entry = _create_dst_entry(sct, chain)
        signature_input = tls_message.encode(entry)

        self._assert_correct_signature_algorithms(sct.signature.hash_algorithm,
                                                  sct.signature.sig_algorithm)

        return self.__sig_verifier.verify(signature_input,
                                          sct.signature.signature)

    def verify_embedded_scts(self, chain):
        """Extract and verify SCTs embedded in an X.509 certificate.

        Args:
            chain: list of cert.Certificate instances.

        Returns:
            List of (SignedCertificateTimestamp, bool) pairs, one for each SCT
                present in the certificate. The boolean is True if the
                corresponding SCT is valid, False otherwise.

        Raises:
            ct.crypto.error.EncodingError: failed to encode signature input,
                or decode the signature.
            ct.crypto.error.IncompleteChainError: the chain is empty.
        """

        try:
            leaf_cert = chain[0]
        except IndexError:
            raise error.IncompleteChainError(
                    "Chain must contain leaf certificate.")

        scts_blob = leaf_cert.embedded_sct_list()
        if scts_blob is None:
            return []

        scts = client_pb2.SignedCertificateTimestampList()
        tls_message.decode(scts_blob, scts)

        result = []
        for sct_blob in scts.sct_list:
            sct = client_pb2.SignedCertificateTimestamp()
            tls_message.decode(sct_blob, sct)

            try:
                self.verify_sct(sct, chain)
                result.append((sct, True))
            except error.VerifyError:
                result.append((sct, False))

        return result


def create_key_info_from_raw_key(log_key):
    """Creates a KeyInfo from the given raw (DER-encoded) key.

    Detects the key type (ECDSA or RSA), returning a client_pb2.KeyInfo
    instance that can be used to construct a LogVerifier.

    Args:
        log_key: A DER-encoded key.

    Returns:
        A client_pb2.KeyInfo instance with all fields correctly filled.
    """
    key_info = client_pb2.KeyInfo()
    decoded_key = x509_common.SubjectPublicKeyInfo.decode(log_key)
    key_algorithm_oid = decoded_key['algorithm']['algorithm']
    if key_algorithm_oid == oid.RSA_ENCRYPTION:
        key_info.type = client_pb2.KeyInfo.RSA
    elif key_algorithm_oid == oid.ID_EC_PUBLICKEY:
        key_info.type = client_pb2.KeyInfo.ECDSA
    else:
        raise error.UnsupportedAlgorithmError(
                'Unknown key type: %s' % key_algorithm_oid)
    key_info.pem_key = pem.to_pem(log_key, 'PUBLIC KEY')
    return key_info
