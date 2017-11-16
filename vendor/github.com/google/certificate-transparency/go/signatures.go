package ct

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/asn1"
	"encoding/pem"
	"errors"
	"flag"
	"fmt"
	"log"
	"math/big"
)

var allowVerificationWithNonCompliantKeys = flag.Bool("allow_verification_with_non_compliant_keys", false,
	"Allow a SignatureVerifier to use keys which are technically non-compliant with RFC6962.")

// PublicKeyFromPEM parses a PEM formatted block and returns the public key contained within and any remaining unread bytes, or an error.
func PublicKeyFromPEM(b []byte) (crypto.PublicKey, SHA256Hash, []byte, error) {
	p, rest := pem.Decode(b)
	if p == nil {
		return nil, [sha256.Size]byte{}, rest, fmt.Errorf("no PEM block found in %s", string(b))
	}
	k, err := x509.ParsePKIXPublicKey(p.Bytes)
	return k, sha256.Sum256(p.Bytes), rest, err
}

// SignatureVerifier can verify signatures on SCTs and STHs
type SignatureVerifier struct {
	pubKey crypto.PublicKey
}

// NewSignatureVerifier creates a new SignatureVerifier using the passed in PublicKey.
func NewSignatureVerifier(pk crypto.PublicKey) (*SignatureVerifier, error) {
	switch pkType := pk.(type) {
	case *rsa.PublicKey:
		if pkType.N.BitLen() < 2048 {
			e := fmt.Errorf("public key is RSA with < 2048 bits (size:%d)", pkType.N.BitLen())
			if !(*allowVerificationWithNonCompliantKeys) {
				return nil, e
			}
			log.Printf("WARNING: %v", e)
		}
	case *ecdsa.PublicKey:
		params := *(pkType.Params())
		if params != *elliptic.P256().Params() {
			e := fmt.Errorf("public is ECDSA, but not on the P256 curve")
			if !(*allowVerificationWithNonCompliantKeys) {
				return nil, e
			}
			log.Printf("WARNING: %v", e)

		}
	default:
		return nil, fmt.Errorf("Unsupported public key type %v", pkType)
	}

	return &SignatureVerifier{
		pubKey: pk,
	}, nil
}

// verifySignature verifies that the passed in signature over data was created by our PublicKey.
// Currently, only SHA256 is supported as a HashAlgorithm, and only ECDSA and RSA signatures are supported.
func (s SignatureVerifier) verifySignature(data []byte, sig DigitallySigned) error {
	if sig.HashAlgorithm != SHA256 {
		return fmt.Errorf("unsupported HashAlgorithm in signature: %v", sig.HashAlgorithm)
	}

	hasherType := crypto.SHA256
	hasher := hasherType.New()
	if _, err := hasher.Write(data); err != nil {
		return fmt.Errorf("failed to write to hasher: %v", err)
	}
	hash := hasher.Sum([]byte{})

	switch sig.SignatureAlgorithm {
	case RSA:
		rsaKey, ok := s.pubKey.(*rsa.PublicKey)
		if !ok {
			return fmt.Errorf("cannot verify RSA signature with %T key", s.pubKey)
		}
		if err := rsa.VerifyPKCS1v15(rsaKey, hasherType, hash, sig.Signature); err != nil {
			return fmt.Errorf("failed to verify rsa signature: %v", err)
		}
	case ECDSA:
		ecdsaKey, ok := s.pubKey.(*ecdsa.PublicKey)
		if !ok {
			return fmt.Errorf("cannot verify ECDSA signature with %T key", s.pubKey)
		}
		var ecdsaSig struct {
			R, S *big.Int
		}
		rest, err := asn1.Unmarshal(sig.Signature, &ecdsaSig)
		if err != nil {
			return fmt.Errorf("failed to unmarshal ECDSA signature: %v", err)
		}
		if len(rest) != 0 {
			log.Printf("Garbage following signature %v", rest)
		}

		if !ecdsa.Verify(ecdsaKey, hash, ecdsaSig.R, ecdsaSig.S) {
			return errors.New("failed to verify ecdsa signature")
		}
	default:
		return fmt.Errorf("unsupported signature type %v", sig.SignatureAlgorithm)
	}
	return nil
}

// VerifySCTSignature verifies that the SCT's signature is valid for the given LogEntry
func (s SignatureVerifier) VerifySCTSignature(sct SignedCertificateTimestamp, entry LogEntry) error {
	sctData, err := SerializeSCTSignatureInput(sct, entry)
	if err != nil {
		return err
	}
	return s.verifySignature(sctData, sct.Signature)
}

// VerifySTHSignature verifies that the STH's signature is valid.
func (s SignatureVerifier) VerifySTHSignature(sth SignedTreeHead) error {
	sthData, err := SerializeSTHSignatureInput(sth)
	if err != nil {
		return err
	}
	return s.verifySignature(sthData, sth.TreeHeadSignature)
}
