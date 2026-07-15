package crypto

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"fmt"
)

// KeyAlgorithm identifies the key generation algorithm.
type KeyAlgorithm string

const (
	// RSAKeyAlgorithm specifies RSA key generation.
	RSAKeyAlgorithm KeyAlgorithm = "RSA"
	// ECDSAKeyAlgorithm specifies ECDSA key generation.
	ECDSAKeyAlgorithm KeyAlgorithm = "ECDSA"
)

// ECDSACurve identifies a named ECDSA curve.
type ECDSACurve string

const (
	// P256 specifies the NIST P-256 curve (secp256r1), providing 128-bit security.
	P256 ECDSACurve = "P256"
	// P384 specifies the NIST P-384 curve (secp384r1), providing 192-bit security.
	P384 ECDSACurve = "P384"
	// P521 specifies the NIST P-521 curve (secp521r1), providing 256-bit security.
	P521 ECDSACurve = "P521"
)

// RSAKeyPairGenerator generates RSA key pairs.
type RSAKeyPairGenerator struct {
	// Bits is the RSA key size in bits. Must be >= 2048.
	Bits int
}

func (g RSAKeyPairGenerator) GenerateKeyPair() (crypto.PublicKey, crypto.PrivateKey, error) {
	bits := g.Bits
	if bits == 0 {
		bits = keyBits
	}
	privateKey, err := rsa.GenerateKey(rand.Reader, bits)
	if err != nil {
		return nil, nil, err
	}
	return &privateKey.PublicKey, privateKey, nil
}

// ECDSAKeyPairGenerator generates ECDSA key pairs.
type ECDSAKeyPairGenerator struct {
	// Curve is the named ECDSA curve.
	Curve ECDSACurve
}

func (g ECDSAKeyPairGenerator) GenerateKeyPair() (crypto.PublicKey, crypto.PrivateKey, error) {
	curve, err := g.ellipticCurve()
	if err != nil {
		return nil, nil, err
	}
	privateKey, err := ecdsa.GenerateKey(curve, rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	return &privateKey.PublicKey, privateKey, nil
}

func (g ECDSAKeyPairGenerator) ellipticCurve() (elliptic.Curve, error) {
	switch g.Curve {
	case P256:
		return elliptic.P256(), nil
	case P384:
		return elliptic.P384(), nil
	case P521:
		return elliptic.P521(), nil
	default:
		return nil, fmt.Errorf("unsupported ECDSA curve: %q", g.Curve)
	}
}

// KeyUsageForPublicKey returns the x509.KeyUsage flags appropriate for the
// given public key type. ECDSA keys use DigitalSignature only; RSA keys also
// include KeyEncipherment.
func KeyUsageForPublicKey(pub crypto.PublicKey) x509.KeyUsage {
	switch pub.(type) {
	case *ecdsa.PublicKey:
		return x509.KeyUsageDigitalSignature
	default:
		return x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature
	}
}

// SubjectKeyIDFromPublicKey computes a truncated SHA-256 hash suitable for
// use as a certificate SubjectKeyId from any supported public key type.
// This uses the first 160 bits of the SHA-256 hash per RFC 7093, consistent
// with the Go standard library since Go 1.25 (go.dev/issue/71746) and
// Let's Encrypt. Prior Go versions used SHA-1 which is not FIPS-compatible.
func SubjectKeyIDFromPublicKey(pub crypto.PublicKey) ([]byte, error) {
	var rawBytes []byte
	switch pub := pub.(type) {
	case *rsa.PublicKey:
		rawBytes = pub.N.Bytes()
	case *ecdsa.PublicKey:
		ecdhKey, err := pub.ECDH()
		if err != nil {
			return nil, fmt.Errorf("failed to convert ECDSA public key: %w", err)
		}
		rawBytes = ecdhKey.Bytes()
	default:
		return nil, fmt.Errorf("unsupported public key type: %T", pub)
	}
	hash := sha256.Sum256(rawBytes)
	return hash[:20], nil
}
