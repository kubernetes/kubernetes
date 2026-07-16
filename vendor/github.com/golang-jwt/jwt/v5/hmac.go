package jwt

import (
	"crypto"
	"crypto/hmac"
	"errors"
)

// SigningMethodHMAC implements the HMAC-SHA family of signing methods.
// Expects key type of []byte for both signing and validation
type SigningMethodHMAC struct {
	Name string
	Hash crypto.Hash
}

// Specific instances for HS256 and company
var (
	SigningMethodHS256  *SigningMethodHMAC
	SigningMethodHS384  *SigningMethodHMAC
	SigningMethodHS512  *SigningMethodHMAC
	ErrSignatureInvalid = errors.New("signature is invalid")
)

func init() {
	// HS256
	SigningMethodHS256 = &SigningMethodHMAC{"HS256", crypto.SHA256}
	RegisterSigningMethod(SigningMethodHS256.Alg(), func() SigningMethod {
		return SigningMethodHS256
	})

	// HS384
	SigningMethodHS384 = &SigningMethodHMAC{"HS384", crypto.SHA384}
	RegisterSigningMethod(SigningMethodHS384.Alg(), func() SigningMethod {
		return SigningMethodHS384
	})

	// HS512
	SigningMethodHS512 = &SigningMethodHMAC{"HS512", crypto.SHA512}
	RegisterSigningMethod(SigningMethodHS512.Alg(), func() SigningMethod {
		return SigningMethodHS512
	})
}

func (m *SigningMethodHMAC) Alg() string {
	return m.Name
}

// Verify implements token verification for the SigningMethod. Returns nil if
// the signature is valid. Key must be []byte.
//
// Note it is not advised to provide a []byte which was converted from a 'human
// readable' string using a subset of ASCII characters. To maximize entropy, you
// should ideally be providing a []byte key which was produced from a
// cryptographically random source, e.g. crypto/rand. Additional information
// about this, and why we intentionally are not supporting string as a key can
// be found on our usage guide
// https://golang-jwt.github.io/jwt/usage/signing_methods/#signing-methods-and-key-types.
func (m *SigningMethodHMAC) Verify(signingString string, sig []byte, key any) error {
	// Verify the key is the right type
	keyBytes, ok := key.([]byte)
	if !ok {
		return newError("HMAC verify expects []byte", ErrInvalidKeyType)
	}

	// Can we use the specified hashing method?
	if !m.Hash.Available() {
		return ErrHashUnavailable
	}

	// This signing method is symmetric, so we validate the signature
	// by reproducing the signature from the signing string and key, then
	// comparing that against the provided signature.
	hasher := hmac.New(m.Hash.New, keyBytes)
	hasher.Write([]byte(signingString))
	if !hmac.Equal(sig, hasher.Sum(nil)) {
		return ErrSignatureInvalid
	}

	// No validation errors.  Signature is good.
	return nil
}

// Sign implements token signing for the SigningMethod. Key must be []byte.
//
// Note it is not advised to provide a []byte which was converted from a 'human
// readable' string using a subset of ASCII characters. To maximize entropy, you
// should ideally be providing a []byte key which was produced from a
// cryptographically random source, e.g. crypto/rand. Additional information
// about this, and why we intentionally are not supporting string as a key can
// be found on our usage guide https://golang-jwt.github.io/jwt/usage/signing_methods/.
func (m *SigningMethodHMAC) Sign(signingString string, key any) ([]byte, error) {
	if keyBytes, ok := key.([]byte); ok {
		if !m.Hash.Available() {
			return nil, ErrHashUnavailable
		}

		hasher := hmac.New(m.Hash.New, keyBytes)
		hasher.Write([]byte(signingString))

		return hasher.Sum(nil), nil
	}

	return nil, newError("HMAC sign expects []byte", ErrInvalidKeyType)
}
