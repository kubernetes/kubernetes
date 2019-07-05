// +build go1.4

package jwt

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
)

// Implements the RSAPSS family of signing methods signing methods
type SigningMethodRSAPSS struct {
	*SigningMethodRSA
	Options *rsa.PSSOptions
}

// Specific instances for RS/PS and company
var (
	SigningMethodPS256 *SigningMethodRSAPSS
	SigningMethodPS384 *SigningMethodRSAPSS
	SigningMethodPS512 *SigningMethodRSAPSS
)

func init() {
	// PS256
	SigningMethodPS256 = &SigningMethodRSAPSS{
		&SigningMethodRSA{
			Name: "PS256",
			Hash: crypto.SHA256,
		},
		&rsa.PSSOptions{
			SaltLength: rsa.PSSSaltLengthAuto,
			Hash:       crypto.SHA256,
		},
	}
	RegisterSigningMethod(SigningMethodPS256.Alg(), func() SigningMethod {
		return SigningMethodPS256
	})

	// PS384
	SigningMethodPS384 = &SigningMethodRSAPSS{
		&SigningMethodRSA{
			Name: "PS384",
			Hash: crypto.SHA384,
		},
		&rsa.PSSOptions{
			SaltLength: rsa.PSSSaltLengthAuto,
			Hash:       crypto.SHA384,
		},
	}
	RegisterSigningMethod(SigningMethodPS384.Alg(), func() SigningMethod {
		return SigningMethodPS384
	})

	// PS512
	SigningMethodPS512 = &SigningMethodRSAPSS{
		&SigningMethodRSA{
			Name: "PS512",
			Hash: crypto.SHA512,
		},
		&rsa.PSSOptions{
			SaltLength: rsa.PSSSaltLengthAuto,
			Hash:       crypto.SHA512,
		},
	}
	RegisterSigningMethod(SigningMethodPS512.Alg(), func() SigningMethod {
		return SigningMethodPS512
	})
}

// Implements the Verify method from SigningMethod
// For this verify method, key must be an rsa.PublicKey struct
func (m *SigningMethodRSAPSS) Verify(signingString, signature string, key interface{}) error {
	var err error

	// Decode the signature
	var sig []byte
	if sig, err = DecodeSegment(signature); err != nil {
		return err
	}

	var rsaKey *rsa.PublicKey
	switch k := key.(type) {
	case *rsa.PublicKey:
		rsaKey = k
	default:
		return ErrInvalidKey
	}

	// Create hasher
	if !m.Hash.Available() {
		return ErrHashUnavailable
	}
	hasher := m.Hash.New()
	hasher.Write([]byte(signingString))

	return rsa.VerifyPSS(rsaKey, m.Hash, hasher.Sum(nil), sig, m.Options)
}

// Implements the Sign method from SigningMethod
// For this signing method, key must be an rsa.PrivateKey struct
func (m *SigningMethodRSAPSS) Sign(signingString string, key interface{}) (string, error) {
	var rsaKey *rsa.PrivateKey

	switch k := key.(type) {
	case *rsa.PrivateKey:
		rsaKey = k
	default:
		return "", ErrInvalidKeyType
	}

	// Create the hasher
	if !m.Hash.Available() {
		return "", ErrHashUnavailable
	}

	hasher := m.Hash.New()
	hasher.Write([]byte(signingString))

	// Sign the string and return the encoded bytes
	if sigBytes, err := rsa.SignPSS(rand.Reader, rsaKey, m.Hash, hasher.Sum(nil), m.Options); err == nil {
		return EncodeSegment(sigBytes), nil
	} else {
		return "", err
	}
}
