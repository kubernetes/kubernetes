package jwt

import (
	"errors"

	"crypto"
	"crypto/ed25519"
	"crypto/rand"
)

var (
	ErrEd25519Verification = errors.New("ed25519: verification error")
)

// SigningMethodEd25519 implements the EdDSA family.
// Expects ed25519.PrivateKey for signing and ed25519.PublicKey for verification
type SigningMethodEd25519 struct{}

// Specific instance for EdDSA
var (
	SigningMethodEdDSA *SigningMethodEd25519
)

func init() {
	SigningMethodEdDSA = &SigningMethodEd25519{}
	RegisterSigningMethod(SigningMethodEdDSA.Alg(), func() SigningMethod {
		return SigningMethodEdDSA
	})
}

func (m *SigningMethodEd25519) Alg() string {
	return "EdDSA"
}

// Verify implements token verification for the SigningMethod.
// For this verify method, key must be an ed25519.PublicKey
func (m *SigningMethodEd25519) Verify(signingString, signature string, key interface{}) error {
	var err error
	var ed25519Key ed25519.PublicKey
	var ok bool

	if ed25519Key, ok = key.(ed25519.PublicKey); !ok {
		return ErrInvalidKeyType
	}

	if len(ed25519Key) != ed25519.PublicKeySize {
		return ErrInvalidKey
	}

	// Decode the signature
	var sig []byte
	if sig, err = DecodeSegment(signature); err != nil {
		return err
	}

	// Verify the signature
	if !ed25519.Verify(ed25519Key, []byte(signingString), sig) {
		return ErrEd25519Verification
	}

	return nil
}

// Sign implements token signing for the SigningMethod.
// For this signing method, key must be an ed25519.PrivateKey
func (m *SigningMethodEd25519) Sign(signingString string, key interface{}) (string, error) {
	var ed25519Key crypto.Signer
	var ok bool

	if ed25519Key, ok = key.(crypto.Signer); !ok {
		return "", ErrInvalidKeyType
	}

	if _, ok := ed25519Key.Public().(ed25519.PublicKey); !ok {
		return "", ErrInvalidKey
	}

	// Sign the string and return the encoded result
	// ed25519 performs a two-pass hash as part of its algorithm. Therefore, we need to pass a non-prehashed message into the Sign function, as indicated by crypto.Hash(0)
	sig, err := ed25519Key.Sign(rand.Reader, []byte(signingString), crypto.Hash(0))
	if err != nil {
		return "", err
	}
	return EncodeSegment(sig), nil
}
