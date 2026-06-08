package jwt

import (
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"errors"
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
func (m *SigningMethodEd25519) Verify(signingString string, sig []byte, key any) error {
	var ed25519Key ed25519.PublicKey
	var ok bool

	if ed25519Key, ok = key.(ed25519.PublicKey); !ok {
		return newError("Ed25519 verify expects ed25519.PublicKey", ErrInvalidKeyType)
	}

	if len(ed25519Key) != ed25519.PublicKeySize {
		return ErrInvalidKey
	}

	// Verify the signature
	if !ed25519.Verify(ed25519Key, []byte(signingString), sig) {
		return ErrEd25519Verification
	}

	return nil
}

// Sign implements token signing for the SigningMethod.
// For this signing method, key must be an ed25519.PrivateKey
func (m *SigningMethodEd25519) Sign(signingString string, key any) ([]byte, error) {
	var ed25519Key crypto.Signer
	var ok bool

	if ed25519Key, ok = key.(crypto.Signer); !ok {
		return nil, newError("Ed25519 sign expects crypto.Signer", ErrInvalidKeyType)
	}

	if _, ok := ed25519Key.Public().(ed25519.PublicKey); !ok {
		return nil, ErrInvalidKey
	}

	// Sign the string and return the result. ed25519 performs a two-pass hash
	// as part of its algorithm. Therefore, we need to pass a non-prehashed
	// message into the Sign function, as indicated by crypto.Hash(0)
	sig, err := ed25519Key.Sign(rand.Reader, []byte(signingString), crypto.Hash(0))
	if err != nil {
		return nil, err
	}

	return sig, nil
}
