package jose

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"fmt"
	"strings"
)

type VerifierRSA struct {
	KeyID     string
	Hash      crypto.Hash
	PublicKey rsa.PublicKey
}

type SignerRSA struct {
	PrivateKey rsa.PrivateKey
	VerifierRSA
}

func NewVerifierRSA(jwk JWK) (*VerifierRSA, error) {
	if strings.ToUpper(jwk.Alg) != "RS256" {
		return nil, fmt.Errorf("unsupported key algorithm %q", jwk.Alg)
	}

	v := VerifierRSA{
		KeyID: jwk.ID,
		PublicKey: rsa.PublicKey{
			N: jwk.Modulus,
			E: jwk.Exponent,
		},
		Hash: crypto.SHA256,
	}

	return &v, nil
}

func NewSignerRSA(kid string, key rsa.PrivateKey) *SignerRSA {
	return &SignerRSA{
		PrivateKey: key,
		VerifierRSA: VerifierRSA{
			KeyID:     kid,
			PublicKey: key.PublicKey,
			Hash:      crypto.SHA256,
		},
	}
}

func (v *VerifierRSA) ID() string {
	return v.KeyID
}

func (v *VerifierRSA) Alg() string {
	return "RS256"
}

func (v *VerifierRSA) Verify(sig []byte, data []byte) error {
	h := v.Hash.New()
	h.Write(data)
	return rsa.VerifyPKCS1v15(&v.PublicKey, v.Hash, h.Sum(nil), sig)
}

func (s *SignerRSA) Sign(data []byte) ([]byte, error) {
	h := s.Hash.New()
	h.Write(data)
	return rsa.SignPKCS1v15(rand.Reader, &s.PrivateKey, s.Hash, h.Sum(nil))
}
