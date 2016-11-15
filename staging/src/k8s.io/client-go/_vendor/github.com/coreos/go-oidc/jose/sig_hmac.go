package jose

import (
	"bytes"
	"crypto"
	"crypto/hmac"
	_ "crypto/sha256"
	"errors"
	"fmt"
)

type VerifierHMAC struct {
	KeyID  string
	Hash   crypto.Hash
	Secret []byte
}

type SignerHMAC struct {
	VerifierHMAC
}

func NewVerifierHMAC(jwk JWK) (*VerifierHMAC, error) {
	if jwk.Alg != "" && jwk.Alg != "HS256" {
		return nil, fmt.Errorf("unsupported key algorithm %q", jwk.Alg)
	}

	v := VerifierHMAC{
		KeyID:  jwk.ID,
		Secret: jwk.Secret,
		Hash:   crypto.SHA256,
	}

	return &v, nil
}

func (v *VerifierHMAC) ID() string {
	return v.KeyID
}

func (v *VerifierHMAC) Alg() string {
	return "HS256"
}

func (v *VerifierHMAC) Verify(sig []byte, data []byte) error {
	h := hmac.New(v.Hash.New, v.Secret)
	h.Write(data)
	if !bytes.Equal(sig, h.Sum(nil)) {
		return errors.New("invalid hmac signature")
	}
	return nil
}

func NewSignerHMAC(kid string, secret []byte) *SignerHMAC {
	return &SignerHMAC{
		VerifierHMAC: VerifierHMAC{
			KeyID:  kid,
			Secret: secret,
			Hash:   crypto.SHA256,
		},
	}
}

func (s *SignerHMAC) Sign(data []byte) ([]byte, error) {
	h := hmac.New(s.Hash.New, s.Secret)
	h.Write(data)
	return h.Sum(nil), nil
}
