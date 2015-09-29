package key

import (
	"crypto/rand"
	"crypto/rsa"
	"encoding/base64"
	"math/big"
	"time"

	"github.com/coreos/go-oidc/jose"
)

func NewPublicKey(jwk jose.JWK) *PublicKey {
	return &PublicKey{jwk: jwk}
}

type PublicKey struct {
	jwk jose.JWK
}

func (k *PublicKey) ID() string {
	return k.jwk.ID
}

func (k *PublicKey) Verifier() (jose.Verifier, error) {
	return jose.NewVerifierRSA(k.jwk)
}

type PrivateKey struct {
	KeyID      string
	PrivateKey *rsa.PrivateKey
}

func (k *PrivateKey) ID() string {
	return k.KeyID
}

func (k *PrivateKey) Signer() jose.Signer {
	return jose.NewSignerRSA(k.ID(), *k.PrivateKey)
}

func (k *PrivateKey) JWK() jose.JWK {
	return jose.JWK{
		ID:       k.KeyID,
		Type:     "RSA",
		Alg:      "RS256",
		Use:      "sig",
		Exponent: k.PrivateKey.PublicKey.E,
		Modulus:  k.PrivateKey.PublicKey.N,
	}
}

type KeySet interface {
	ExpiresAt() time.Time
}

type PublicKeySet struct {
	keys      []PublicKey
	index     map[string]*PublicKey
	expiresAt time.Time
}

func NewPublicKeySet(jwks []jose.JWK, exp time.Time) *PublicKeySet {
	keys := make([]PublicKey, len(jwks))
	index := make(map[string]*PublicKey)
	for i, jwk := range jwks {
		keys[i] = *NewPublicKey(jwk)
		index[keys[i].ID()] = &keys[i]
	}
	return &PublicKeySet{
		keys:      keys,
		index:     index,
		expiresAt: exp,
	}
}

func (s *PublicKeySet) ExpiresAt() time.Time {
	return s.expiresAt
}

func (s *PublicKeySet) Keys() []PublicKey {
	return s.keys
}

func (s *PublicKeySet) Key(id string) *PublicKey {
	return s.index[id]
}

type PrivateKeySet struct {
	keys        []*PrivateKey
	ActiveKeyID string
	expiresAt   time.Time
}

func NewPrivateKeySet(keys []*PrivateKey, exp time.Time) *PrivateKeySet {
	return &PrivateKeySet{
		keys:        keys,
		ActiveKeyID: keys[0].ID(),
		expiresAt:   exp.UTC(),
	}
}

func (s *PrivateKeySet) Keys() []*PrivateKey {
	return s.keys
}

func (s *PrivateKeySet) ExpiresAt() time.Time {
	return s.expiresAt
}

func (s *PrivateKeySet) Active() *PrivateKey {
	for i, k := range s.keys {
		if k.ID() == s.ActiveKeyID {
			return s.keys[i]
		}
	}

	return nil
}

type GeneratePrivateKeyFunc func() (*PrivateKey, error)

func GeneratePrivateKey() (*PrivateKey, error) {
	pk, err := rsa.GenerateKey(rand.Reader, 1024)
	if err != nil {
		return nil, err
	}

	k := PrivateKey{
		KeyID:      base64BigInt(pk.PublicKey.N),
		PrivateKey: pk,
	}

	return &k, nil
}

func base64BigInt(b *big.Int) string {
	return base64.URLEncoding.EncodeToString(b.Bytes())
}
