package key

import (
	"crypto/rsa"
	"math/big"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
)

func TestPrivateRSAKeyJWK(t *testing.T) {
	n := big.NewInt(int64(17))
	if n == nil {
		panic("NewInt returned nil")
	}

	k := &PrivateKey{
		KeyID: "foo",
		PrivateKey: &rsa.PrivateKey{
			PublicKey: rsa.PublicKey{N: n, E: 65537},
		},
	}

	want := jose.JWK{
		ID:       "foo",
		Type:     "RSA",
		Alg:      "RS256",
		Use:      "sig",
		Modulus:  n,
		Exponent: 65537,
	}

	got := k.JWK()
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("JWK mismatch: want=%#v got=%#v", want, got)
	}
}

func TestPublicKeySetKey(t *testing.T) {
	n := big.NewInt(int64(17))
	if n == nil {
		panic("NewInt returned nil")
	}

	k := jose.JWK{
		ID:       "foo",
		Type:     "RSA",
		Alg:      "RS256",
		Use:      "sig",
		Modulus:  n,
		Exponent: 65537,
	}
	now := time.Now().UTC()
	ks := NewPublicKeySet([]jose.JWK{k}, now)

	want := &PublicKey{jwk: k}
	got := ks.Key("foo")
	if !reflect.DeepEqual(want, got) {
		t.Errorf("Unexpected response from PublicKeySet.Key: want=%#v got=%#v", want, got)
	}

	got = ks.Key("bar")
	if got != nil {
		t.Errorf("Expected nil response from PublicKeySet.Key, got %#v", got)
	}
}
