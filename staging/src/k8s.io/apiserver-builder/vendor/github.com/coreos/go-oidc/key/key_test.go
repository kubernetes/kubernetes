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

func TestPublicKeyMarshalJSON(t *testing.T) {
	k := jose.JWK{
		ID:       "foo",
		Type:     "RSA",
		Alg:      "RS256",
		Use:      "sig",
		Modulus:  big.NewInt(int64(17)),
		Exponent: 65537,
	}
	want := `{"kid":"foo","kty":"RSA","alg":"RS256","use":"sig","e":"AQAB","n":"EQ"}`
	pubKey := NewPublicKey(k)
	gotBytes, err := pubKey.MarshalJSON()
	if err != nil {
		t.Fatalf("failed to marshal public key: %v", err)
	}
	got := string(gotBytes)
	if got != want {
		t.Errorf("got != want:\n%s\n%s", got, want)
	}
}

func TestGeneratePrivateKeyIDs(t *testing.T) {
	key1, err := GeneratePrivateKey()
	if err != nil {
		t.Fatalf("GeneratePrivateKey(): %v", err)
	}
	key2, err := GeneratePrivateKey()
	if err != nil {
		t.Fatalf("GeneratePrivateKey(): %v", err)
	}
	if key1.KeyID == key2.KeyID {
		t.Fatalf("expected different keys to have different key IDs")
	}
}
