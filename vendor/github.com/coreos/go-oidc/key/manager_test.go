package key

import (
	"crypto/rsa"
	"math/big"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/jonboulle/clockwork"

	"github.com/coreos/go-oidc/jose"
)

var (
	jwk1 jose.JWK
	jwk2 jose.JWK
	jwk3 jose.JWK
)

func init() {
	jwk1 = jose.JWK{
		ID:       "1",
		Type:     "RSA",
		Alg:      "RS256",
		Use:      "sig",
		Modulus:  big.NewInt(1),
		Exponent: 65537,
	}

	jwk2 = jose.JWK{
		ID:       "2",
		Type:     "RSA",
		Alg:      "RS256",
		Use:      "sig",
		Modulus:  big.NewInt(2),
		Exponent: 65537,
	}

	jwk3 = jose.JWK{
		ID:       "3",
		Type:     "RSA",
		Alg:      "RS256",
		Use:      "sig",
		Modulus:  big.NewInt(3),
		Exponent: 65537,
	}
}

func generatePrivateKeyStatic(t *testing.T, idAndN int) *PrivateKey {
	n := big.NewInt(int64(idAndN))
	if n == nil {
		t.Fatalf("Call to NewInt(%d) failed", idAndN)
	}

	pk := &rsa.PrivateKey{
		PublicKey: rsa.PublicKey{N: n, E: 65537},
	}

	return &PrivateKey{
		KeyID:      strconv.Itoa(idAndN),
		PrivateKey: pk,
	}
}

func TestPrivateKeyManagerJWKsRotate(t *testing.T) {
	k1 := generatePrivateKeyStatic(t, 1)
	k2 := generatePrivateKeyStatic(t, 2)
	k3 := generatePrivateKeyStatic(t, 3)
	km := NewPrivateKeyManager()
	err := km.Set(&PrivateKeySet{
		keys:        []*PrivateKey{k1, k2, k3},
		ActiveKeyID: k1.KeyID,
		expiresAt:   time.Now().Add(time.Minute),
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	want := []jose.JWK{jwk1, jwk2, jwk3}
	got, err := km.JWKs()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("JWK mismatch: want=%#v got=%#v", want, got)
	}
}

func TestPrivateKeyManagerSigner(t *testing.T) {
	k := generatePrivateKeyStatic(t, 13)

	km := NewPrivateKeyManager()
	err := km.Set(&PrivateKeySet{
		keys:        []*PrivateKey{k},
		ActiveKeyID: k.KeyID,
		expiresAt:   time.Now().Add(time.Minute),
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	signer, err := km.Signer()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	wantID := "13"
	gotID := signer.ID()
	if wantID != gotID {
		t.Fatalf("Signer has incorrect ID: want=%s got=%s", wantID, gotID)
	}
}

func TestPrivateKeyManagerHealthyFail(t *testing.T) {
	keyFixture := generatePrivateKeyStatic(t, 1)
	tests := []*privateKeyManager{
		// keySet nil
		&privateKeyManager{
			keySet: nil,
			clock:  clockwork.NewRealClock(),
		},
		// zero keys
		&privateKeyManager{
			keySet: &PrivateKeySet{
				keys:      []*PrivateKey{},
				expiresAt: time.Now().Add(time.Minute),
			},
			clock: clockwork.NewRealClock(),
		},
		// key set expired
		&privateKeyManager{
			keySet: &PrivateKeySet{
				keys:      []*PrivateKey{keyFixture},
				expiresAt: time.Now().Add(-1 * time.Minute),
			},
			clock: clockwork.NewRealClock(),
		},
	}

	for i, tt := range tests {
		if err := tt.Healthy(); err == nil {
			t.Errorf("case %d: nil error", i)
		}
	}
}

func TestPrivateKeyManagerHealthyFailsOtherMethods(t *testing.T) {
	km := NewPrivateKeyManager()
	if _, err := km.JWKs(); err == nil {
		t.Fatalf("Expected non-nil error")
	}
	if _, err := km.Signer(); err == nil {
		t.Fatalf("Expected non-nil error")
	}
}

func TestPrivateKeyManagerExpiresAt(t *testing.T) {
	fc := clockwork.NewFakeClock()
	now := fc.Now().UTC()

	k := generatePrivateKeyStatic(t, 17)
	km := &privateKeyManager{
		clock: fc,
	}

	want := fc.Now().UTC()
	got := km.ExpiresAt()
	if want != got {
		t.Fatalf("Incorrect expiration time: want=%v got=%v", want, got)
	}

	err := km.Set(&PrivateKeySet{
		keys:        []*PrivateKey{k},
		ActiveKeyID: k.KeyID,
		expiresAt:   now.Add(2 * time.Minute),
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	want = fc.Now().UTC().Add(2 * time.Minute)
	got = km.ExpiresAt()
	if want != got {
		t.Fatalf("Incorrect expiration time: want=%v got=%v", want, got)
	}
}

func TestPublicKeys(t *testing.T) {
	km := NewPrivateKeyManager()
	k1 := generatePrivateKeyStatic(t, 1)
	k2 := generatePrivateKeyStatic(t, 2)
	k3 := generatePrivateKeyStatic(t, 3)

	tests := [][]*PrivateKey{
		[]*PrivateKey{k1},
		[]*PrivateKey{k1, k2},
		[]*PrivateKey{k1, k2, k3},
	}

	for i, tt := range tests {
		ks := &PrivateKeySet{
			keys:      tt,
			expiresAt: time.Now().Add(time.Hour),
		}
		km.Set(ks)

		jwks, err := km.JWKs()
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		pks := NewPublicKeySet(jwks, time.Now().Add(time.Hour))
		want := pks.Keys()
		got, err := km.PublicKeys()
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if !reflect.DeepEqual(want, got) {
			t.Errorf("case %d: Invalid public keys: want=%v got=%v", i, want, got)
		}
	}
}
