package oidc

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	jose "gopkg.in/square/go-jose.v2"
)

type keyServer struct {
	keys       jose.JSONWebKeySet
	setHeaders func(h http.Header)
}

func (k *keyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if k.setHeaders != nil {
		k.setHeaders(w.Header())
	}
	if err := json.NewEncoder(w).Encode(k.keys); err != nil {
		panic(err)
	}
}

type signingKey struct {
	keyID string // optional
	priv  interface{}
	pub   interface{}
	alg   jose.SignatureAlgorithm
}

// sign creates a JWS using the private key from the provided payload.
func (s *signingKey) sign(t *testing.T, payload []byte) string {
	privKey := &jose.JSONWebKey{Key: s.priv, Algorithm: string(s.alg), KeyID: s.keyID}

	signer, err := jose.NewSigner(jose.SigningKey{Algorithm: s.alg, Key: privKey}, nil)
	if err != nil {
		t.Fatal(err)
	}
	jws, err := signer.Sign(payload)
	if err != nil {
		t.Fatal(err)
	}

	data, err := jws.CompactSerialize()
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// jwk returns the public part of the signing key.
func (s *signingKey) jwk() jose.JSONWebKey {
	return jose.JSONWebKey{Key: s.pub, Use: "sig", Algorithm: string(s.alg), KeyID: s.keyID}
}

func newRSAKey(t *testing.T) *signingKey {
	priv, err := rsa.GenerateKey(rand.Reader, 1028)
	if err != nil {
		t.Fatal(err)
	}
	return &signingKey{"", priv, priv.Public(), jose.RS256}
}

func newECDSAKey(t *testing.T) *signingKey {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	return &signingKey{"", priv, priv.Public(), jose.ES256}
}

func TestRSAVerify(t *testing.T) {
	good := newRSAKey(t)
	bad := newRSAKey(t)

	testKeyVerify(t, good, bad, good)
}

func TestECDSAVerify(t *testing.T) {
	good := newECDSAKey(t)
	bad := newECDSAKey(t)
	testKeyVerify(t, good, bad, good)
}

func TestMultipleKeysVerify(t *testing.T) {
	key1 := newRSAKey(t)
	key2 := newRSAKey(t)
	bad := newECDSAKey(t)

	key1.keyID = "key1"
	key2.keyID = "key2"
	bad.keyID = "key3"

	testKeyVerify(t, key2, bad, key1, key2)
}

func TestMismatchedKeyID(t *testing.T) {
	key1 := newRSAKey(t)
	key2 := newRSAKey(t)

	// shallow copy
	bad := new(signingKey)
	*bad = *key1

	// The bad key is a valid key this time, but has a different Key ID.
	// It shouldn't match key1 because of the mismatched ID, even though
	// it would confirm the signature just fine.
	bad.keyID = "key3"

	key1.keyID = "key1"
	key2.keyID = "key2"

	testKeyVerify(t, key2, bad, key1, key2)
}

func testKeyVerify(t *testing.T, good, bad *signingKey, verification ...*signingKey) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	keySet := jose.JSONWebKeySet{}
	for _, v := range verification {
		keySet.Keys = append(keySet.Keys, v.jwk())
	}

	payload := []byte("a secret")

	jws, err := jose.ParseSigned(good.sign(t, payload))
	if err != nil {
		t.Fatal(err)
	}
	badJWS, err := jose.ParseSigned(bad.sign(t, payload))
	if err != nil {
		t.Fatal(err)
	}

	s := httptest.NewServer(&keyServer{keys: keySet})
	defer s.Close()

	rks := newRemoteKeySet(ctx, s.URL, nil)

	// Ensure the token verifies.
	gotPayload, err := rks.verify(ctx, jws)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotPayload, payload) {
		t.Errorf("expected payload %s got %s", payload, gotPayload)
	}

	// Ensure the token verifies from the cache.
	gotPayload, err = rks.verify(ctx, jws)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotPayload, payload) {
		t.Errorf("expected payload %s got %s", payload, gotPayload)
	}

	// Ensure item signed by wrong token doesn't verify.
	if _, err := rks.verify(context.Background(), badJWS); err == nil {
		t.Errorf("incorrectly verified signature")
	}
}

func TestCacheControl(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	key1 := newRSAKey(t)
	key2 := newRSAKey(t)

	key1.keyID = "key1"
	key2.keyID = "key2"

	payload := []byte("a secret")
	jws1, err := jose.ParseSigned(key1.sign(t, payload))
	if err != nil {
		t.Fatal(err)
	}
	jws2, err := jose.ParseSigned(key2.sign(t, payload))
	if err != nil {
		t.Fatal(err)
	}

	cacheForSeconds := 1200
	now := time.Now()

	server := &keyServer{
		keys: jose.JSONWebKeySet{
			Keys: []jose.JSONWebKey{key1.jwk()},
		},
		setHeaders: func(h http.Header) {
			h.Set("Cache-Control", "max-age="+strconv.Itoa(cacheForSeconds))
		},
	}
	s := httptest.NewServer(server)
	defer s.Close()

	rks := newRemoteKeySet(ctx, s.URL, func() time.Time { return now })

	if _, err := rks.verify(ctx, jws1); err != nil {
		t.Errorf("failed to verify valid signature: %v", err)
	}
	if _, err := rks.verify(ctx, jws2); err == nil {
		t.Errorf("incorrectly verified signature")
	}

	// Add second key to public list.
	server.keys = jose.JSONWebKeySet{
		Keys: []jose.JSONWebKey{key1.jwk(), key2.jwk()},
	}

	if _, err := rks.verify(ctx, jws1); err != nil {
		t.Errorf("failed to verify valid signature: %v", err)
	}
	if _, err := rks.verify(ctx, jws2); err == nil {
		t.Errorf("incorrectly verified signature, still within cache limit")
	}

	// Move time forward. Remote key set should not query the remote server.
	now = now.Add(time.Duration(cacheForSeconds) * time.Second)

	if _, err := rks.verify(ctx, jws1); err != nil {
		t.Errorf("failed to verify valid signature: %v", err)
	}
	if _, err := rks.verify(ctx, jws2); err != nil {
		t.Errorf("failed to verify valid signature: %v", err)
	}

	// Kill server and move time forward again. Keys should still verify.
	s.Close()
	now = now.Add(time.Duration(cacheForSeconds) * time.Second)

	if _, err := rks.verify(ctx, jws1); err != nil {
		t.Errorf("failed to verify valid signature: %v", err)
	}
	if _, err := rks.verify(ctx, jws2); err != nil {
		t.Errorf("failed to verify valid signature: %v", err)
	}
}
