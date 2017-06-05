package oidc

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/json"
	"net/http/httptest"
	"testing"
	"time"

	"golang.org/x/net/context"
	jose "gopkg.in/square/go-jose.v2"
)

func TestVerify(t *testing.T) {
	tests := []verificationTest{
		{
			name: "good token",
			idToken: idToken{
				Issuer: "https://foo",
			},
			config: Config{
				SkipClientIDCheck: true,
				SkipNonceCheck:    true,
				SkipExpiryCheck:   true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
		},
		{
			name: "invalid issuer",
			idToken: idToken{
				Issuer: "foo",
			},
			config: Config{
				SkipClientIDCheck: true,
				SkipNonceCheck:    true,
				SkipExpiryCheck:   true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
			wantErr: true,
		},
		{
			name:   "google accounts without scheme",
			issuer: "https://accounts.google.com",
			idToken: idToken{
				Issuer: "accounts.google.com",
			},
			config: Config{
				SkipClientIDCheck: true,
				SkipNonceCheck:    true,
				SkipExpiryCheck:   true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
		},
		{
			name: "expired token",
			idToken: idToken{
				Issuer: "https://foo",
				Expiry: jsonTime(time.Now().Add(-time.Hour)),
			},
			config: Config{
				SkipClientIDCheck: true,
				SkipNonceCheck:    true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
			wantErr: true,
		},
		{
			name: "invalid signature",
			idToken: idToken{
				Issuer: "https://foo",
			},
			config: Config{
				SkipClientIDCheck: true,
				SkipNonceCheck:    true,
				SkipExpiryCheck:   true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_1},
			wantErr: true,
		},
	}
	for _, test := range tests {
		test.run(t)
	}
}

func TestVerifyAudience(t *testing.T) {
	tests := []verificationTest{
		{
			name: "good audience",
			idToken: idToken{
				Issuer:   "https://foo",
				Audience: []string{"client1"},
			},
			config: Config{
				ClientID:        "client1",
				SkipNonceCheck:  true,
				SkipExpiryCheck: true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
		},
		{
			name: "mismatched audience",
			idToken: idToken{
				Issuer:   "https://foo",
				Audience: []string{"client2"},
			},
			config: Config{
				ClientID:        "client1",
				SkipNonceCheck:  true,
				SkipExpiryCheck: true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
			wantErr: true,
		},
		{
			name: "multiple audiences, one matches",
			idToken: idToken{
				Issuer:   "https://foo",
				Audience: []string{"client2", "client1"},
			},
			config: Config{
				ClientID:        "client1",
				SkipNonceCheck:  true,
				SkipExpiryCheck: true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
		},
	}
	for _, test := range tests {
		test.run(t)
	}
}

func TestVerifySigningAlg(t *testing.T) {
	tests := []verificationTest{
		{
			name: "default signing alg",
			idToken: idToken{
				Issuer: "https://foo",
			},
			config: Config{
				SkipClientIDCheck: true,
				SkipNonceCheck:    true,
				SkipExpiryCheck:   true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			signAlg: RS256, // By default we only support RS256.
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
		},
		{
			name: "bad signing alg",
			idToken: idToken{
				Issuer: "https://foo",
			},
			config: Config{
				SkipClientIDCheck: true,
				SkipNonceCheck:    true,
				SkipExpiryCheck:   true,
			},
			signKey: testKeyRSA_2048_0_Priv,
			signAlg: RS512,
			pubKeys: []jose.JSONWebKey{testKeyRSA_2048_0},
			wantErr: true,
		},
		{
			name: "ecdsa signing",
			idToken: idToken{
				Issuer: "https://foo",
			},
			config: Config{
				SupportedSigningAlgs: []string{ES384},
				SkipClientIDCheck:    true,
				SkipNonceCheck:       true,
				SkipExpiryCheck:      true,
			},
			signAlg: ES384,
			signKey: testKeyECDSA_384_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyECDSA_384_0},
		},
		{
			name: "one of many supported",
			idToken: idToken{
				Issuer: "https://foo",
			},
			config: Config{
				SkipClientIDCheck:    true,
				SkipNonceCheck:       true,
				SkipExpiryCheck:      true,
				SupportedSigningAlgs: []string{RS256, ES384},
			},
			signAlg: ES384,
			signKey: testKeyECDSA_384_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyECDSA_384_0},
		},
		{
			name: "not in requiredAlgs",
			idToken: idToken{
				Issuer: "https://foo",
			},
			config: Config{
				SupportedSigningAlgs: []string{RS256, ES512},
				SkipClientIDCheck:    true,
				SkipNonceCheck:       true,
				SkipExpiryCheck:      true,
			},
			signAlg: ES384,
			signKey: testKeyECDSA_384_0_Priv,
			pubKeys: []jose.JSONWebKey{testKeyECDSA_384_0},
			wantErr: true,
		},
	}
	for _, test := range tests {
		test.run(t)
	}
}

type verificationTest struct {
	name string

	// if not provided defaults to "https://foo"
	issuer string

	// ID token claims and a signing key to create the JWT.
	idToken idToken
	signKey jose.JSONWebKey
	// If supplied use this signing algorithm. If not, guess
	// from the signingKey.
	signAlg string

	config  Config
	pubKeys []jose.JSONWebKey

	wantErr bool
}

func algForKey(t *testing.T, k jose.JSONWebKey) string {
	switch key := k.Key.(type) {
	case *rsa.PrivateKey:
		return RS256
	case *ecdsa.PrivateKey:
		name := key.PublicKey.Params().Name
		switch name {
		case elliptic.P256().Params().Name:
			return ES256
		case elliptic.P384().Params().Name:
			return ES384
		case elliptic.P521().Params().Name:
			return ES512
		}
		t.Fatalf("unsupported ecdsa curve: %s", name)
	default:
		t.Fatalf("unsupported key type %T", key)
	}
	return ""
}

func (v verificationTest) run(t *testing.T) {
	payload, err := json.Marshal(v.idToken)
	if err != nil {
		t.Fatal(err)
	}
	signingAlg := v.signAlg
	if signingAlg == "" {
		signingAlg = algForKey(t, v.signKey)
	}

	signer, err := jose.NewSigner(jose.SigningKey{
		Algorithm: jose.SignatureAlgorithm(signingAlg),
		Key:       &v.signKey,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}

	jws, err := signer.Sign(payload)
	if err != nil {
		t.Fatal(err)
	}

	token, err := jws.CompactSerialize()
	if err != nil {
		t.Fatal(err)
	}

	t0 := time.Now()
	now := func() time.Time { return t0 }

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server := httptest.NewServer(newKeyServer(v.pubKeys...))
	defer server.Close()

	issuer := "https://foo"
	if v.issuer != "" {
		issuer = v.issuer
	}
	verifier := newVerifier(newRemoteKeySet(ctx, server.URL, now), &v.config, issuer)

	if _, err := verifier.Verify(ctx, token); err != nil {
		if !v.wantErr {
			t.Errorf("%s: verify %v", v.name, err)
		}
	} else {
		if v.wantErr {
			t.Errorf("%s: expected error", v.name)
		}
	}
}
