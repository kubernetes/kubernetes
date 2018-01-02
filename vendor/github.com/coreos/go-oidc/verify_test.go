package oidc

import (
	"context"
	"strconv"
	"testing"
	"time"

	jose "gopkg.in/square/go-jose.v2"
)

type testVerifier struct {
	jwk jose.JSONWebKey
}

func (t *testVerifier) verify(ctx context.Context, jws *jose.JSONWebSignature) ([]byte, error) {
	return jws.Verify(&t.jwk)
}

func TestVerify(t *testing.T) {
	tests := []verificationTest{
		{
			name:    "good token",
			idToken: `{"iss":"https://foo"}`,
			config: Config{
				SkipClientIDCheck: true,
				SkipExpiryCheck:   true,
			},
			signKey: newRSAKey(t),
		},
		{
			name:    "invalid issuer",
			issuer:  "https://bar",
			idToken: `{"iss":"https://foo"}`,
			config: Config{
				SkipClientIDCheck: true,
				SkipExpiryCheck:   true,
			},
			signKey: newRSAKey(t),
			wantErr: true,
		},
		{
			name:    "invalid sig",
			idToken: `{"iss":"https://foo"}`,
			config: Config{
				SkipClientIDCheck: true,
				SkipExpiryCheck:   true,
			},
			signKey:         newRSAKey(t),
			verificationKey: newRSAKey(t),
			wantErr:         true,
		},
		{
			name:    "google accounts without scheme",
			issuer:  "https://accounts.google.com",
			idToken: `{"iss":"accounts.google.com"}`,
			config: Config{
				SkipClientIDCheck: true,
				SkipExpiryCheck:   true,
			},
			signKey: newRSAKey(t),
		},
		{
			name:    "expired token",
			idToken: `{"iss":"https://foo","exp":` + strconv.FormatInt(time.Now().Add(-time.Hour).Unix(), 10) + `}`,
			config: Config{
				SkipClientIDCheck: true,
			},
			signKey: newRSAKey(t),
			wantErr: true,
		},
		{
			name:    "unexpired token",
			idToken: `{"iss":"https://foo","exp":` + strconv.FormatInt(time.Now().Add(time.Hour).Unix(), 10) + `}`,
			config: Config{
				SkipClientIDCheck: true,
			},
			signKey: newRSAKey(t),
		},
		{
			name: "expiry as float",
			idToken: `{"iss":"https://foo","exp":` +
				strconv.FormatFloat(float64(time.Now().Add(time.Hour).Unix()), 'E', -1, 64) +
				`}`,
			config: Config{
				SkipClientIDCheck: true,
			},
			signKey: newRSAKey(t),
		},
	}
	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestVerifyAudience(t *testing.T) {
	tests := []verificationTest{
		{
			name:    "good audience",
			idToken: `{"iss":"https://foo","aud":"client1"}`,
			config: Config{
				ClientID:        "client1",
				SkipExpiryCheck: true,
			},
			signKey: newRSAKey(t),
		},
		{
			name:    "mismatched audience",
			idToken: `{"iss":"https://foo","aud":"client2"}`,
			config: Config{
				ClientID:        "client1",
				SkipExpiryCheck: true,
			},
			signKey: newRSAKey(t),
			wantErr: true,
		},
		{
			name:    "multiple audiences, one matches",
			idToken: `{"iss":"https://foo","aud":["client1","client2"]}`,
			config: Config{
				ClientID:        "client2",
				SkipExpiryCheck: true,
			},
			signKey: newRSAKey(t),
		},
	}
	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestVerifySigningAlg(t *testing.T) {
	tests := []verificationTest{
		{
			name:    "default signing alg",
			idToken: `{"iss":"https://foo"}`,
			config: Config{
				SkipClientIDCheck: true,
				SkipExpiryCheck:   true,
			},
			signKey: newRSAKey(t),
		},
		{
			name:    "bad signing alg",
			idToken: `{"iss":"https://foo"}`,
			config: Config{
				SkipClientIDCheck: true,
				SkipExpiryCheck:   true,
			},
			signKey: newECDSAKey(t),
			wantErr: true,
		},
		{
			name:    "ecdsa signing",
			idToken: `{"iss":"https://foo"}`,
			config: Config{
				SupportedSigningAlgs: []string{ES256},
				SkipClientIDCheck:    true,
				SkipExpiryCheck:      true,
			},
			signKey: newECDSAKey(t),
		},
		{
			name:    "one of many supported",
			idToken: `{"iss":"https://foo"}`,
			config: Config{
				SkipClientIDCheck:    true,
				SkipExpiryCheck:      true,
				SupportedSigningAlgs: []string{RS256, ES256},
			},
			signKey: newECDSAKey(t),
		},
		{
			name:    "not in requiredAlgs",
			idToken: `{"iss":"https://foo"}`,
			config: Config{
				SupportedSigningAlgs: []string{RS256, ES512},
				SkipClientIDCheck:    true,
				SkipExpiryCheck:      true,
			},
			signKey: newECDSAKey(t),
			wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

type verificationTest struct {
	// Name of the subtest.
	name string

	// If not provided defaults to "https://foo"
	issuer string

	// JWT payload (just the claims).
	idToken string

	// Key to sign the ID Token with.
	signKey *signingKey
	// If not provided defaults to signKey. Only useful when
	// testing invalid signatures.
	verificationKey *signingKey

	config  Config
	wantErr bool
}

func (v verificationTest) run(t *testing.T) {
	token := v.signKey.sign(t, []byte(v.idToken))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	issuer := "https://foo"
	if v.issuer != "" {
		issuer = v.issuer
	}
	var ks keySet
	if v.verificationKey == nil {
		ks = &testVerifier{v.signKey.jwk()}
	} else {
		ks = &testVerifier{v.verificationKey.jwk()}
	}
	verifier := newVerifier(ks, &v.config, issuer)

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
