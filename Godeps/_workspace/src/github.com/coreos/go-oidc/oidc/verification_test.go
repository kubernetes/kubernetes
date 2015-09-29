package oidc

import (
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
)

func TestVerifyClientClaims(t *testing.T) {
	validIss := "https://example.com"
	validClientID := "valid-client"
	now := time.Now()
	tomorrow := now.Add(24 * time.Hour)
	header := jose.JOSEHeader{
		jose.HeaderKeyAlgorithm: "test-alg",
		jose.HeaderKeyID:        "1",
	}

	tests := []struct {
		claims jose.Claims
		ok     bool
	}{
		// valid token
		{
			claims: jose.Claims{
				"iss": validIss,
				"sub": validClientID,
				"aud": validClientID,
				"iat": float64(now.Unix()),
				"exp": float64(tomorrow.Unix()),
			},
			ok: true,
		},
		// missing 'iss' claim
		{
			claims: jose.Claims{
				"sub": validClientID,
				"aud": validClientID,
				"iat": float64(now.Unix()),
				"exp": float64(tomorrow.Unix()),
			},
			ok: false,
		},
		// invalid 'iss' claim
		{
			claims: jose.Claims{
				"iss": "INVALID",
				"sub": validClientID,
				"aud": validClientID,
				"iat": float64(now.Unix()),
				"exp": float64(tomorrow.Unix()),
			},
			ok: false,
		},
		// missing 'sub' claim
		{
			claims: jose.Claims{
				"iss": validIss,
				"aud": validClientID,
				"iat": float64(now.Unix()),
				"exp": float64(tomorrow.Unix()),
			},
			ok: false,
		},
		// invalid 'sub' claim
		{
			claims: jose.Claims{
				"iss": validIss,
				"sub": "INVALID",
				"aud": validClientID,
				"iat": float64(now.Unix()),
				"exp": float64(tomorrow.Unix()),
			},
			ok: false,
		},
		// missing 'aud' claim
		{
			claims: jose.Claims{
				"iss": validIss,
				"sub": validClientID,
				"iat": float64(now.Unix()),
				"exp": float64(tomorrow.Unix()),
			},
			ok: false,
		},
		// invalid 'aud' claim
		{
			claims: jose.Claims{
				"iss": validIss,
				"sub": validClientID,
				"aud": "INVALID",
				"iat": float64(now.Unix()),
				"exp": float64(tomorrow.Unix()),
			},
			ok: false,
		},
		// expired
		{
			claims: jose.Claims{
				"iss": validIss,
				"sub": validClientID,
				"aud": validClientID,
				"iat": float64(now.Unix()),
				"exp": float64(now.Unix()),
			},
			ok: false,
		},
	}

	for i, tt := range tests {
		jwt, err := jose.NewJWT(header, tt.claims)
		if err != nil {
			t.Fatalf("case %d: Failed to generate JWT, error=%v", i, err)
		}

		got, err := VerifyClientClaims(jwt, validIss)
		if tt.ok {
			if err != nil {
				t.Errorf("case %d: unexpected error, err=%v", i, err)
			}
			if got != validClientID {
				t.Errorf("case %d: incorrect client ID, want=%s, got=%s", i, validClientID, got)
			}
		} else if err == nil {
			t.Errorf("case %d: expected error but err is nil", i)
		}
	}
}

func TestJWTVerifier(t *testing.T) {
	iss := "http://example.com"
	now := time.Now()
	future12 := now.Add(12 * time.Hour)
	past36 := now.Add(-36 * time.Hour)
	past12 := now.Add(-12 * time.Hour)

	priv1, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("failed to generate private key, error=%v", err)
	}
	pk1 := *key.NewPublicKey(priv1.JWK())

	priv2, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("failed to generate private key, error=%v", err)
	}
	pk2 := *key.NewPublicKey(priv2.JWK())

	jwtPK1, err := jose.NewSignedJWT(NewClaims(iss, "XXX", "XXX", past12, future12), priv1.Signer())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	jwtPK1BadClaims, err := jose.NewSignedJWT(NewClaims(iss, "XXX", "YYY", past12, future12), priv1.Signer())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	jwtExpired, err := jose.NewSignedJWT(NewClaims(iss, "XXX", "XXX", past36, past12), priv1.Signer())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	jwtPK2, err := jose.NewSignedJWT(NewClaims(iss, "XXX", "XXX", past12, future12), priv2.Signer())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tests := []struct {
		verifier JWTVerifier
		jwt      jose.JWT
		wantErr  bool
	}{
		// JWT signed with available key
		{
			verifier: JWTVerifier{
				issuer:   "example.com",
				clientID: "XXX",
				syncFunc: func() error { return nil },
				keysFunc: func() []key.PublicKey {
					return []key.PublicKey{pk1}
				},
			},
			jwt:     *jwtPK1,
			wantErr: false,
		},

		// JWT signed with available key, with bad claims
		{
			verifier: JWTVerifier{
				issuer:   "example.com",
				clientID: "XXX",
				syncFunc: func() error { return nil },
				keysFunc: func() []key.PublicKey {
					return []key.PublicKey{pk1}
				},
			},
			jwt:     *jwtPK1BadClaims,
			wantErr: true,
		},

		// expired JWT signed with available key
		{
			verifier: JWTVerifier{
				issuer:   "example.com",
				clientID: "XXX",
				syncFunc: func() error { return nil },
				keysFunc: func() []key.PublicKey {
					return []key.PublicKey{pk1}
				},
			},
			jwt:     *jwtExpired,
			wantErr: true,
		},

		// JWT signed with unrecognized key, verifiable after sync
		{
			verifier: JWTVerifier{
				issuer:   "example.com",
				clientID: "XXX",
				syncFunc: func() error { return nil },
				keysFunc: func() func() []key.PublicKey {
					var i int
					return func() []key.PublicKey {
						defer func() { i++ }()
						return [][]key.PublicKey{
							[]key.PublicKey{pk1},
							[]key.PublicKey{pk2},
						}[i]
					}
				}(),
			},
			jwt:     *jwtPK2,
			wantErr: false,
		},

		// JWT signed with unrecognized key, not verifiable after sync
		{
			verifier: JWTVerifier{
				issuer:   "example.com",
				clientID: "XXX",
				syncFunc: func() error { return nil },
				keysFunc: func() []key.PublicKey {
					return []key.PublicKey{pk1}
				},
			},
			jwt:     *jwtPK2,
			wantErr: true,
		},

		// verifier gets no keys from keysFunc, still not verifiable after sync
		{
			verifier: JWTVerifier{
				issuer:   "example.com",
				clientID: "XXX",
				syncFunc: func() error { return nil },
				keysFunc: func() []key.PublicKey {
					return []key.PublicKey{}
				},
			},
			jwt:     *jwtPK1,
			wantErr: true,
		},

		// verifier gets no keys from keysFunc, verifiable after sync
		{
			verifier: JWTVerifier{
				issuer:   "example.com",
				clientID: "XXX",
				syncFunc: func() error { return nil },
				keysFunc: func() func() []key.PublicKey {
					var i int
					return func() []key.PublicKey {
						defer func() { i++ }()
						return [][]key.PublicKey{
							[]key.PublicKey{},
							[]key.PublicKey{pk2},
						}[i]
					}
				}(),
			},
			jwt:     *jwtPK2,
			wantErr: false,
		},
	}

	for i, tt := range tests {
		err := tt.verifier.Verify(tt.jwt)
		if tt.wantErr && (err == nil) {
			t.Errorf("case %d: wanted non-nil error", i)
		} else if !tt.wantErr && (err != nil) {
			t.Errorf("case %d: wanted nil error, got %v", i, err)
		}
	}
}
