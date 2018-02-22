/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package oidc

import (
	"context"
	"crypto"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"reflect"
	"strings"
	"testing"
	"time"

	oidc "github.com/coreos/go-oidc"
	jose "gopkg.in/square/go-jose.v2"
	"k8s.io/apiserver/pkg/authentication/user"
)

// utilities for loading JOSE keys.

func loadRSAKey(t *testing.T, filepath string, alg jose.SignatureAlgorithm) *jose.JSONWebKey {
	return loadKey(t, filepath, alg, func(b []byte) (interface{}, error) {
		key, err := x509.ParsePKCS1PrivateKey(b)
		if err != nil {
			return nil, err
		}
		return key.Public(), nil
	})
}

func loadRSAPrivKey(t *testing.T, filepath string, alg jose.SignatureAlgorithm) *jose.JSONWebKey {
	return loadKey(t, filepath, alg, func(b []byte) (interface{}, error) {
		return x509.ParsePKCS1PrivateKey(b)
	})
}

func loadECDSAKey(t *testing.T, filepath string, alg jose.SignatureAlgorithm) *jose.JSONWebKey {
	return loadKey(t, filepath, alg, func(b []byte) (interface{}, error) {
		key, err := x509.ParseECPrivateKey(b)
		if err != nil {
			return nil, err
		}
		return key.Public(), nil
	})
}

func loadECDSAPrivKey(t *testing.T, filepath string, alg jose.SignatureAlgorithm) *jose.JSONWebKey {
	return loadKey(t, filepath, alg, func(b []byte) (interface{}, error) {
		return x509.ParseECPrivateKey(b)
	})
}

func loadKey(t *testing.T, filepath string, alg jose.SignatureAlgorithm, unmarshal func([]byte) (interface{}, error)) *jose.JSONWebKey {
	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		t.Fatalf("load file: %v", err)
	}
	block, _ := pem.Decode(data)
	if block == nil {
		t.Fatalf("file contained no PEM encoded data: %s", filepath)
	}
	priv, err := unmarshal(block.Bytes)
	if err != nil {
		t.Fatalf("unmarshal key: %v", err)
	}
	key := &jose.JSONWebKey{Key: priv, Use: "sig", Algorithm: string(alg)}
	thumbprint, err := key.Thumbprint(crypto.SHA256)
	if err != nil {
		t.Fatalf("computing thumbprint: %v", err)
	}
	key.KeyID = hex.EncodeToString(thumbprint)
	return key
}

// staticKeySet implements oidc.KeySet.
type staticKeySet struct {
	keys []*jose.JSONWebKey
}

func (s *staticKeySet) VerifySignature(ctx context.Context, jwt string) (payload []byte, err error) {
	jws, err := jose.ParseSigned(jwt)
	if err != nil {
		return nil, err
	}
	if len(jws.Signatures) == 0 {
		return nil, fmt.Errorf("jwt contained no signatures")
	}
	kid := jws.Signatures[0].Header.KeyID

	for _, key := range s.keys {
		if key.KeyID == kid {
			return jws.Verify(key)
		}
	}

	return nil, fmt.Errorf("no keys matches jwk keyid")
}

var (
	expired, _ = time.Parse(time.RFC3339Nano, "2009-11-10T22:00:00Z")
	now, _     = time.Parse(time.RFC3339Nano, "2009-11-10T23:00:00Z")
	valid, _   = time.Parse(time.RFC3339Nano, "2009-11-11T00:00:00Z")
)

type claimsTest struct {
	name        string
	options     Options
	now         time.Time
	signingKey  *jose.JSONWebKey
	pubKeys     []*jose.JSONWebKey
	claims      string
	want        *user.DefaultInfo
	wantSkip    bool
	wantErr     bool
	wantInitErr bool
}

func (c *claimsTest) run(t *testing.T) {
	a, err := newAuthenticator(c.options, func(ctx context.Context, a *Authenticator, config *oidc.Config) {
		// Set the verifier to use the public key set instead of reading
		// from a remote.
		a.setVerifier(oidc.NewVerifier(
			c.options.IssuerURL,
			&staticKeySet{keys: c.pubKeys},
			config,
		))
	})
	if err != nil {
		if !c.wantInitErr {
			t.Fatalf("initialize authenticator: %v", err)
		}
		return
	}
	if c.wantInitErr {
		t.Fatalf("wanted initialization error")
	}

	// Sign and serialize the claims in a JWT.
	signer, err := jose.NewSigner(jose.SigningKey{
		Algorithm: jose.SignatureAlgorithm(c.signingKey.Algorithm),
		Key:       c.signingKey,
	}, nil)
	if err != nil {
		t.Fatalf("initialize signer: %v", err)
	}
	jws, err := signer.Sign([]byte(c.claims))
	if err != nil {
		t.Fatalf("sign claims: %v", err)
	}
	token, err := jws.CompactSerialize()
	if err != nil {
		t.Fatalf("serialize token: %v", err)
	}

	got, ok, err := a.AuthenticateToken(token)
	if err != nil {
		if !c.wantErr {
			t.Fatalf("authenticate token: %v", err)
		}
		return
	}

	if c.wantErr {
		t.Fatalf("expected error authenticating token")
	}
	if !ok {
		if !c.wantSkip {
			// We don't have any cases where we return (nil, false, nil)
			t.Fatalf("no error but token not authenticated")
		}
		return
	}
	if c.wantSkip {
		t.Fatalf("expected authenticator to skip token")
	}

	gotUser := got.(*user.DefaultInfo)
	if !reflect.DeepEqual(gotUser, c.want) {
		t.Fatalf("wanted user=%#v, got=%#v", c.want, gotUser)
	}
}

func TestToken(t *testing.T) {
	tests := []claimsTest{
		{
			name: "token",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "no-username",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"exp": %d
			}`, valid.Unix()),
			wantErr: true,
		},
		{
			name: "email",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "email",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"email": "jane@example.com",
				"email_verified": true,
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane@example.com",
			},
		},
		{
			name: "email-not-verified",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "email",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"email": "jane@example.com",
				"email_verified": false,
				"exp": %d
			}`, valid.Unix()),
			wantErr: true,
		},
		{
			// If "email_verified" isn't present, assume false
			name: "no-email-verified-claim",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "email",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"email": "jane@example.com",
				"exp": %d
			}`, valid.Unix()),
			wantErr: true,
		},
		{
			name: "groups",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "groups",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"groups": ["team1", "team2"],
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1", "team2"},
			},
		},
		{
			// Groups should be able to be a single string, not just a slice.
			name: "group-string-claim",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "groups",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"groups": "team1",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1"},
			},
		},
		{
			// if the groups claim isn't provided, this shouldn't error out
			name: "no-groups-claim",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "groups",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "invalid-groups-claim",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "groups",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"groups": 42,
				"exp": %d
			}`, valid.Unix()),
			wantErr: true,
		},
		{
			name: "invalid-signature",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_2.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			wantErr: true,
		},
		{
			name: "expired",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, expired.Unix()),
			wantErr: true,
		},
		{
			name: "invalid-aud",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "not-my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			wantErr: true,
		},
		{
			// ID tokens may contain multiple audiences:
			// https://openid.net/specs/openid-connect-core-1_0.html#IDToken
			name: "multiple-audiences",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": ["not-my-client", "my-client"],
				"azp": "not-my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "invalid-issuer",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			wantSkip: true,
		},
		{
			name: "username-prefix",
			options: Options{
				IssuerURL:      "https://auth.example.com",
				ClientID:       "my-client",
				UsernameClaim:  "username",
				UsernamePrefix: "oidc:",
				now:            func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "oidc:jane",
			},
		},
		{
			name: "groups-prefix",
			options: Options{
				IssuerURL:      "https://auth.example.com",
				ClientID:       "my-client",
				UsernameClaim:  "username",
				UsernamePrefix: "oidc:",
				GroupsClaim:    "groups",
				GroupsPrefix:   "groups:",
				now:            func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"groups": ["team1", "team2"],
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "oidc:jane",
				Groups: []string{"groups:team1", "groups:team2"},
			},
		},
		{
			name: "invalid-signing-alg",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			// Correct key but invalid signature algorithm "PS256"
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.PS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			wantErr: true,
		},
		{
			name: "ps256",
			options: Options{
				IssuerURL:            "https://auth.example.com",
				ClientID:             "my-client",
				UsernameClaim:        "username",
				SupportedSigningAlgs: []string{"PS256"},
				now:                  func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.PS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.PS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "es512",
			options: Options{
				IssuerURL:            "https://auth.example.com",
				ClientID:             "my-client",
				UsernameClaim:        "username",
				SupportedSigningAlgs: []string{"ES512"},
				now:                  func() time.Time { return now },
			},
			signingKey: loadECDSAPrivKey(t, "testdata/ecdsa_2.pem", jose.ES512),
			pubKeys: []*jose.JSONWebKey{
				loadECDSAKey(t, "testdata/ecdsa_1.pem", jose.ES512),
				loadECDSAKey(t, "testdata/ecdsa_2.pem", jose.ES512),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "not-https",
			options: Options{
				IssuerURL:     "http://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				now:           func() time.Time { return now },
			},
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			wantInitErr: true,
		},
		{
			name: "no-username-claim",
			options: Options{
				IssuerURL: "https://auth.example.com",
				ClientID:  "my-client",
				now:       func() time.Time { return now },
			},
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			wantInitErr: true,
		},
		{
			name: "invalid-sig-alg",
			options: Options{
				IssuerURL:            "https://auth.example.com",
				ClientID:             "my-client",
				UsernameClaim:        "username",
				SupportedSigningAlgs: []string{"HS256"},
				now:                  func() time.Time { return now },
			},
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			wantInitErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestUnmarshalClaimError(t *testing.T) {
	// Ensure error strings returned by unmarshaling claims don't include the claim.
	const token = "96bb299a-02e9-11e8-8673-54ee7553240e"
	payload := fmt.Sprintf(`{
		"token": "%s"
	}`, token)

	var c claims
	if err := json.Unmarshal([]byte(payload), &c); err != nil {
		t.Fatal(err)
	}
	var n int
	err := c.unmarshalClaim("token", &n)
	if err == nil {
		t.Fatal("expected error")
	}

	if strings.Contains(err.Error(), token) {
		t.Fatalf("unmarshal error included token")
	}
}

func TestUnmarshalClaim(t *testing.T) {
	tests := []struct {
		name    string
		claims  string
		do      func(claims) (interface{}, error)
		want    interface{}
		wantErr bool
	}{
		{
			name:   "string claim",
			claims: `{"aud":"foo"}`,
			do: func(c claims) (interface{}, error) {
				var s string
				err := c.unmarshalClaim("aud", &s)
				return s, err
			},
			want: "foo",
		},
		{
			name:   "mismatched types",
			claims: `{"aud":"foo"}`,
			do: func(c claims) (interface{}, error) {
				var n int
				err := c.unmarshalClaim("aud", &n)
				return n, err

			},
			wantErr: true,
		},
		{
			name:   "bool claim",
			claims: `{"email":"foo@coreos.com","email_verified":true}`,
			do: func(c claims) (interface{}, error) {
				var verified bool
				err := c.unmarshalClaim("email_verified", &verified)
				return verified, err
			},
			want: true,
		},
		{
			name:   "strings claim",
			claims: `{"groups":["a","b","c"]}`,
			do: func(c claims) (interface{}, error) {
				var groups []string
				err := c.unmarshalClaim("groups", &groups)
				return groups, err
			},
			want: []string{"a", "b", "c"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var c claims
			if err := json.Unmarshal([]byte(test.claims), &c); err != nil {
				t.Fatal(err)
			}

			got, err := test.do(c)
			if err != nil {
				if test.wantErr {
					return
				}
				t.Fatalf("unexpected error: %v", err)
			}
			if test.wantErr {
				t.Fatalf("expected error")
			}

			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("wanted=%#v, got=%#v", test.want, got)
			}
		})
	}
}
