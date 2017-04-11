/*
Copyright 2016 The Kubernetes Authors.

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
	"encoding/base64"
	"errors"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
	"github.com/coreos/go-oidc/oauth2"

	oidctesting "k8s.io/client-go/plugin/pkg/auth/authenticator/token/oidc/testing"
)

func clearCache() {
	cache = newClientCache()
}

type persister struct{}

// we don't need to actually persist anything because there's no way for us to
// read from a persister.
func (p *persister) Persist(map[string]string) error { return nil }

type noRefreshOIDCClient struct{}

func (c *noRefreshOIDCClient) refreshToken(rt string) (oauth2.TokenResponse, error) {
	return oauth2.TokenResponse{}, errors.New("alwaysErrOIDCClient: cannot refresh token")
}

func (c *noRefreshOIDCClient) verifyJWT(jwt *jose.JWT) error {
	return nil
}

type mockOIDCClient struct {
	tokenResponse oauth2.TokenResponse
}

func (c *mockOIDCClient) refreshToken(rt string) (oauth2.TokenResponse, error) {
	return c.tokenResponse, nil
}

func (c *mockOIDCClient) verifyJWT(jwt *jose.JWT) error {
	return nil
}

func TestNewOIDCAuthProvider(t *testing.T) {
	tempDir, err := ioutil.TempDir(os.TempDir(), "oidc_test")
	if err != nil {
		t.Fatalf("Cannot make temp dir %v", err)
	}
	cert := path.Join(tempDir, "oidc-cert")
	key := path.Join(tempDir, "oidc-key")
	defer os.RemoveAll(tempDir)

	oidctesting.GenerateSelfSignedCert(t, "127.0.0.1", cert, key)
	op := oidctesting.NewOIDCProvider(t, "")
	srv, err := op.ServeTLSWithKeyPair(cert, key)
	if err != nil {
		t.Fatalf("Cannot start server %v", err)
	}
	defer srv.Close()

	certData, err := ioutil.ReadFile(cert)
	if err != nil {
		t.Fatalf("Could not read cert bytes %v", err)
	}

	makeToken := func(exp time.Time) *jose.JWT {
		jwt, err := jose.NewSignedJWT(jose.Claims(map[string]interface{}{
			"exp": exp.UTC().Unix(),
		}), op.PrivKey.Signer())
		if err != nil {
			t.Fatalf("Could not create signed JWT %v", err)
		}
		return jwt
	}

	t0 := time.Now()

	goodToken := makeToken(t0.Add(time.Hour)).Encode()
	expiredToken := makeToken(t0.Add(-time.Hour)).Encode()

	tests := []struct {
		name string

		cfg         map[string]string
		wantInitErr bool

		client       OIDCClient
		wantCfg      map[string]string
		wantTokenErr bool
	}{
		{
			// A Valid configuration
			name: "no id token and no refresh token",
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
			},
			wantTokenErr: true,
		},
		{
			name: "valid config with an initial token",
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
				cfgIDToken:              goodToken,
			},
			client: new(noRefreshOIDCClient),
			wantCfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
				cfgIDToken:              goodToken,
			},
		},
		{
			name: "invalid ID token with a refresh token",
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
				cfgRefreshToken:         "foo",
				cfgIDToken:              expiredToken,
			},
			client: &mockOIDCClient{
				tokenResponse: oauth2.TokenResponse{
					IDToken: goodToken,
				},
			},
			wantCfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
				cfgRefreshToken:         "foo",
				cfgIDToken:              goodToken,
			},
		},
		{
			name: "invalid ID token with a refresh token, server returns new refresh token",
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
				cfgRefreshToken:         "foo",
				cfgIDToken:              expiredToken,
			},
			client: &mockOIDCClient{
				tokenResponse: oauth2.TokenResponse{
					IDToken:      goodToken,
					RefreshToken: "bar",
				},
			},
			wantCfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
				cfgRefreshToken:         "bar",
				cfgIDToken:              goodToken,
			},
		},
		{
			name: "expired token and no refresh otken",
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "client-secret",
				cfgIDToken:              expiredToken,
			},
			wantTokenErr: true,
		},
		{
			name: "valid base64d ca",
			cfg: map[string]string{
				cfgIssuerUrl:                srv.URL,
				cfgCertificateAuthorityData: base64.StdEncoding.EncodeToString(certData),
				cfgClientID:                 "client-id",
				cfgClientSecret:             "client-secret",
			},
			client:       new(noRefreshOIDCClient),
			wantTokenErr: true,
		},
		{
			name: "missing client ID",
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientSecret:         "client-secret",
			},
			wantInitErr: true,
		},
		{
			name: "missing client secret",
			cfg: map[string]string{
				cfgIssuerUrl:            srv.URL,
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
			},
			wantInitErr: true,
		},
		{
			name: "missing issuer URL",
			cfg: map[string]string{
				cfgCertificateAuthority: cert,
				cfgClientID:             "client-id",
				cfgClientSecret:         "secret",
			},
			wantInitErr: true,
		},
		{
			name: "missing TLS config",
			cfg: map[string]string{
				cfgIssuerUrl:    srv.URL,
				cfgClientID:     "client-id",
				cfgClientSecret: "secret",
			},
			wantInitErr: true,
		},
	}

	for _, tt := range tests {
		clearCache()

		p, err := newOIDCAuthProvider("cluster.example.com", tt.cfg, new(persister))
		if tt.wantInitErr {
			if err == nil {
				t.Errorf("%s: want non-nil err", tt.name)
			}
			continue
		}

		if err != nil {
			t.Errorf("%s: unexpected error on newOIDCAuthProvider: %v", tt.name, err)
			continue
		}

		provider := p.(*oidcAuthProvider)
		provider.client = tt.client
		provider.now = func() time.Time { return t0 }

		if _, err := provider.idToken(); err != nil {
			if !tt.wantTokenErr {
				t.Errorf("%s: failed to get id token: %v", tt.name, err)
			}
			continue
		}
		if tt.wantTokenErr {
			t.Errorf("%s: expected to not get id token: %v", tt.name, err)
			continue
		}

		if !reflect.DeepEqual(tt.wantCfg, provider.cfg) {
			t.Errorf("%s: expected config %#v got %#v", tt.name, tt.wantCfg, provider.cfg)
		}
	}
}

func TestVerifyJWTExpiry(t *testing.T) {
	privKey, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("can't generate private key: %v", err)
	}
	makeToken := func(s string, exp time.Time, count int) *jose.JWT {
		jwt, err := jose.NewSignedJWT(jose.Claims(map[string]interface{}{
			"test":  s,
			"exp":   exp.UTC().Unix(),
			"count": count,
		}), privKey.Signer())
		if err != nil {
			t.Fatalf("Could not create signed JWT %v", err)
		}
		return jwt
	}

	t0 := time.Now()

	tests := []struct {
		name        string
		jwt         *jose.JWT
		now         time.Time
		wantErr     bool
		wantExpired bool
	}{
		{
			name: "valid jwt",
			jwt:  makeToken("foo", t0.Add(time.Hour), 1),
			now:  t0,
		},
		{
			name:    "invalid jwt",
			jwt:     &jose.JWT{},
			now:     t0,
			wantErr: true,
		},
		{
			name:        "expired jwt",
			jwt:         makeToken("foo", t0.Add(-time.Hour), 1),
			now:         t0,
			wantExpired: true,
		},
		{
			name:        "jwt expires soon enough to be marked expired",
			jwt:         makeToken("foo", t0, 1),
			now:         t0,
			wantExpired: true,
		},
	}

	for _, tc := range tests {
		func() {
			valid, err := verifyJWTExpiry(tc.now, tc.jwt.Encode())
			if err != nil {
				if !tc.wantErr {
					t.Errorf("%s: %v", tc.name, err)
				}
				return
			}
			if tc.wantErr {
				t.Errorf("%s: expected error", tc.name)
				return
			}

			if valid && tc.wantExpired {
				t.Errorf("%s: expected token to be expired", tc.name)
			}
			if !valid && !tc.wantExpired {
				t.Errorf("%s: expected token to be valid", tc.name)
			}
		}()
	}
}

func TestClientCache(t *testing.T) {
	cache := newClientCache()

	if _, ok := cache.getClient("issuer1", "id1", "secret1"); ok {
		t.Fatalf("got client before putting one in the cache")
	}

	cli1 := new(oidcAuthProvider)
	cli2 := new(oidcAuthProvider)

	gotcli := cache.setClient("issuer1", "id1", "secret1", cli1)
	if cli1 != gotcli {
		t.Fatalf("set first client and got a different one")
	}

	gotcli = cache.setClient("issuer1", "id1", "secret1", cli2)
	if cli1 != gotcli {
		t.Fatalf("set a second client and didn't get the first")
	}
}
