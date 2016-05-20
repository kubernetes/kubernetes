/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"net/http/httptest"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oidc"

	"k8s.io/kubernetes/pkg/auth/user"
	oidctesting "k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc/testing"
)

func generateToken(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups []string, iat, exp time.Time) string {
	signer := op.PrivKey.Signer()
	claims := oidc.NewClaims(iss, sub, aud, iat, exp)
	claims.Add(usernameClaim, value)
	if groups != nil && groupsClaim != "" {
		claims.Add(groupsClaim, groups)
	}

	jwt, err := jose.NewSignedJWT(claims, signer)
	if err != nil {
		t.Fatalf("Cannot generate token: %v", err)
		return ""
	}
	return jwt.Encode()
}

func generateGoodToken(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups []string) string {
	return generateToken(t, op, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now(), time.Now().Add(time.Hour))
}

func generateMalformedToken(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups []string) string {
	return generateToken(t, op, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now(), time.Now().Add(time.Hour)) + "randombits"
}

func generateExpiredToken(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups []string) string {
	return generateToken(t, op, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now().Add(-2*time.Hour), time.Now().Add(-1*time.Hour))
}

func TestOIDCDiscoveryTimeout(t *testing.T) {
	expectErr := fmt.Errorf("failed to fetch provider config after 1 retries")
	_, err := New(OIDCOptions{"https://127.0.0.1:9999/bar", "client-foo", "", "sub", "", 1, 100 * time.Millisecond})
	if !reflect.DeepEqual(err, expectErr) {
		t.Errorf("Expecting %v, but got %v", expectErr, err)
	}
}

func TestOIDCDiscoveryNoKeyEndpoint(t *testing.T) {
	var err error
	expectErr := fmt.Errorf("failed to fetch provider config after 0 retries")

	cert := path.Join(os.TempDir(), "oidc-cert")
	key := path.Join(os.TempDir(), "oidc-key")

	defer os.Remove(cert)
	defer os.Remove(key)

	oidctesting.GenerateSelfSignedCert(t, "127.0.0.1", cert, key)

	op := oidctesting.NewOIDCProvider(t)
	srv, err := op.ServeTLSWithKeyPair(cert, key)
	if err != nil {
		t.Fatalf("Cannot start server %v", err)
	}
	defer srv.Close()

	op.PCFG = oidc.ProviderConfig{
		Issuer: oidctesting.MustParseURL(srv.URL), // An invalid ProviderConfig. Keys endpoint is required.
	}

	_, err = New(OIDCOptions{srv.URL, "client-foo", cert, "sub", "", 0, 0})
	if !reflect.DeepEqual(err, expectErr) {
		t.Errorf("Expecting %v, but got %v", expectErr, err)
	}
}

func TestOIDCDiscoverySecureConnection(t *testing.T) {
	// Verify that plain HTTP issuer URL is forbidden.
	op := oidctesting.NewOIDCProvider(t)
	srv := httptest.NewServer(op.Mux)
	defer srv.Close()

	op.PCFG = oidc.ProviderConfig{
		Issuer:       oidctesting.MustParseURL(srv.URL),
		KeysEndpoint: oidctesting.MustParseURL(srv.URL + "/keys"),
	}

	expectErr := fmt.Errorf("'oidc-issuer-url' (%q) has invalid scheme (%q), require 'https'", srv.URL, "http")

	_, err := New(OIDCOptions{srv.URL, "client-foo", "", "sub", "", 0, 0})
	if !reflect.DeepEqual(err, expectErr) {
		t.Errorf("Expecting %v, but got %v", expectErr, err)
	}

	// Verify the cert/key pair works.
	cert1 := path.Join(os.TempDir(), "oidc-cert-1")
	key1 := path.Join(os.TempDir(), "oidc-key-1")
	cert2 := path.Join(os.TempDir(), "oidc-cert-2")
	key2 := path.Join(os.TempDir(), "oidc-key-2")

	defer os.Remove(cert1)
	defer os.Remove(key1)
	defer os.Remove(cert2)
	defer os.Remove(key2)

	oidctesting.GenerateSelfSignedCert(t, "127.0.0.1", cert1, key1)
	oidctesting.GenerateSelfSignedCert(t, "127.0.0.1", cert2, key2)

	// Create a TLS server using cert/key pair 1.
	tlsSrv, err := op.ServeTLSWithKeyPair(cert1, key1)
	if err != nil {
		t.Fatalf("Cannot start server: %v", err)
	}
	defer tlsSrv.Close()

	op.PCFG = oidc.ProviderConfig{
		Issuer:       oidctesting.MustParseURL(tlsSrv.URL),
		KeysEndpoint: oidctesting.MustParseURL(tlsSrv.URL + "/keys"),
	}

	// Create a client using cert2, should fail.
	_, err = New(OIDCOptions{tlsSrv.URL, "client-foo", cert2, "sub", "", 0, 0})
	if err == nil {
		t.Fatalf("Expecting error, but got nothing")
	}

}

func TestOIDCAuthentication(t *testing.T) {
	var err error

	cert := path.Join(os.TempDir(), "oidc-cert")
	key := path.Join(os.TempDir(), "oidc-key")

	defer os.Remove(cert)
	defer os.Remove(key)

	oidctesting.GenerateSelfSignedCert(t, "127.0.0.1", cert, key)

	// Create a TLS server and a client.
	op := oidctesting.NewOIDCProvider(t)
	srv, err := op.ServeTLSWithKeyPair(cert, key)
	if err != nil {
		t.Fatalf("Cannot start server: %v", err)
	}
	defer srv.Close()

	// A provider config with all required fields.
	op.AddMinimalProviderConfig(srv)

	tests := []struct {
		userClaim   string
		groupsClaim string
		token       string
		userInfo    user.Info
		verified    bool
		err         string
	}{
		{
			"sub",
			"",
			generateGoodToken(t, op, srv.URL, "client-foo", "client-foo", "sub", "user-foo", "", nil),
			&user.DefaultInfo{Name: fmt.Sprintf("%s#%s", srv.URL, "user-foo")},
			true,
			"",
		},
		{
			// Use user defined claim (email here).
			"email",
			"",
			generateGoodToken(t, op, srv.URL, "client-foo", "client-foo", "email", "foo@example.com", "", nil),
			&user.DefaultInfo{Name: "foo@example.com"},
			true,
			"",
		},
		{
			// Use user defined claim (email here).
			"email",
			"",
			generateGoodToken(t, op, srv.URL, "client-foo", "client-foo", "email", "foo@example.com", "groups", []string{"group1", "group2"}),
			&user.DefaultInfo{Name: "foo@example.com"},
			true,
			"",
		},
		{
			// Use user defined claim (email here).
			"email",
			"groups",
			generateGoodToken(t, op, srv.URL, "client-foo", "client-foo", "email", "foo@example.com", "groups", []string{"group1", "group2"}),
			&user.DefaultInfo{Name: "foo@example.com", Groups: []string{"group1", "group2"}},
			true,
			"",
		},
		{
			"sub",
			"",
			generateMalformedToken(t, op, srv.URL, "client-foo", "client-foo", "sub", "user-foo", "", nil),
			nil,
			false,
			"oidc: unable to verify JWT signature: no matching keys",
		},
		{
			// Invalid 'aud'.
			"sub",
			"",
			generateGoodToken(t, op, srv.URL, "client-foo", "client-bar", "sub", "user-foo", "", nil),
			nil,
			false,
			"oidc: JWT claims invalid: invalid claims, 'aud' claim and 'client_id' do not match",
		},
		{
			// Invalid issuer.
			"sub",
			"",
			generateGoodToken(t, op, "http://foo-bar.com", "client-foo", "client-foo", "sub", "user-foo", "", nil),
			nil,
			false,
			"oidc: JWT claims invalid: invalid claim value: 'iss'.",
		},
		{
			"sub",
			"",
			generateExpiredToken(t, op, srv.URL, "client-foo", "client-foo", "sub", "user-foo", "", nil),
			nil,
			false,
			"oidc: JWT claims invalid: token is expired",
		},
	}

	for i, tt := range tests {
		client, err := New(OIDCOptions{srv.URL, "client-foo", cert, tt.userClaim, tt.groupsClaim, 1, 100 * time.Millisecond})
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}

		user, result, err := client.AuthenticateToken(tt.token)
		if tt.err != "" {
			if !strings.HasPrefix(err.Error(), tt.err) {
				t.Errorf("#%d: Expecting: %v..., but got: %v", i, tt.err, err)
			}
		} else {
			if err != nil {
				t.Errorf("#%d: Unexpected error: %v", i, err)
			}
		}
		if !reflect.DeepEqual(tt.verified, result) {
			t.Errorf("#%d: Expecting: %v, but got: %v", i, tt.verified, result)
		}
		if !reflect.DeepEqual(tt.userInfo, user) {
			t.Errorf("#%d: Expecting: %v, but got: %v", i, tt.userInfo, user)
		}
		client.Close()
	}
}
