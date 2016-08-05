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
	"fmt"
	"os"
	"path"
	"reflect"
	"sort"
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

func TestTLSConfig(t *testing.T) {
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

	tests := []struct {
		testCase string

		serverCertFile string
		serverKeyFile  string

		trustedCertFile string

		wantErr bool
	}{
		{
			testCase:       "provider using untrusted custom cert",
			serverCertFile: cert1,
			serverKeyFile:  key1,
			wantErr:        true,
		},
		{
			testCase:        "provider using untrusted cert",
			serverCertFile:  cert1,
			serverKeyFile:   key1,
			trustedCertFile: cert2,
			wantErr:         true,
		},
		{
			testCase:        "provider using trusted cert",
			serverCertFile:  cert1,
			serverKeyFile:   key1,
			trustedCertFile: cert1,
			wantErr:         false,
		},
	}

	for _, tc := range tests {
		func() {
			op := oidctesting.NewOIDCProvider(t, "")
			srv, err := op.ServeTLSWithKeyPair(tc.serverCertFile, tc.serverKeyFile)
			if err != nil {
				t.Errorf("%s: %v", tc.testCase, err)
				return
			}
			defer srv.Close()

			issuer := srv.URL
			clientID := "client-foo"

			options := OIDCOptions{
				IssuerURL:     srv.URL,
				ClientID:      clientID,
				CAFile:        tc.trustedCertFile,
				UsernameClaim: "email",
				GroupsClaim:   "groups",
			}

			authenticator, err := New(options)
			if err != nil {
				t.Errorf("%s: failed to initialize authenticator: %v", tc.testCase, err)
				return
			}
			defer authenticator.Close()

			email := "user-1@example.com"
			groups := []string{"group1", "group2"}
			sort.Strings(groups)

			token := generateGoodToken(t, op, issuer, "user-1", clientID, "email", email, "groups", groups)

			// Because this authenticator behaves differently for subsequent requests, run these
			// tests multiple times (but expect the same result).
			for i := 1; i < 4; i++ {

				user, ok, err := authenticator.AuthenticateToken(token)
				if err != nil {
					if !tc.wantErr {
						t.Errorf("%s (req #%d): failed to authenticate token: %v", tc.testCase, i, err)
					}
					continue
				}

				if tc.wantErr {
					t.Errorf("%s (req #%d): expected error authenticating", tc.testCase, i)
					continue
				}
				if !ok {
					t.Errorf("%s (req #%d): did not get user or error", tc.testCase, i)
					continue
				}

				if gotUsername := user.GetName(); email != gotUsername {
					t.Errorf("%s (req #%d): GetName() expected=%q got %q", tc.testCase, i, email, gotUsername)
				}
				gotGroups := user.GetGroups()
				sort.Strings(gotGroups)
				if !reflect.DeepEqual(gotGroups, groups) {
					t.Errorf("%s (req #%d): GetGroups() expected=%q got %q", tc.testCase, i, groups, gotGroups)
				}
			}
		}()
	}
}

func TestOIDCAuthentication(t *testing.T) {
	cert := path.Join(os.TempDir(), "oidc-cert")
	key := path.Join(os.TempDir(), "oidc-key")

	defer os.Remove(cert)
	defer os.Remove(key)

	oidctesting.GenerateSelfSignedCert(t, "127.0.0.1", cert, key)

	// Ensure all tests pass when the issuer is not at a base URL.
	for _, path := range []string{"", "/path/with/trailing/slash/"} {

		// Create a TLS server and a client.
		op := oidctesting.NewOIDCProvider(t, path)
		srv, err := op.ServeTLSWithKeyPair(cert, key)
		if err != nil {
			t.Fatalf("Cannot start server: %v", err)
		}
		defer srv.Close()

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
			client, err := New(OIDCOptions{srv.URL, "client-foo", cert, tt.userClaim, tt.groupsClaim})
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
}
