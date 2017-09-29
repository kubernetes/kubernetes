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
	"os"
	"path"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oidc"

	"k8s.io/apiserver/pkg/authentication/user"
	oidctesting "k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/testing"
)

func generateToken(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups interface{}, iat, exp time.Time, emailVerified bool) string {
	claims := oidc.NewClaims(iss, sub, aud, iat, exp)
	claims.Add(usernameClaim, value)
	if groups != nil && groupsClaim != "" {
		claims.Add(groupsClaim, groups)
	}
	claims.Add("email_verified", emailVerified)

	signer := op.PrivKey.Signer()
	jwt, err := jose.NewSignedJWT(claims, signer)
	if err != nil {
		t.Fatalf("Cannot generate token: %v", err)
		return ""
	}
	return jwt.Encode()
}

func generateTokenWithUnverifiedEmail(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, email string) string {
	return generateToken(t, op, iss, sub, aud, "email", email, "", nil, time.Now(), time.Now().Add(time.Hour), false)
}

func generateGoodToken(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups interface{}) string {
	return generateToken(t, op, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now(), time.Now().Add(time.Hour), true)
}

func generateMalformedToken(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups interface{}) string {
	return generateToken(t, op, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now(), time.Now().Add(time.Hour), true) + "randombits"
}

func generateExpiredToken(t *testing.T, op *oidctesting.OIDCProvider, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups interface{}) string {
	return generateToken(t, op, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now().Add(-2*time.Hour), time.Now().Add(-1*time.Hour), true)
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
				&user.DefaultInfo{Name: "user-foo"},
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
				// Group claim is a string rather than an array. Map that string to a single group.
				"email",
				"groups",
				generateGoodToken(t, op, srv.URL, "client-foo", "client-foo", "email", "foo@example.com", "groups", "group1"),
				&user.DefaultInfo{Name: "foo@example.com", Groups: []string{"group1"}},
				true,
				"",
			},
			{
				// Group claim is not a string or array of strings. Throw out this as invalid.
				"email",
				"groups",
				generateGoodToken(t, op, srv.URL, "client-foo", "client-foo", "email", "foo@example.com", "groups", 1),
				nil,
				false,
				"custom group claim contains invalid type: float64",
			},
			{
				// Email not verified
				"email",
				"",
				generateTokenWithUnverifiedEmail(t, op, srv.URL, "client-foo", "client-foo", "foo@example.com"),
				nil,
				false,
				"email not verified",
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
			client, err := New(OIDCOptions{srv.URL, "client-foo", cert, tt.userClaim, "", tt.groupsClaim, ""})
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

func TestParseTokenClaims(t *testing.T) {
	tests := []struct {
		name string

		// Note this is missing a lot of configuration options because
		// parseTokenClaim doesn't handle:
		//
		// - 'iss' claim matching issuer URL
		// - 'exp' claim having not expired
		// - 'sub' claim matching a trusted client id
		//
		// That logic has coverage in other tests.

		issuerURL      string
		usernameClaim  string
		usernamePrefix string
		groupsClaim    string
		groupsPrefix   string

		claims jose.Claims

		wantUser *user.DefaultInfo
		wantErr  bool
	}{
		{
			name:          "email username",
			issuerURL:     "https://foo.com/",
			usernameClaim: "email",
			claims: jose.Claims{
				"email":          "jane.doe@example.com",
				"email_verified": true,
			},
			wantUser: &user.DefaultInfo{
				Name: "jane.doe@example.com",
			},
		},
		{
			name:          "no email_verified claim",
			issuerURL:     "https://foo.com/",
			usernameClaim: "email",
			claims: jose.Claims{
				"email": "jane.doe@example.com",
			},
			wantErr: true,
		},
		{
			name:          "email unverified",
			issuerURL:     "https://foo.com/",
			usernameClaim: "email",
			claims: jose.Claims{
				"email":          "jane.doe@example.com",
				"email_verified": false,
			},
			wantErr: true,
		},
		{
			name:          "non-email user claim",
			issuerURL:     "https://foo.com/",
			usernameClaim: "name",
			claims: jose.Claims{
				"name": "janedoe",
			},
			wantUser: &user.DefaultInfo{
				Name: "janedoe",
			},
		},
		{
			name:          "groups claim",
			issuerURL:     "https://foo.com/",
			usernameClaim: "name",
			groupsClaim:   "groups",
			claims: jose.Claims{
				"name":   "janedoe",
				"groups": []string{"foo", "bar"},
			},
			wantUser: &user.DefaultInfo{
				Name:   "janedoe",
				Groups: []string{"foo", "bar"},
			},
		},
		{
			name:          "groups claim string",
			issuerURL:     "https://foo.com/",
			usernameClaim: "name",
			groupsClaim:   "groups",
			claims: jose.Claims{
				"name":   "janedoe",
				"groups": "foo",
			},
			wantUser: &user.DefaultInfo{
				Name:   "janedoe",
				Groups: []string{"foo"},
			},
		},
		{
			name:           "username prefix",
			issuerURL:      "https://foo.com/",
			usernameClaim:  "name",
			groupsClaim:    "groups",
			usernamePrefix: "oidc:",
			claims: jose.Claims{
				"name":   "janedoe",
				"groups": []string{"foo", "bar"},
			},
			wantUser: &user.DefaultInfo{
				Name:   "oidc:janedoe",
				Groups: []string{"foo", "bar"},
			},
		},
		{
			name:           "username prefix with email",
			issuerURL:      "https://foo.com/",
			usernameClaim:  "email",
			groupsClaim:    "groups",
			usernamePrefix: "oidc:",
			claims: jose.Claims{
				"email":          "jane.doe@example.com",
				"email_verified": true,
				"groups":         []string{"foo", "bar"},
			},
			wantUser: &user.DefaultInfo{
				Name:   "oidc:jane.doe@example.com",
				Groups: []string{"foo", "bar"},
			},
		},
		{
			name:          "groups prefix",
			issuerURL:     "https://foo.com/",
			usernameClaim: "name",
			groupsClaim:   "groups",
			groupsPrefix:  "oidc:",
			claims: jose.Claims{
				"name":   "janedoe",
				"groups": []string{"foo", "bar"},
			},
			wantUser: &user.DefaultInfo{
				Name:   "janedoe",
				Groups: []string{"oidc:foo", "oidc:bar"},
			},
		},
		{
			name:           "username and groups prefix",
			issuerURL:      "https://foo.com/",
			usernameClaim:  "name",
			groupsClaim:    "groups",
			usernamePrefix: "oidc-user:",
			groupsPrefix:   "oidc:",
			claims: jose.Claims{
				"name":   "janedoe",
				"groups": []string{"foo", "bar"},
			},
			wantUser: &user.DefaultInfo{
				Name:   "oidc-user:janedoe",
				Groups: []string{"oidc:foo", "oidc:bar"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			o := OIDCAuthenticator{
				issuerURL:      test.issuerURL,
				usernameClaim:  test.usernameClaim,
				usernamePrefix: test.usernamePrefix,
				groupsClaim:    test.groupsClaim,
				groupsPrefix:   test.groupsPrefix,
			}

			u, ok, err := o.parseTokenClaims(test.claims)
			if err != nil {
				if !test.wantErr {
					t.Errorf("failed to authenticate user: %v", err)
				}
				return
			}
			if test.wantErr {
				t.Fatalf("expected authentication to fail")
			}

			if !ok {
				// We don't have any cases today when the claims can return
				// no error with an unauthenticated signal.
				//
				// In the future we might.
				t.Fatalf("user wasn't authenticated")
			}

			got := &user.DefaultInfo{
				Name:   u.GetName(),
				UID:    u.GetUID(),
				Groups: u.GetGroups(),
				Extra:  u.GetExtra(),
			}
			if !reflect.DeepEqual(got, test.wantUser) {
				t.Errorf("wanted user=%#v, got=%#v", test.wantUser, got)
			}
		})
	}
}
