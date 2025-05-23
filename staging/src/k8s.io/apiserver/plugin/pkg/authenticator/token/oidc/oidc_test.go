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
	"bytes"
	"context"
	"crypto"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"text/template"
	"time"

	"gopkg.in/go-jose/go-jose.v2"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/utils/pointer"
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
	data, err := os.ReadFile(filepath)
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
	name                string
	options             Options
	optsFunc            func(*Options)
	signingKey          *jose.JSONWebKey
	pubKeys             []*jose.JSONWebKey
	claims              string
	want                *user.DefaultInfo
	wantSkip            bool
	wantErr             string
	wantInitErr         string
	wantHealthErrPrefix string
	claimToResponseMap  map[string]string
	openIDConfig        string
	fetchKeysFromRemote bool
}

// Replace formats the contents of v into the provided template.
func replace(tmpl string, v interface{}) string {
	t := template.Must(template.New("test").Parse(tmpl))
	buf := bytes.NewBuffer(nil)
	t.Execute(buf, &v)
	ret := buf.String()
	klog.V(4).Infof("Replaced: %v into: %v", tmpl, ret)
	return ret
}

// newClaimServer returns a new test HTTPS server, which is rigged to return
// OIDC responses to requests that resolve distributed claims. signer is the
// signer used for the served JWT tokens.  claimToResponseMap is a map of
// responses that the server will return for each claim it is given.
func newClaimServer(t *testing.T, keys jose.JSONWebKeySet, signer jose.Signer, claimToResponseMap map[string]string, openIDConfig *string) *httptest.Server {
	ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		klog.V(5).Infof("request: %+v", *r)
		switch r.URL.Path {
		case "/.testing/keys":
			w.Header().Set("Content-Type", "application/json")
			keyBytes, err := json.Marshal(keys)
			if err != nil {
				t.Fatalf("unexpected error while marshaling keys: %v", err)
			}
			klog.V(5).Infof("%v: returning: %+v", r.URL, string(keyBytes))
			w.Write(keyBytes)

		// /c/d/bar/.well-known/openid-configuration is used to test issuer url and discovery url with a path
		case "/.well-known/openid-configuration", "/c/d/bar/.well-known/openid-configuration":
			w.Header().Set("Content-Type", "application/json")
			klog.V(5).Infof("%v: returning: %+v", r.URL, *openIDConfig)
			w.Write([]byte(*openIDConfig))
		// These claims are tested in the unit tests.
		case "/groups":
			fallthrough
		case "/rabbits":
			if claimToResponseMap == nil {
				t.Errorf("no claims specified in response")
			}
			claim := r.URL.Path[1:] // "/groups" -> "groups"
			expectedAuth := fmt.Sprintf("Bearer %v_token", claim)
			auth := r.Header.Get("Authorization")
			if auth != expectedAuth {
				t.Errorf("bearer token expected: %q, was %q", expectedAuth, auth)
			}
			jws, err := signer.Sign([]byte(claimToResponseMap[claim]))
			if err != nil {
				t.Errorf("while signing response token: %v", err)
			}
			token, err := jws.CompactSerialize()
			if err != nil {
				t.Errorf("while serializing response token: %v", err)
			}
			w.Write([]byte(token))
		default:
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, "unexpected URL: %v", r.URL)
		}
	}))
	klog.V(4).Infof("Serving OIDC at: %v", ts.URL)
	return ts
}

func toKeySet(keys []*jose.JSONWebKey) jose.JSONWebKeySet {
	ret := jose.JSONWebKeySet{}
	for _, k := range keys {
		ret.Keys = append(ret.Keys, *k)
	}
	return ret
}

func (c *claimsTest) run(t *testing.T) {
	var (
		signer jose.Signer
		err    error
	)
	if c.signingKey != nil {
		// Initialize the signer only in the tests that make use of it.  We can
		// not defer this initialization because the test server uses it too.
		signer, err = jose.NewSigner(jose.SigningKey{
			Algorithm: jose.SignatureAlgorithm(c.signingKey.Algorithm),
			Key:       c.signingKey,
		}, nil)
		if err != nil {
			t.Fatalf("initialize signer: %v", err)
		}
	}
	// The HTTPS server used for requesting distributed groups claims.
	ts := newClaimServer(t, toKeySet(c.pubKeys), signer, c.claimToResponseMap, &c.openIDConfig)
	defer ts.Close()

	// Make the certificate of the helper server available to the authenticator
	caBundle := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: ts.Certificate().Raw,
	})
	caContent, err := dynamiccertificates.NewStaticCAContent("oidc-authenticator", caBundle)
	if err != nil {
		t.Fatalf("initialize ca: %v", err)
	}
	c.options.CAContentProvider = caContent

	// Allow claims to refer to the serving URL of the test server.  For this,
	// substitute all references to {{.URL}} in appropriate places.
	// Use {{.Expired}} to handle the token expiry date string with correct timezone handling.
	v := struct {
		URL     string
		Expired string
	}{
		URL:     ts.URL,
		Expired: fmt.Sprintf("%v", time.Unix(expired.Unix(), 0)),
	}
	c.claims = replace(c.claims, &v)
	c.openIDConfig = replace(c.openIDConfig, &v)
	c.options.JWTAuthenticator.Issuer.URL = replace(c.options.JWTAuthenticator.Issuer.URL, &v)
	c.options.JWTAuthenticator.Issuer.DiscoveryURL = replace(c.options.JWTAuthenticator.Issuer.DiscoveryURL, &v)
	for claim, response := range c.claimToResponseMap {
		c.claimToResponseMap[claim] = replace(response, &v)
	}
	c.wantErr = replace(c.wantErr, &v)
	c.wantInitErr = replace(c.wantInitErr, &v)

	if !c.fetchKeysFromRemote {
		// Set the verifier to use the public key set instead of reading from a remote.
		c.options.KeySet = &staticKeySet{keys: c.pubKeys}
	}

	if c.optsFunc != nil {
		c.optsFunc(&c.options)
	}

	expectInitErr := len(c.wantInitErr) > 0

	ctx := testContext(t)

	// Initialize the authenticator.
	a, err := New(ctx, c.options)
	if err != nil {
		if !expectInitErr {
			t.Fatalf("initialize authenticator: %v", err)
		}
		if got := err.Error(); c.wantInitErr != got {
			t.Fatalf("expected initialization error %q but got %q", c.wantInitErr, got)
		}
		return
	}
	if expectInitErr {
		t.Fatalf("wanted initialization error %q but got none", c.wantInitErr)
	}

	if len(c.wantHealthErrPrefix) > 0 {
		if err := wait.PollUntilContextTimeout(ctx, time.Second, time.Minute, true, func(context.Context) (bool, error) {
			healthErr := a.HealthCheck()
			if healthErr == nil {
				return false, fmt.Errorf("authenticator reported healthy when it should not")
			}

			if strings.HasPrefix(healthErr.Error(), c.wantHealthErrPrefix) {
				return true, nil
			}

			t.Logf("saw health error prefix that did not match: want=%q got=%q", c.wantHealthErrPrefix, healthErr.Error())
			return false, nil
		}); err != nil {
			t.Fatalf("authenticator did not match wanted health error: %v", err)
		}
		return
	}

	claims := struct{}{}
	if err := json.Unmarshal([]byte(c.claims), &claims); err != nil {
		t.Fatalf("failed to unmarshal claims: %v", err)
	}

	// Sign and serialize the claims in a JWT.
	jws, err := signer.Sign([]byte(c.claims))
	if err != nil {
		t.Fatalf("sign claims: %v", err)
	}
	token, err := jws.CompactSerialize()
	if err != nil {
		t.Fatalf("serialize token: %v", err)
	}

	// wait for the authenticator to be healthy
	err = wait.PollUntilContextCancel(ctx, time.Millisecond, true, func(context.Context) (bool, error) {
		return a.HealthCheck() == nil, nil
	})
	if err != nil {
		t.Fatalf("failed to initialize the authenticator: %v", err)
	}

	got, ok, err := a.AuthenticateToken(ctx, token)

	expectErr := len(c.wantErr) > 0

	if err != nil {
		if !expectErr {
			t.Fatalf("authenticate token: %v", err)
		}
		if got := err.Error(); c.wantErr != got {
			t.Fatalf("expected error %q when authenticating token but got %q", c.wantErr, got)
		}
		return
	}

	if expectErr {
		t.Fatalf("expected error %q when authenticating token but got none", c.wantErr)
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

	gotUser := got.User.(*user.DefaultInfo)
	if !reflect.DeepEqual(gotUser, c.want) {
		t.Fatalf("wanted user=%#v, got=%#v", c.want, gotUser)
	}
}

func TestToken(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StructuredAuthenticationConfiguration, true)

	synchronizeTokenIDVerifierForTest = true
	tests := []claimsTest{
		{
			name: "token",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: `oidc: parse username claims "username": claim not present`,
		},
		{
			name: "email",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "email",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "email",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: "oidc: email not verified",
		},
		{
			// If "email_verified" isn't present, assume true
			name: "no-email-verified-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "email",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			want: &user.DefaultInfo{
				Name: "jane@example.com",
			},
		},
		{
			name: "invalid-email-verified-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "email",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			// string value for "email_verified"
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"email": "jane@example.com",
				"email_verified": "false",
				"exp": %d
			}`, valid.Unix()),
			wantErr: "oidc: parse 'email_verified' claim: json: cannot unmarshal string into Go value of type bool",
		},
		{
			name: "groups",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			name: "groups-distributed",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"groups": ["team1", "team2"],
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1", "team2"},
			},
		},
		{
			name: "groups-distributed invalid client",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				Client: &http.Client{Transport: errTransport("some unexpected oidc error")}, // return an error that we can assert against
				now:    func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"groups": ["team1", "team2"],
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			optsFunc: func(opts *Options) {
				opts.CAContentProvider = nil // unset CA automatically set by the test to allow us to use a custom client
			},
			wantErr: `oidc: could not expand distributed claims: while getting distributed claim "groups": Get "{{.URL}}/groups": some unexpected oidc error`,
		},
		{
			name: "groups-distributed-malformed-claim-names",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "nonexistent-claim-source"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"groups": ["team1", "team2"],
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			wantErr: "oidc: verify token: oidc: source does not exist",
		},
		{
			name: "groups-distributed-malformed-names-and-sources",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"groups": ["team1", "team2"],
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			wantErr: "oidc: verify token: oidc: source does not exist",
		},
		{
			name: "groups-distributed-malformed-distributed-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				// Doesn't contain the "groups" claim as it promises.
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			wantErr: `oidc: could not expand distributed claims: jwt returned by distributed claim endpoint "{{.URL}}/groups" did not contain claim: groups`,
		},
		{
			name: "groups-distributed-unusual-name",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "rabbits",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"rabbits": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/rabbits",
								"access_token": "rabbits_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				"rabbits": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"rabbits": ["team1", "team2"],
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1", "team2"},
			},
		},
		{
			name: "groups-distributed-wrong-audience",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				// Note mismatching "aud"
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "your-client",
					"groups": ["team1", "team2"],
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			wantErr: `oidc: could not expand distributed claims: verify distributed claim token: oidc: expected audience "my-client" got ["your-client"]`,
		},
		{
			name: "groups-distributed-expired-token",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				// Note expired timestamp.
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"groups": ["team1", "team2"],
					"exp": %d
			     }`, expired.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			wantErr: "oidc: could not expand distributed claims: verify distributed claim token: oidc: token is expired (Token Expiry: {{.Expired}})",
		},
		{
			// Specs are unclear about this behavior.  We adopt a behavior where
			// normal claim wins over a distributed claim by the same name.
			name: "groups-distributed-normal-claim-wins",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"groups": "team1",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"groups": ["team2"],
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			want: &user.DefaultInfo{
				Name: "jane",
				// "team1" is from the normal "groups" claim.
				Groups: []string{"team1"},
			},
		},
		{
			// Groups should be able to be a single string, not just a slice.
			name: "group-string-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			// Groups should be able to be a single string, not just a slice.
			name: "group-string-claim-distributed",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"groups": "team1",
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1"},
			},
		},
		{
			name: "group-string-claim-aggregated-not-supported",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"JWT": "some.jwt.token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			// if the groups claim isn't provided, this shouldn't error out
			name: "no-groups-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: `oidc: parse groups claim "groups": json: cannot unmarshal number into Go value of type string`,
		},
		{
			name: "required-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Claim:         "hd",
							RequiredValue: "example.com",
						},
						{
							Claim:         "sub",
							RequiredValue: "test",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"hd": "example.com",
				"sub": "test",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "no-required-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Claim:         "hd",
							RequiredValue: "example.com",
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: "oidc: required claim hd not present in ID token",
		},
		{
			name: "invalid-required-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Claim:         "hd",
							RequiredValue: "example.com",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"hd": "example.org",
				"exp": %d
			}`, valid.Unix()),
			wantErr: "oidc: required claim hd value does not match. Got = example.org, want = example.com",
		},
		{
			name: "invalid-signature",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: "oidc: verify token: failed to verify signature: no keys matches jwk keyid",
		},
		{
			name: "expired",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: `oidc: verify token: oidc: token is expired (Token Expiry: {{.Expired}})`,
		},
		{
			name: "invalid-aud",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: `oidc: verify token: oidc: expected audience "my-client" got ["not-my-client"]`,
		},
		{
			// ID tokens may contain multiple audiences:
			// https://openid.net/specs/openid-connect-core-1_0.html#IDToken
			name: "multiple-audiences",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			name: "multiple-audiences in authentication config",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:                 "https://auth.example.com",
						Audiences:           []string{"random-client", "my-client"},
						AudienceMatchPolicy: "MatchAny",
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			name: "multiple-audiences in authentication config, multiple matches",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:                 "https://auth.example.com",
						Audiences:           []string{"random-client", "my-client", "other-client"},
						AudienceMatchPolicy: "MatchAny",
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": ["not-my-client", "my-client", "other-client"],
				"azp": "not-my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "multiple-audiences in authentication config, no match",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:                 "https://auth.example.com",
						Audiences:           []string{"random-client", "my-client"},
						AudienceMatchPolicy: "MatchAny",
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": ["not-my-client"],
				"azp": "not-my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			wantErr: `oidc: verify token: oidc: expected audience in ["my-client" "random-client"] got ["not-my-client"]`,
		},
		{
			name: "nuanced audience validation using claim validation rules",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:                 "https://auth.example.com",
						Audiences:           []string{"bar", "foo", "baz"},
						AudienceMatchPolicy: "MatchAny",
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: `sets.equivalent(claims.aud, ["bar", "foo", "baz"])`,
							Message:    "audience must exactly contain [bar, foo, baz]",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": ["foo", "bar", "baz"],
				"azp": "not-my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "audience validation using claim validation rules fails",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:                 "https://auth.example.com",
						Audiences:           []string{"bar", "foo", "baz"},
						AudienceMatchPolicy: "MatchAny",
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: `sets.equivalent(claims.aud, ["bar", "foo", "baz"])`,
							Message:    "audience must exactly contain [bar, foo, baz]",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": ["foo", "baz"],
				"azp": "not-my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			wantErr: `oidc: error evaluating claim validation expression: validation expression 'sets.equivalent(claims.aud, ["bar", "foo", "baz"])' failed: audience must exactly contain [bar, foo, baz]`,
		},
		{
			name: "invalid-issuer",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				now: func() time.Time { return now },
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
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("oidc:"),
						},
					},
				},
				now: func() time.Time { return now },
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
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("oidc:"),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String("groups:"),
						},
					},
				},
				now: func() time.Time { return now },
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
			name: "groups-prefix-distributed",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "{{.URL}}",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("oidc:"),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String("groups:"),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "{{.URL}}",
				"aud": "my-client",
				"username": "jane",
				"_claim_names": {
						"groups": "src1"
				},
				"_claim_sources": {
						"src1": {
								"endpoint": "{{.URL}}/groups",
								"access_token": "groups_token"
						}
				},
				"exp": %d
			}`, valid.Unix()),
			claimToResponseMap: map[string]string{
				"groups": fmt.Sprintf(`{
					"iss": "{{.URL}}",
				    "aud": "my-client",
					"groups": ["team1", "team2"],
					"exp": %d
			     }`, valid.Unix()),
			},
			openIDConfig: `{
					"issuer": "{{.URL}}",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			want: &user.DefaultInfo{
				Name:   "oidc:jane",
				Groups: []string{"groups:team1", "groups:team2"},
			},
		},
		{
			name: "invalid-signing-alg",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: `oidc: verify token: oidc: id token signed with unsupported algorithm, expected ["RS256"] got "PS256"`,
		},
		{
			name: "ps256",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
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
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
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
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "http://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				now: func() time.Time { return now },
			},
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			wantInitErr: `issuer.url: Invalid value: "http://auth.example.com": URL scheme must be https`,
		},
		{
			name: "no-username-claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			wantInitErr: `claimMappings.username: Required value: claim or expression is required`,
		},
		{
			name: "invalid-sig-alg",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				SupportedSigningAlgs: []string{"HS256"},
				now:                  func() time.Time { return now },
			},
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			wantInitErr: `oidc: unsupported signing alg: "HS256"`,
		},
		{
			name: "client and ca mutually exclusive",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				SupportedSigningAlgs: []string{"RS256"},
				now:                  func() time.Time { return now },
				Client:               http.DefaultClient, // test automatically sets CAContentProvider
			},
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			wantInitErr: "oidc: Client and CAContentProvider are mutually exclusive",
		},
		{
			name: "keyset and discovery URL mutually exclusive",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:          "https://auth.example.com",
						DiscoveryURL: "https://auth.example.com/foo",
						Audiences:    []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				SupportedSigningAlgs: []string{"RS256"},
				now:                  func() time.Time { return now },
				KeySet:               &staticKeySet{},
			},
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			wantInitErr: "oidc: KeySet and DiscoveryURL are mutually exclusive",
		},
		{
			name: "health check failure",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://this-will-not-work.notatld",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				SupportedSigningAlgs: []string{"RS256"},
			},
			fetchKeysFromRemote: true,
			wantHealthErrPrefix: `oidc: authenticator for issuer "https://this-will-not-work.notatld" is not healthy: Get "https://this-will-not-work.notatld/.well-known/openid-configuration": dial tcp: lookup this-will-not-work.notatld`,
		},
		{
			name: "accounts.google.com issuer",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://accounts.google.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "email",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			claims: fmt.Sprintf(`{
				"iss": "accounts.google.com",
				"email": "thomas.jefferson@gmail.com",
				"aud": "my-client",
				"exp": %d
			}`, valid.Unix()),
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			want: &user.DefaultInfo{
				Name: "thomas.jefferson@gmail.com",
			},
		},
		{
			name: "good token with bad client id",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("prefix:"),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-wrong-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			wantErr: `oidc: verify token: oidc: expected audience "my-client" got ["my-wrong-client"]`,
		},
		{
			name: "user validation rule fails for user.username",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("system:"),
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: "!user.username.startsWith('system:')",
							Message:    "username cannot used reserved system: prefix",
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: `oidc: error evaluating user info validation rule: validation expression '!user.username.startsWith('system:')' failed: username cannot used reserved system: prefix`,
		},
		{
			name: "user validation rule fails for user.groups",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Claim:  "groups",
							Prefix: pointer.String("system:"),
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: "user.groups.all(group, !group.startsWith('system:'))",
							Message:    "groups cannot used reserved system: prefix",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"groups": ["team1", "team2"]
			}`, valid.Unix()),
			wantErr: `oidc: error evaluating user info validation rule: validation expression 'user.groups.all(group, !group.startsWith('system:'))' failed: groups cannot used reserved system: prefix`,
		},
		{
			name: "claim validation rule with expression fails",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: `claims.hd == "example.com"`,
							Message:    "hd claim must be example.com",
						},
					},
				},
				now: func() time.Time { return now },
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
			wantErr: `oidc: error evaluating claim validation expression: expression 'claims.hd == "example.com"' resulted in error: no such key: hd`,
		},
		{
			name: "claim validation rule with expression",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: `claims.hd == "example.com"`,
							Message:    "hd claim must be example.com",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"hd": "example.com"
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "claim validation rule with expression and nested claims",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: `claims.foo.bar == "baz"`,
							Message:    "foo.bar claim must be baz",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"hd": "example.com",
				"foo": {
					"bar": "baz"
				}
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "claim validation rule with mix of expression and claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: `claims.foo.bar == "baz"`,
							Message:    "foo.bar claim must be baz",
						},
						{
							Claim:         "hd",
							RequiredValue: "example.com",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"hd": "example.com",
				"foo": {
					"bar": "baz"
				}
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "username claim mapping with expression",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
					},
				},
				now: func() time.Time { return now },
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
			name: "username claim mapping with expression and nested claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.foo.username",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"foo": {
					"username": "jane"
				}
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "claim mappings with expressions and deeply nested claim - success",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: "claims.turtle.foo.other1.bit1 && !claims.turtle.foo.bar.other1.bit2",
						},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.panda[0]",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.panda[1]",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.panda",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "bio.snorlax.org/1",
								ValueExpression: "string(claims.turtle.foo.bar.other1.bit2)",
							},
							{
								Key:             "bio.snorlax.org/2",
								ValueExpression: "string(claims.turtle.foo.bar.baz.other1.bit3)",
							},
							{
								Key:             "bio.snorlax.org/3",
								ValueExpression: "[string(claims.turtle.foo.bar.baz.other1.bit1)] + ['a', 'b', 'c']",
							},
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: `user.username != "bad"`,
						},
						{
							Expression: `user.uid == "007"`,
						},
						{
							Expression: `"claus" in user.groups`,
						},
						{
							Expression: `user.extra.bio__dot__snorlax__dot__org__slash__3.size() == 4`,
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"exp": %d,
				"turtle": {
					"foo": {
						"1": "a",
						"2": "b",
						"other1": {
							"bit1": true,
							"bit2": false,
							"bit3": 1
						},
						"bar": {
							"3": "c",
							"4": "d",
							"other1": {
								"bit1": true,
								"bit2": false,
								"bit3": 1
							},
							"baz": {
								"5": "e",
								"6": "f",
								"panda": [
									"snorlax",
									"007",
									"santa",
									"claus"
								],
								"other1": {
									"bit1": true,
									"bit2": false,
									"bit3": 1
								}
							}
						}
					}
				}
}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "snorlax",
				UID:    "007",
				Groups: []string{"snorlax", "007", "santa", "claus"},
				Extra: map[string][]string{
					"bio.snorlax.org/1": {"false"},
					"bio.snorlax.org/2": {"1"},
					"bio.snorlax.org/3": {"true", "a", "b", "c"},
				},
			},
		},
		{
			name: "claim mappings with expressions and deeply nested claim - success via optional",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: "claims.turtle.foo.other1.bit1 && !claims.turtle.foo.bar.other1.bit2",
						},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.panda[0]",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.panda[1]",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.?a.b.c.d.orValue([ 'claus' ])", // this passes because of the optional
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "bio.snorlax.org/1",
								ValueExpression: "string(claims.turtle.foo.bar.other1.bit2)",
							},
							{
								Key:             "bio.snorlax.org/2",
								ValueExpression: "string(claims.turtle.foo.bar.baz.other1.bit3)",
							},
							{
								Key:             "bio.snorlax.org/3",
								ValueExpression: "[string(claims.turtle.foo.bar.baz.other1.bit1)] + ['a', 'b', 'c']",
							},
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: `user.username != "bad"`,
						},
						{
							Expression: `user.uid == "007"`,
						},
						{
							Expression: `"claus" in user.groups`,
						},
						{
							Expression: `user.extra.bio__dot__snorlax__dot__org__slash__3.size() == 4`,
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"exp": %d,
				"turtle": {
					"foo": {
						"1": "a",
						"2": "b",
						"other1": {
							"bit1": true,
							"bit2": false,
							"bit3": 1
						},
						"bar": {
							"3": "c",
							"4": "d",
							"other1": {
								"bit1": true,
								"bit2": false,
								"bit3": 1
							},
							"baz": {
								"5": "e",
								"6": "f",
								"panda": [
									"snorlax",
									"007",
									"santa",
									"claus"
								],
								"other1": {
									"bit1": true,
									"bit2": false,
									"bit3": 1
								}
							}
						}
					}
				}
}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "snorlax",
				UID:    "007",
				Groups: []string{"claus"},
				Extra: map[string][]string{
					"bio.snorlax.org/1": {"false"},
					"bio.snorlax.org/2": {"1"},
					"bio.snorlax.org/3": {"true", "a", "b", "c"},
				},
			},
		},
		{
			name: "claim mappings with expressions and deeply nested claim - failure without optional",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: "claims.turtle.foo.other1.bit1 && !claims.turtle.foo.bar.other1.bit2",
						},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.panda[0]",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.panda[1]",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.turtle.foo.bar.baz.a.b.c.d", // this fails because the key does not exist
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "bio.snorlax.org/1",
								ValueExpression: "string(claims.turtle.foo.bar.other1.bit2)",
							},
							{
								Key:             "bio.snorlax.org/2",
								ValueExpression: "string(claims.turtle.foo.bar.baz.other1.bit3)",
							},
							{
								Key:             "bio.snorlax.org/3",
								ValueExpression: "[string(claims.turtle.foo.bar.baz.other1.bit1)] + ['a', 'b', 'c']",
							},
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: `user.username != "bad"`,
						},
						{
							Expression: `user.uid == "007"`,
						},
						{
							Expression: `"claus" in user.groups`,
						},
						{
							Expression: `user.extra.bio__dot__snorlax__dot__org__slash__3.size() == 4`,
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"exp": %d,
				"turtle": {
					"foo": {
						"1": "a",
						"2": "b",
						"other1": {
							"bit1": true,
							"bit2": false,
							"bit3": 1
						},
						"bar": {
							"3": "c",
							"4": "d",
							"other1": {
								"bit1": true,
								"bit2": false,
								"bit3": 1
							},
							"baz": {
								"5": "e",
								"6": "f",
								"panda": [
									"snorlax",
									"007",
									"santa",
									"claus"
								],
								"other1": {
									"bit1": true,
									"bit2": false,
									"bit3": 1
								}
							}
						}
					}
				}
}`, valid.Unix()),
			wantErr: "oidc: error evaluating group claim expression: expression 'claims.turtle.foo.bar.baz.a.b.c.d' resulted in error: no such key: a",
		},
		{
			name: "claim mappings with expressions and deeply nested claim - success - service account payload",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://kubernetes.default.svc",
						Audiences: []string{"https://kubernetes.default.svc"},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{
						{
							Expression: `claims.sub == ("system:serviceaccount:" + claims.kubernetes__dot__io.namespace + ":" + claims.kubernetes__dot__io.serviceaccount.name)`,
						},
						{
							Expression: `has(claims.kubernetes__dot__io.pod.uid)`, // has requires field based access
						},
						{
							Expression: `claims["kubernetes.io"]["node"]["uid"] != ""`, // the [ based syntax is preferred and is easier to read
						},
						{
							Expression: `claims[?"kubernetes.io"]["node"]["ip"].orValue(true)`, // optionals can be used even when has() cannot
						},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: `"system:serviceaccount:" + claims.kubernetes__dot__io.namespace + ":" + claims.kubernetes__dot__io.serviceaccount.name`,
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.kubernetes__dot__io.serviceaccount.uid",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: `[ "system:serviceaccounts", "system:serviceaccounts:" + claims.kubernetes__dot__io.namespace ]`,
						},
						Extra: []apiserver.ExtraMapping{ // use x-kubernetes.io since validation prevents the use of kubernetes.io
							{
								Key:             "authentication.x-kubernetes.io/pod-name",
								ValueExpression: "claims.kubernetes__dot__io.pod.name",
							},
							{
								Key:             "authentication.x-kubernetes.io/pod-uid",
								ValueExpression: "claims.kubernetes__dot__io.pod.uid",
							},
							{
								Key:             "authentication.x-kubernetes.io/node-name",
								ValueExpression: "claims.kubernetes__dot__io.node.name",
							},
							{
								Key:             "authentication.x-kubernetes.io/node-uid",
								ValueExpression: "claims.kubernetes__dot__io.node.uid",
							},
							{
								Key:             "authentication.x-kubernetes.io/credential-id",
								ValueExpression: `"JTI=" + claims.jti`,
							},
							{
								Key:             "test.x-kubernetes.io/warnafter",
								ValueExpression: "string(int(claims.kubernetes__dot__io.warnafter))",
							},
							{
								Key:             "test.x-kubernetes.io/namespace",
								ValueExpression: "claims.kubernetes__dot__io.namespace",
							},
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: `user.username.startsWith("system:serviceaccount:" + user.extra["test.x-kubernetes.io/namespace"][0] + ":")`, // the [ based syntax is preferred and is easier to read
						},
						{
							Expression: `user.uid != ""`,
						},
						{
							Expression: `"system:serviceaccounts" in user.groups && user.groups.size() == 2`,
						},
						{
							Expression: `user.extra.authentication__dot__x__dash__kubernetes__dot__io__slash__node__dash__name[0] == "127.0.0.1"`,
						},
						{
							Expression: `user.extra.authentication__dot__x__dash__kubernetes__dot__io__slash__credential__dash__id[0] == user.extra.authentication__dot__kubernetes__dot__io__slash__credential__dash__id[0]`,
						},
						{
							Expression: `user.extra["test.x-kubernetes.io/warnafter"][0] == "1700081020"`,
						},
						{
							Expression: `has(user.extra.authentication__dot__x__dash__kubernetes__dot__io__slash__pod__dash__name)`, // has requires field based access
						},
						{
							Expression: `user.extra[?"test.x-kubernetes.io/missing"].orValue([]) == []`, // optionals can be used even when has() cannot
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"aud": [
					"https://kubernetes.default.svc"
				],
				"exp": %d,
				"iat": 1700077413,
				"iss": "https://kubernetes.default.svc",
				"jti": "ea28ed49-2e11-4280-9ec5-bc3d1d84661a",
				"kubernetes.io": {
					"namespace": "kube-system",
					"node": {
						"name": "127.0.0.1",
						"uid": "58456cb0-dd00-45ed-b797-5578fdceaced"
					},
					"pod": {
						"name": "coredns-69cbfb9798-jv9gn",
						"uid": "778a530c-b3f4-47c0-9cd5-ab018fb64f33"
					},
					"serviceaccount": {
						"name": "coredns",
						"uid": "a087d5a0-e1dd-43ec-93ac-f13d89cd13af"
					},
					"warnafter": 1700081020
				},
				"nbf": %d,
				"sub": "system:serviceaccount:kube-system:coredns"
}`, valid.Unix(), now.Unix()),
			want: &user.DefaultInfo{
				Name: "system:serviceaccount:kube-system:coredns",
				UID:  "a087d5a0-e1dd-43ec-93ac-f13d89cd13af",
				Groups: []string{
					"system:serviceaccounts", "system:serviceaccounts:kube-system",
				},
				Extra: map[string][]string{
					"authentication.x-kubernetes.io/pod-name": {
						"coredns-69cbfb9798-jv9gn",
					},
					"authentication.x-kubernetes.io/pod-uid": {
						"778a530c-b3f4-47c0-9cd5-ab018fb64f33",
					},
					"authentication.x-kubernetes.io/node-name": {
						"127.0.0.1",
					},
					"authentication.x-kubernetes.io/node-uid": {
						"58456cb0-dd00-45ed-b797-5578fdceaced",
					},
					"authentication.kubernetes.io/credential-id": {
						"JTI=ea28ed49-2e11-4280-9ec5-bc3d1d84661a",
					},
					"authentication.x-kubernetes.io/credential-id": {
						"JTI=ea28ed49-2e11-4280-9ec5-bc3d1d84661a",
					},
					"test.x-kubernetes.io/warnafter": {
						"1700081020",
					},
					"test.x-kubernetes.io/namespace": {
						"kube-system",
					},
				},
			},
		},
		{
			name: "claim mappings with expressions and deeply nested claim - success - claims payload with deep use of __dot__",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimValidationRules: []apiserver.ClaimValidationRule{ // has requires field based access
						{
							Expression: `has(claims.turtle.foo.bar.baz.username__dot__io)`,
						},
						{
							Expression: `has(claims.turtle.foo.bar.baz.link[0].uid__dot__io)`,
						},
						{
							Expression: `has(claims.data[0].groups__dot__io)`,
						},
						{
							Expression: `claims.turtle.foo.bar.bee.username__dot__io.company__dot__io == "fun-corp"`,
						},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: `claims.turtle.foo.bar.baz.username__dot__io`,
						},
						UID: apiserver.ClaimOrExpression{
							Expression: `claims.turtle.foo.bar.baz.link[0].uid__dot__io`,
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: `claims.data[0].groups__dot__io`,
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"exp": %d,
				"data": [
					{
						"groups.io": "gen1"
					}
				],
				"turtle": {
					"foo": {
						"1": "a",
						"2": "b",
						"other1": {
							"bit1": true,
							"bit2": false,
							"bit3": 1
						},
						"bar": {
							"3": "c",
							"4": "d",
							"other1": {
								"bit1": true,
								"bit2": false,
								"bit3": 1
							},
							"bee": {
								"username.io": {
									"company.io": "fun-corp"
								}
							},
							"baz": {
								"5": "e",
								"6": "f",
								"panda": [
									"snorlax",
									"007",
									"santa",
									"claus"
								],
								"other1": {
									"bit1": true,
									"bit2": false,
									"bit3": 1
								},
								"username.io": "snorlax.io",
								"link": [
									{
										"uid.io": "143"
									}
								]
							}
						}
					}
				}
}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "snorlax.io",
				UID:    "143",
				Groups: []string{"gen1"},
			},
		},
		{
			name: "groups claim mapping with expression",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
					},
				},
				now: func() time.Time { return now },
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
			name: "groups claim with expression",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String("oidc:"),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: `(claims.roles.split(",") + claims.other_roles.split(",")).map(role, "groups:" + role)`,
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"roles": "foo,bar",
				"other_roles": "baz,qux",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "oidc:jane",
				Groups: []string{"groups:foo", "groups:bar", "groups:baz", "groups:qux"},
			},
		},
		{
			name: "uid claim mapping with expression",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.uid",
						},
					},
				},
				now: func() time.Time { return now },
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
				"exp": %d,
				"uid": "1234"
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1", "team2"},
				UID:    "1234",
			},
		},
		{
			name: "uid claim mapping with claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
						UID: apiserver.ClaimOrExpression{
							Claim: "uid",
						},
					},
				},
				now: func() time.Time { return now },
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
				"exp": %d,
				"uid": "1234"
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1", "team2"},
				UID:    "1234",
			},
		},
		{
			name: "extra claim mapping with expression",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.uid",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "example.org/foo",
								ValueExpression: "claims.foo",
							},
							{
								Key:             "example.org/bar",
								ValueExpression: "claims.bar",
							},
						},
					},
				},
				now: func() time.Time { return now },
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
				"exp": %d,
				"uid": "1234",
				"foo": "bar",
				"bar": [
					"baz",
					"qux"
				]
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1", "team2"},
				UID:    "1234",
				Extra: map[string][]string{
					"example.org/foo": {"bar"},
					"example.org/bar": {"baz", "qux"},
				},
			},
		},
		{
			name: "extra claim mapping, value derived from claim value",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "example.org/admin",
								ValueExpression: `(has(claims.is_admin) && claims.is_admin) ? "true":""`,
							},
							{
								Key:             "example.org/admin_1",
								ValueExpression: `claims.?is_admin.orValue(false) == true ? "true":""`,
							},
							{
								Key:             "example.org/non_existent",
								ValueExpression: `claims.?non_existent.orValue("default") == "default" ? "true":""`,
							},
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"is_admin": true
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
				Extra: map[string][]string{
					"example.org/admin":        {"true"},
					"example.org/admin_1":      {"true"},
					"example.org/non_existent": {"true"},
				},
			},
		},
		{
			name: "hardcoded extra claim mapping",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "example.org/admin",
								ValueExpression: `"true"`,
							},
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"is_admin": true
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
				Extra: map[string][]string{
					"example.org/admin": {"true"},
				},
			},
		},
		{
			name: "extra claim mapping, multiple expressions for same key",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.uid",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "example.org/foo",
								ValueExpression: "claims.foo",
							},
							{
								Key:             "example.org/bar",
								ValueExpression: "claims.bar",
							},
							{
								Key:             "example.org/foo",
								ValueExpression: "claims.bar",
							},
						},
					},
				},
				now: func() time.Time { return now },
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
				"exp": %d,
				"uid": "1234",
				"foo": "bar",
				"bar": [
					"baz",
					"qux"
				]
			}`, valid.Unix()),
			wantInitErr: `claimMappings.extra[2].key: Duplicate value: "example.org/foo"`,
		},
		{
			name: "disallowed issuer via configured value",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.uid",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "example.org/foo",
								ValueExpression: "claims.foo",
							},
							{
								Key:             "example.org/bar",
								ValueExpression: "claims.bar",
							},
						},
					},
				},
				DisallowedIssuers: []string{"https://auth.example.com"},
				now:               func() time.Time { return now },
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
				"exp": %d,
				"uid": "1234",
				"foo": "bar",
				"bar": [
					"baz",
					"qux"
				]
			}`, valid.Unix()),
			wantInitErr: `issuer.url: Invalid value: "https://auth.example.com": URL must not overlap with disallowed issuers: [https://auth.example.com]`,
		},
		{
			name: "extra claim mapping, empty string value for key",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.uid",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "example.org/foo",
								ValueExpression: "claims.foo",
							},
							{
								Key:             "example.org/bar",
								ValueExpression: "claims.bar",
							},
						},
					},
				},
				now: func() time.Time { return now },
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
				"exp": %d,
				"uid": "1234",
				"foo": "",
				"bar": [
					"baz",
					"qux"
				]
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1", "team2"},
				UID:    "1234",
				Extra: map[string][]string{
					"example.org/bar": {"baz", "qux"},
				},
			},
		},
		{
			name: "extra claim mapping with user validation rule succeeds",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
						UID: apiserver.ClaimOrExpression{
							Expression: "claims.uid",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "example.org/foo",
								ValueExpression: "'bar'",
							},
							{
								Key:             "example.org/baz",
								ValueExpression: "claims.baz",
							},
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: "'bar' in user.extra['example.org/foo'] && 'qux' in user.extra['example.org/baz']",
							Message:    "example.org/foo must be bar and example.org/baz must be qux",
						},
					},
				},
				now: func() time.Time { return now },
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
				"exp": %d,
				"uid": "1234",
				"baz": "qux"
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name:   "jane",
				Groups: []string{"team1", "team2"},
				UID:    "1234",
				Extra: map[string][]string{
					"example.org/foo": {"bar"},
					"example.org/baz": {"qux"},
				},
			},
		},
		{
			name: "groups expression returns null",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"groups": null,
				"exp": %d,
				"uid": "1234",
				"baz": "qux"
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		// test to ensure omitempty fields not included in user info
		// are set and accessible for CEL evaluation.
		{
			name: "test user validation rule doesn't fail when user info is empty except username",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: `user.username == " "`,
							Message:    "username must be single space",
						},
						{
							Expression: `user.uid == ""`,
							Message:    "uid must be empty string",
						},
						{
							Expression: `!('bar' in user.groups)`,
							Message:    "groups must not contain bar",
						},
						{
							Expression: `!('bar' in user.extra)`,
							Message:    "extra must not contain bar",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": " ",
				"groups": null,
				"exp": %d,
				"baz": "qux"
			}`, valid.Unix()),
			want: &user.DefaultInfo{Name: " "},
		},
		{
			name: "empty username is allowed via claim",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
						Groups: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.groups",
						},
					},
					UserValidationRules: []apiserver.UserValidationRule{
						{
							Expression: `user.username == ""`,
							Message:    "username must be empty string",
						},
						{
							Expression: `user.uid == ""`,
							Message:    "uid must be empty string",
						},
						{
							Expression: `!('bar' in user.groups)`,
							Message:    "groups must not contain bar",
						},
						{
							Expression: `!('bar' in user.extra)`,
							Message:    "extra must not contain bar",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "",
				"groups": null,
				"exp": %d,
				"baz": "qux"
			}`, valid.Unix()),
			want: &user.DefaultInfo{},
		},
		// test to assert the minimum valid jwt payload
		// the required claims are iss, aud, exp and <claimMappings.Username> (in this case user).
		{
			name: "minimum valid jwt payload",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.user",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"user": "jane",
				"exp": %d
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "discovery-url",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:          "https://auth.example.com",
						DiscoveryURL: "{{.URL}}/.well-known/openid-configuration",
						Audiences:    []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			openIDConfig: `{
					"issuer": "https://auth.example.com",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			fetchKeysFromRemote: true,
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "discovery url, issuer has a path",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:          "https://auth.example.com/a/b/foo",
						DiscoveryURL: "{{.URL}}/.well-known/openid-configuration",
						Audiences:    []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com/a/b/foo",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			openIDConfig: `{
					"issuer": "https://auth.example.com/a/b/foo",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			fetchKeysFromRemote: true,
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "discovery url has a path, issuer url has no path",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:          "https://auth.example.com",
						DiscoveryURL: "{{.URL}}/c/d/bar/.well-known/openid-configuration",
						Audiences:    []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
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
			openIDConfig: `{
					"issuer": "https://auth.example.com",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			fetchKeysFromRemote: true,
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "discovery url and issuer url have paths",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:          "https://auth.example.com/a/b/foo",
						DiscoveryURL: "{{.URL}}/c/d/bar/.well-known/openid-configuration",
						Audiences:    []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com/a/b/foo",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			openIDConfig: `{
					"issuer": "https://auth.example.com/a/b/foo",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			fetchKeysFromRemote: true,
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "discovery url and issuer url have paths, issuer url has trailing slash",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:          "https://auth.example.com/a/b/foo/",
						DiscoveryURL: "{{.URL}}/c/d/bar/.well-known/openid-configuration",
						Audiences:    []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Claim:  "username",
							Prefix: pointer.String(""),
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com/a/b/foo/",
				"aud": "my-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			openIDConfig: `{
					"issuer": "https://auth.example.com/a/b/foo/",
					"jwks_uri": "{{.URL}}/.testing/keys"
			}`,
			fetchKeysFromRemote: true,
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "credential id set in extra even when no extra claim mappings are defined",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"jti": "1234"
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
				Extra: map[string][]string{
					user.CredentialIDKey: {"JTI=1234"},
				},
			},
		},
		{
			name: "credential id set in extra when extra claim mappings are defined",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
						Extra: []apiserver.ExtraMapping{
							{
								Key:             "example.org/foo",
								ValueExpression: "claims.foo",
							},
							{
								Key:             "example.org/bar",
								ValueExpression: "claims.bar",
							},
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"jti": "1234",
				"foo": "bar",
				"bar": [
					"baz",
					"qux"
				]
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
				Extra: map[string][]string{
					user.CredentialIDKey: {"JTI=1234"},
					"example.org/foo":    {"bar"},
					"example.org/bar":    {"baz", "qux"},
				},
			},
		},
		{
			name: "non-string jti claim does not set credential id in extra or error",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
					},
				},
				now: func() time.Time { return now },
			},
			signingKey: loadRSAPrivKey(t, "testdata/rsa_1.pem", jose.RS256),
			pubKeys: []*jose.JSONWebKey{
				loadRSAKey(t, "testdata/rsa_1.pem", jose.RS256),
			},
			claims: fmt.Sprintf(`{
				"iss": "https://auth.example.com",
				"aud": "my-client",
				"username": "jane",
				"exp": %d,
				"jti": 1234
			}`, valid.Unix()),
			want: &user.DefaultInfo{
				Name: "jane",
			},
		},
		{
			name: "missing jti claim does not set credential id in extra or error",
			options: Options{
				JWTAuthenticator: apiserver.JWTAuthenticator{
					Issuer: apiserver.Issuer{
						URL:       "https://auth.example.com",
						Audiences: []string{"my-client"},
					},
					ClaimMappings: apiserver.ClaimMappings{
						Username: apiserver.PrefixedClaimOrExpression{
							Expression: "claims.username",
						},
					},
				},
				now: func() time.Time { return now },
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
	}

	var successTestCount, failureTestCount int
	for _, test := range tests {
		var called bool
		t.Run(test.name, func(t *testing.T) {
			called = true
			test.run(t)
		})
		if test.wantSkip || len(test.wantInitErr) > 0 || len(test.wantHealthErrPrefix) > 0 || !called {
			continue
		}
		// check metrics for success and failure
		if test.wantErr == "" {
			successTestCount++
			testutil.AssertHistogramTotalCount(t, "apiserver_authentication_jwt_authenticator_latency_seconds", map[string]string{"result": "success"}, successTestCount)
		} else {
			failureTestCount++
			testutil.AssertHistogramTotalCount(t, "apiserver_authentication_jwt_authenticator_latency_seconds", map[string]string{"result": "failure"}, failureTestCount)
		}
	}
}

func TestUnmarshalClaimError(t *testing.T) {
	// Ensure error strings returned by unmarshaling claims don't include the claim.
	const token = "96bb299a-02e9-11e8-8673-54ee7553240e" // Fake token for testing.
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

type errTransport string

func (e errTransport) RoundTrip(_ *http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("%s", e)
}

func testContext(t *testing.T) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return ctx
}
