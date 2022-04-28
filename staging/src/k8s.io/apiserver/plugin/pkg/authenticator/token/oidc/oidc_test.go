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
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"text/template"
	"time"

	"gopkg.in/square/go-jose.v2"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/klog/v2"
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
	name               string
	options            Options
	optsFunc           func(*Options)
	signingKey         *jose.JSONWebKey
	pubKeys            []*jose.JSONWebKey
	claims             string
	want               *user.DefaultInfo
	wantSkip           bool
	wantErr            string
	wantInitErr        string
	claimToResponseMap map[string]string
	openIDConfig       string
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

		case "/.well-known/openid-configuration":
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
	c.options.IssuerURL = replace(c.options.IssuerURL, &v)
	for claim, response := range c.claimToResponseMap {
		c.claimToResponseMap[claim] = replace(response, &v)
	}
	c.wantErr = replace(c.wantErr, &v)
	c.wantInitErr = replace(c.wantInitErr, &v)

	// Set the verifier to use the public key set instead of reading from a remote.
	c.options.KeySet = &staticKeySet{keys: c.pubKeys}

	if c.optsFunc != nil {
		c.optsFunc(&c.options)
	}

	expectInitErr := len(c.wantInitErr) > 0

	// Initialize the authenticator.
	a, err := New(c.options)
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

	// Sign and serialize the claims in a JWT.
	jws, err := signer.Sign([]byte(c.claims))
	if err != nil {
		t.Fatalf("sign claims: %v", err)
	}
	token, err := jws.CompactSerialize()
	if err != nil {
		t.Fatalf("serialize token: %v", err)
	}

	got, ok, err := a.AuthenticateToken(context.Background(), token)

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
	synchronizeTokenIDVerifierForTest = true
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
			wantErr: `oidc: parse username claims "username": claim not present`,
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
			wantErr: "oidc: email not verified",
		},
		{
			// If "email_verified" isn't present, assume true
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
			want: &user.DefaultInfo{
				Name: "jane@example.com",
			},
		},
		{
			name: "invalid-email-verified-claim",
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
			name: "groups-distributed",
			options: Options{
				IssuerURL:     "{{.URL}}",
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
				IssuerURL:     "{{.URL}}",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "groups",
				Client:        &http.Client{Transport: errTransport("some unexpected oidc error")}, // return an error that we can assert against
				now:           func() time.Time { return now },
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
				IssuerURL:     "{{.URL}}",
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
				IssuerURL:     "{{.URL}}",
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
				IssuerURL:     "{{.URL}}",
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
				IssuerURL:     "{{.URL}}",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "rabbits",
				now:           func() time.Time { return now },
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
				IssuerURL:     "{{.URL}}",
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
				IssuerURL:     "{{.URL}}",
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
				IssuerURL:     "{{.URL}}",
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
			// Groups should be able to be a single string, not just a slice.
			name: "group-string-claim-distributed",
			options: Options{
				IssuerURL:     "{{.URL}}",
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
			wantErr: `oidc: parse groups claim "groups": json: cannot unmarshal number into Go value of type string`,
		},
		{
			name: "required-claim",
			options: Options{
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "groups",
				RequiredClaims: map[string]string{
					"hd":  "example.com",
					"sub": "test",
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
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "groups",
				RequiredClaims: map[string]string{
					"hd": "example.com",
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
				IssuerURL:     "https://auth.example.com",
				ClientID:      "my-client",
				UsernameClaim: "username",
				GroupsClaim:   "groups",
				RequiredClaims: map[string]string{
					"hd": "example.com",
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
			wantErr: "oidc: verify token: failed to verify signature: no keys matches jwk keyid",
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
			wantErr: `oidc: verify token: oidc: token is expired (Token Expiry: {{.Expired}})`,
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
			wantErr: `oidc: verify token: oidc: expected audience "my-client" got ["not-my-client"]`,
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
			name: "groups-prefix-distributed",
			options: Options{
				IssuerURL:      "{{.URL}}",
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
			wantErr: `oidc: verify token: oidc: id token signed with unsupported algorithm, expected ["RS256"] got "PS256"`,
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
			wantInitErr: `'oidc-issuer-url' ("http://auth.example.com") has invalid scheme ("http"), require 'https'`,
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
			wantInitErr: "no username claim provided",
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
			wantInitErr: `oidc: unsupported signing alg: "HS256"`,
		},
		{
			name: "client and ca mutually exclusive",
			options: Options{
				IssuerURL:            "https://auth.example.com",
				ClientID:             "my-client",
				UsernameClaim:        "username",
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
			name: "accounts.google.com issuer",
			options: Options{
				IssuerURL:     "https://accounts.google.com",
				ClientID:      "my-client",
				UsernameClaim: "email",
				now:           func() time.Time { return now },
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
				"aud": "my-wrong-client",
				"username": "jane",
				"exp": %d
			}`, valid.Unix()),
			wantErr: `oidc: verify token: oidc: expected audience "my-client" got ["my-wrong-client"]`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, test.run)
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
