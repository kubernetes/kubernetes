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
	"os"
	"reflect"
	"strings"
	"testing"
	"text/template"
	"time"

	oidc "github.com/coreos/go-oidc"
	"github.com/golang/glog"
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
	name               string
	options            Options
	now                time.Time
	signingKey         *jose.JSONWebKey
	pubKeys            []*jose.JSONWebKey
	claims             string
	want               *user.DefaultInfo
	wantSkip           bool
	wantErr            bool
	wantInitErr        bool
	claimToResponseMap map[string]string
	openIDConfig       string
}

// Replace formats the contents of v into the provided template.
func replace(tmpl string, v interface{}) string {
	t := template.Must(template.New("test").Parse(tmpl))
	buf := bytes.NewBuffer(nil)
	t.Execute(buf, &v)
	ret := buf.String()
	glog.V(4).Infof("Replaced: %v into: %v", tmpl, ret)
	return ret
}

// newClaimServer returns a new test HTTPS server, which is rigged to return
// OIDC responses to requests that resolve distributed claims. signer is the
// signer used for the served JWT tokens.  claimToResponseMap is a map of
// responses that the server will return for each claim it is given.
func newClaimServer(t *testing.T, keys jose.JSONWebKeySet, signer jose.Signer, claimToResponseMap map[string]string, openIDConfig *string) *httptest.Server {
	ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		glog.V(5).Infof("request: %+v", *r)
		switch r.URL.Path {
		case "/.testing/keys":
			w.Header().Set("Content-Type", "application/json")
			keyBytes, err := json.Marshal(keys)
			if err != nil {
				t.Fatalf("unexpected error while marshaling keys: %v", err)
			}
			glog.V(5).Infof("%v: returning: %+v", r.URL, string(keyBytes))
			w.Write(keyBytes)

		case "/.well-known/openid-configuration":
			w.Header().Set("Content-Type", "application/json")
			glog.V(5).Infof("%v: returning: %+v", r.URL, *openIDConfig)
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
	glog.V(4).Infof("Serving OIDC at: %v", ts.URL)
	return ts
}

// writeTempCert writes out the supplied certificate into a temporary file in
// PEM-encoded format.  Returns the name of the temporary file used.  The caller
// is responsible for cleaning the file up.
func writeTempCert(t *testing.T, cert []byte) string {
	tempFile, err := ioutil.TempFile("", "ca.crt")
	if err != nil {
		t.Fatalf("could not open temp file: %v", err)
	}
	block := &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cert,
	}
	if err := pem.Encode(tempFile, block); err != nil {
		t.Fatalf("could not write to temp file %v: %v", tempFile.Name(), err)
	}
	tempFile.Close()
	return tempFile.Name()
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
	// by writing its root CA certificate into a temporary file.
	tempFileName := writeTempCert(t, ts.TLS.Certificates[0].Certificate[0])
	defer os.Remove(tempFileName)
	c.options.CAFile = tempFileName

	// Allow claims to refer to the serving URL of the test server.  For this,
	// substitute all references to {{.URL}} in appropriate places.
	v := struct{ URL string }{URL: ts.URL}
	c.claims = replace(c.claims, &v)
	c.openIDConfig = replace(c.openIDConfig, &v)
	c.options.IssuerURL = replace(c.options.IssuerURL, &v)
	for claim, response := range c.claimToResponseMap {
		c.claimToResponseMap[claim] = replace(response, &v)
	}

	// Initialize the authenticator.
	a, err := newAuthenticator(c.options, func(ctx context.Context, a *Authenticator, config *oidc.Config) {
		// Set the verifier to use the public key set instead of reading from a remote.
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
			wantErr: true,
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
			wantErr: true,
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
			wantErr: true,
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
			// "aud" was "your-client", not "my-client"
			wantErr: true,
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
			// The distributed token is expired.
			wantErr: true,
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
			wantErr: true,
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
			wantErr: true,
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
