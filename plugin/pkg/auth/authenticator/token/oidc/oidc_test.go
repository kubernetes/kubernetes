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
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io/ioutil"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
	"github.com/coreos/go-oidc/oidc"
	"k8s.io/kubernetes/pkg/auth/user"

	oidctesting "k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc/testing"
)

type oidcProvider struct {
	mux     *http.ServeMux
	pcfg    oidc.ProviderConfig
	privKey *key.PrivateKey
}

func newOIDCProvider(t *testing.T) *oidcProvider {
	privKey, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("Cannot create OIDC Provider: %v", err)
		return nil
	}

	op := &oidcProvider{
		mux:     http.NewServeMux(),
		privKey: privKey,
	}

	op.mux.HandleFunc("/.well-known/openid-configuration", op.handleConfig)
	op.mux.HandleFunc("/keys", op.handleKeys)

	return op

}

func mustParseURL(t *testing.T, s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil {
		t.Fatalf("Failed to parse url: %v", err)
	}
	return u
}

func (op *oidcProvider) handleConfig(w http.ResponseWriter, req *http.Request) {
	b, err := json.Marshal(&op.pcfg)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(b)
}

func (op *oidcProvider) handleKeys(w http.ResponseWriter, req *http.Request) {
	keys := struct {
		Keys []jose.JWK `json:"keys"`
	}{
		Keys: []jose.JWK{op.privKey.JWK()},
	}

	b, err := json.Marshal(keys)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Cache-Control", fmt.Sprintf("public, max-age=%d", int(time.Hour.Seconds())))
	w.Header().Set("Expires", time.Now().Add(time.Hour).Format(time.RFC1123))
	w.Header().Set("Content-Type", "application/json")
	w.Write(b)
}

func (op *oidcProvider) generateToken(t *testing.T, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups []string, iat, exp time.Time) string {
	signer := op.privKey.Signer()
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

func (op *oidcProvider) generateGoodToken(t *testing.T, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups []string) string {
	return op.generateToken(t, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now(), time.Now().Add(time.Hour))
}

func (op *oidcProvider) generateMalformedToken(t *testing.T, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups []string) string {
	return op.generateToken(t, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now(), time.Now().Add(time.Hour)) + "randombits"
}

func (op *oidcProvider) generateExpiredToken(t *testing.T, iss, sub, aud string, usernameClaim, value, groupsClaim string, groups []string) string {
	return op.generateToken(t, iss, sub, aud, usernameClaim, value, groupsClaim, groups, time.Now().Add(-2*time.Hour), time.Now().Add(-1*time.Hour))
}

// generateSelfSignedCert generates a self-signed cert/key pairs and writes to the certPath/keyPath.
// This method is mostly identical to util.GenerateSelfSignedCert except for the 'IsCA' and 'KeyUsage'
// in the certificate template. (Maybe we can merge these two methods).
func generateSelfSignedCert(t *testing.T, host, certPath, keyPath string) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}

	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: fmt.Sprintf("%s@%d", host, time.Now().Unix()),
		},
		NotBefore: time.Now(),
		NotAfter:  time.Now().Add(time.Hour * 24 * 365),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA: true,
	}

	if ip := net.ParseIP(host); ip != nil {
		template.IPAddresses = append(template.IPAddresses, ip)
	} else {
		template.DNSNames = append(template.DNSNames, host)
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		t.Fatal(err)
	}

	// Generate cert
	certBuffer := bytes.Buffer{}
	if err := pem.Encode(&certBuffer, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		t.Fatal(err)
	}

	// Generate key
	keyBuffer := bytes.Buffer{}
	if err := pem.Encode(&keyBuffer, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)}); err != nil {
		t.Fatal(err)
	}

	// Write cert
	if err := os.MkdirAll(filepath.Dir(certPath), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(certPath, certBuffer.Bytes(), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// Write key
	if err := os.MkdirAll(filepath.Dir(keyPath), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(keyPath, keyBuffer.Bytes(), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}
}

func TestOIDCDiscoveryTimeout(t *testing.T) {
	maxRetries = 3
	retryBackoff = time.Second
	expectErr := fmt.Errorf("failed to fetch provider config after 3 retries")

	_, err := New("https://foo/bar", "client-foo", "client-secret", "", "sub", "")
	if !reflect.DeepEqual(err, expectErr) {
		t.Errorf("Expecting %v, but got %v", expectErr, err)
	}
}

func TestOIDCDiscoveryNoKeyEndpoint(t *testing.T) {
	var err error
	expectErr := fmt.Errorf("failed to fetch provider config after 3 retries")

	cert := path.Join(os.TempDir(), "oidc-cert")
	key := path.Join(os.TempDir(), "oidc-key")

	defer os.Remove(cert)
	defer os.Remove(key)

	generateSelfSignedCert(t, "127.0.0.1", cert, key)

	op := newOIDCProvider(t)
	srv := httptest.NewUnstartedServer(op.mux)
	srv.TLS = &tls.Config{Certificates: make([]tls.Certificate, 1)}
	srv.TLS.Certificates[0], err = tls.LoadX509KeyPair(cert, key)
	if err != nil {
		t.Fatalf("Cannot load cert/key pair: %v", err)
	}
	srv.StartTLS()
	// TODO: Uncomment when fix #19254
	// defer srv.Close()

	op.pcfg = oidc.ProviderConfig{
		Issuer: mustParseURL(t, srv.URL), // An invalid ProviderConfig. Keys endpoint is required.
	}

	_, err = New(srv.URL, "client-foo", "client-secret", cert, "sub", "")
	if !reflect.DeepEqual(err, expectErr) {
		t.Errorf("Expecting %v, but got %v", expectErr, err)
	}
}

func TestOIDCDiscoverySecureConnection(t *testing.T) {
	maxRetries = 3
	retryBackoff = time.Second

	// Verify that plain HTTP issuer URL is forbidden.
	op := newOIDCProvider(t)
	srv := httptest.NewServer(op.mux)
	// TODO: Uncomment when fix #19254
	// defer srv.Close()

	op.pcfg = oidc.ProviderConfig{
		Issuer:       mustParseURL(t, srv.URL),
		KeysEndpoint: mustParseURL(t, srv.URL+"/keys"),
	}

	expectErr := fmt.Errorf("'oidc-issuer-url' (%q) has invalid scheme (%q), require 'https'", srv.URL, "http")

	_, err := New(srv.URL, "client-foo", "client-secret", "", "sub", "")
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

	generateSelfSignedCert(t, "127.0.0.1", cert1, key1)
	generateSelfSignedCert(t, "127.0.0.1", cert2, key2)

	// Create a TLS server using cert/key pair 1.
	tlsSrv := httptest.NewUnstartedServer(op.mux)
	tlsSrv.TLS = &tls.Config{Certificates: make([]tls.Certificate, 1)}
	tlsSrv.TLS.Certificates[0], err = tls.LoadX509KeyPair(cert1, key1)
	if err != nil {
		t.Fatalf("Cannot load cert/key pair: %v", err)
	}
	tlsSrv.StartTLS()
	// TODO: Uncomment when fix #19254
	// defer tlsSrv.Close()

	op.pcfg = oidc.ProviderConfig{
		Issuer:       mustParseURL(t, tlsSrv.URL),
		KeysEndpoint: mustParseURL(t, tlsSrv.URL+"/keys"),
	}

	// Create a client using cert2, should fail.
	_, err = New(tlsSrv.URL, "client-foo", "client-secret", cert2, "sub", "")
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

	generateSelfSignedCert(t, "127.0.0.1", cert, key)

	// Create a TLS server and a client.
	op := newOIDCProvider(t)
	srv := httptest.NewUnstartedServer(op.mux)
	srv.TLS = &tls.Config{Certificates: make([]tls.Certificate, 1)}
	srv.TLS.Certificates[0], err = tls.LoadX509KeyPair(cert, key)
	if err != nil {
		t.Fatalf("Cannot load cert/key pair: %v", err)
	}
	srv.StartTLS()
	// TODO: Uncomment when fix #19254
	// defer srv.Close()

	// A provider config with all required fields.
	op.pcfg = oidc.ProviderConfig{
		Issuer:                  mustParseURL(t, srv.URL),
		AuthEndpoint:            mustParseURL(t, srv.URL+"/auth"),
		TokenEndpoint:           mustParseURL(t, srv.URL+"/token"),
		KeysEndpoint:            mustParseURL(t, srv.URL+"/keys"),
		ResponseTypesSupported:  []string{"code"},
		SubjectTypesSupported:   []string{"public"},
		IDTokenSigningAlgValues: []string{"RS256"},
	}

	tests := []struct {
		userClaim        string
		groupsClaim      string
		token            string
		userInfo         user.Info
		verified         bool
		err              string
		path             string
		handlersAttached bool
	}{
		{
			userClaim:   "sub",
			groupsClaim: "",
			token:       op.generateGoodToken(t, srv.URL, "client-foo", "client-foo", "sub", "user-foo", "", nil),
			userInfo:    &user.DefaultInfo{Name: fmt.Sprintf("%s#%s", srv.URL, "user-foo")},
			verified:    true,
			err:         "",
		},
		{
			// Use user defined claim (email here).
			userClaim:   "email",
			groupsClaim: "",
			token:       op.generateGoodToken(t, srv.URL, "client-foo", "client-foo", "email", "foo@example.com", "", nil),
			userInfo:    &user.DefaultInfo{Name: "foo@example.com"},
			verified:    true,
			err:         "",
		},
		{
			// Use user defined claim (email here).
			userClaim:   "email",
			groupsClaim: "",
			token:       op.generateGoodToken(t, srv.URL, "client-foo", "client-foo", "email", "foo@example.com", "groups", []string{"group1", "group2"}),
			userInfo:    &user.DefaultInfo{Name: "foo@example.com"},
			verified:    true,
			err:         "",
		},
		{
			// Use user defined claim (email here).
			userClaim:   "email",
			groupsClaim: "groups",
			token:       op.generateGoodToken(t, srv.URL, "client-foo", "client-foo", "email", "foo@example.com", "groups", []string{"group1", "group2"}),
			userInfo:    &user.DefaultInfo{Name: "foo@example.com", Groups: []string{"group1", "group2"}},
			verified:    true,
			err:         "",
		},
		{
			userClaim:   "sub",
			groupsClaim: "",
			token:       op.generateMalformedToken(t, srv.URL, "client-foo", "client-foo", "sub", "user-foo", "", nil),
			userInfo:    nil,
			verified:    false,
			err:         "oidc: unable to verify JWT signature: no matching keys",
		},
		{
			// Invalid 'aud'.
			userClaim:   "sub",
			groupsClaim: "",
			token:       op.generateGoodToken(t, srv.URL, "client-foo", "client-bar", "sub", "user-foo", "", nil),
			userInfo:    nil,
			verified:    false,
			err:         "oidc: JWT claims invalid: invalid claims, 'aud' claim and 'client_id' do not match",
		},
		{
			// Invalid issuer.
			userClaim:   "sub",
			groupsClaim: "",
			token:       op.generateGoodToken(t, "http://foo-bar.com", "client-foo", "client-foo", "sub", "user-foo", "", nil),
			userInfo:    nil,
			verified:    false,
			err:         "oidc: JWT claims invalid: invalid claim value: 'iss'.",
		},
		{
			userClaim:   "sub",
			groupsClaim: "",
			token:       op.generateExpiredToken(t, srv.URL, "client-foo", "client-foo", "sub", "user-foo", "", nil),
			userInfo:    nil,
			verified:    false,
			err:         "oidc: JWT claims invalid: token is expired",
		},
		{
			// handlers are attached, but the path is not one that gets bypassed
			// by OIDC auth.
			userClaim:        "sub",
			groupsClaim:      "",
			token:            "",
			userInfo:         nil,
			verified:         false,
			err:              "",
			handlersAttached: true,
			path:             "/not_an_oidc_path",
		},
		{
			// The path is an OIDC path, but handlers are not attached.
			userClaim:        "sub",
			groupsClaim:      "",
			token:            "",
			userInfo:         nil,
			verified:         false,
			err:              "",
			handlersAttached: false,
			path:             PathExchangeRefreshToken,
		},
		{
			// handlers are attached, and path is an OIDC path.
			userClaim:        "sub",
			groupsClaim:      "",
			token:            "",
			userInfo:         UserUnauthenticated,
			verified:         true,
			err:              "",
			handlersAttached: true,
			path:             PathExchangeRefreshToken,
		},
		{
			// handlers are attached, and path is an OIDC path.
			userClaim:        "sub",
			groupsClaim:      "",
			token:            "",
			userInfo:         UserUnauthenticated,
			verified:         true,
			err:              "",
			handlersAttached: true,
			path:             PathAuthenticate,
		},
		{
			// handlers are attached, and path is an OIDC path.
			userClaim:        "sub",
			groupsClaim:      "",
			token:            "",
			userInfo:         UserUnauthenticated,
			verified:         true,
			err:              "",
			handlersAttached: true,
			path:             PathAuthCallback,
		},
		{
			// handlers are attached, and path is an OIDC path.
			userClaim:        "sub",
			groupsClaim:      "",
			token:            "",
			userInfo:         UserUnauthenticated,
			verified:         true,
			err:              "",
			handlersAttached: true,
			path:             PathExchangeCode,
		},
		{
			// A valid token is passed, but this is an OIDC path and handlers
			// are attached, so we bypass the token auth.
			userClaim:        "sub",
			groupsClaim:      "",
			token:            op.generateGoodToken(t, srv.URL, "client-foo", "client-foo", "sub", "user-foo", "", nil),
			userInfo:         UserUnauthenticated,
			verified:         true,
			err:              "",
			handlersAttached: true,
			path:             PathExchangeRefreshToken,
		},
	}

	for i, tt := range tests {
		client, err := New(srv.URL, "client-foo", "", cert, tt.userClaim, tt.groupsClaim)
		client.SetHandlersAttached(tt.handlersAttached)

		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}

		path := tt.path
		if path == "" {
			path = "/some_path"
		}
		req := &http.Request{
			URL: &url.URL{
				Path: path,
			},
			Header: http.Header{"Authorization": []string{"Bearer " + tt.token}},
		}
		user, result, err := client.AuthenticateRequest(req)
		//user, result, err := client.AuthenticateToken(tt.token)
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

func TestNewOIDCHttpHandler(t *testing.T) {

	tests := []struct {
		clientSecret string
		wantErr      bool
	}{
		{
			clientSecret: "secret",
			wantErr:      false,
		},
		{
			clientSecret: "",
			wantErr:      true,
		},
	}

	for i, tt := range tests {
		auth := OIDCAuthenticator{
			clientConfig: oidc.ClientConfig{
				Credentials: oidc.ClientCredentials{
					Secret: tt.clientSecret,
				},
			},
		}

		hdlr, err := auth.NewOIDCHTTPHandler()
		gotErr := err != nil
		if tt.wantErr != gotErr {
			t.Errorf("#%d: wantErr: %v, gotErr %v, err: %v", i, tt.wantErr, gotErr, err)
		}

		if !tt.wantErr && hdlr == nil {
			t.Errorf("#%d: Want non-nil hdlr", i)
		}
	}
}

func TestHandleExchangeRefreshToken(t *testing.T) {
	op := newOIDCProvider(t)
	goodToken := op.generateGoodToken(t, "http://auth.example.com", "client-foo", "client-foo", "sub", "user-foo", "", nil)
	goodTokenBytes := []byte(goodToken)
	goodTokenJWT, err := jose.ParseJWT(goodToken)
	if err != nil {
		t.Fatalf("Could not parse JWT: %v", err)
	}

	tests := []struct {
		// creating the request
		path   string
		method string
		rt     string

		// response from OIDC Client
		returnIDToken jose.JWT
		returnErr     error

		// expectations
		wantIDTokenBytes []byte
		wantStatus       int
	}{
		{
			// The Happy Path
			path:   PathExchangeRefreshToken,
			method: "POST",
			rt:     "rt_good",

			returnIDToken: goodTokenJWT,

			wantIDTokenBytes: goodTokenBytes,
			wantStatus:       http.StatusOK,
		},
		{
			// Wrong method
			path:   PathExchangeRefreshToken,
			method: "GET",
			rt:     "rt_good",

			wantStatus: http.StatusMethodNotAllowed,
		},
		{
			// Wrong Path
			path:   PathExchangeRefreshToken + "_oops",
			method: "POST",
			rt:     "rt_good",

			wantStatus: http.StatusNotFound,
		},
		{
			// No refresh token provided
			path:   PathExchangeRefreshToken,
			method: "POST",

			wantStatus: http.StatusBadRequest,
		},
		{
			// Error during exchange
			path:   PathExchangeRefreshToken,
			method: "POST",
			rt:     "rt_good",

			returnErr: errors.New("No such Refresh Token"),

			wantStatus: http.StatusBadRequest,
		},
	}

	for i, tt := range tests {
		req, err := http.NewRequest(
			tt.method,
			"https://auth.example.com"+tt.path,
			strings.NewReader(
				url.Values{
					"refresh_token": []string{tt.rt},
				}.Encode(),
			))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		if err != nil {
			t.Errorf("case %d: could not make request: %v", i, err)
		}

		client := &oidctesting.FakeClient{
			Err:     tt.returnErr,
			IDToken: tt.returnIDToken,
		}
		h := NewOIDCHTTPHandler(client)

		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)

		if w.Code != tt.wantStatus {
			t.Errorf("case %d: want %d, got %d", i, tt.wantStatus, w.Code)
		}

		if tt.wantIDTokenBytes != nil {
			resVals, err := url.ParseQuery(w.Body.String())
			if err != nil {
				t.Errorf("case %d: unexpected error parsing response body: %v", i, err)
				continue
			}

			gotIDTokenBytes := []byte(resVals.Get("id_token"))
			if string(gotIDTokenBytes) != string(tt.wantIDTokenBytes) {
				t.Errorf("case %d: want %v, got %v", i, tt.wantIDTokenBytes, gotIDTokenBytes)
			}
		}

	}
}
