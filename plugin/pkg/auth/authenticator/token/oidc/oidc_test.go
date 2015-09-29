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
	"fmt"
	"io/ioutil"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
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

func (op *oidcProvider) handleConfig(w http.ResponseWriter, req *http.Request) {
	b, err := json.Marshal(op.pcfg)
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

func (op *oidcProvider) generateToken(t *testing.T, iss, sub, aud string, usernameClaim, value string, iat, exp time.Time) string {
	signer := op.privKey.Signer()
	claims := oidc.NewClaims(iss, sub, aud, iat, exp)
	claims.Add(usernameClaim, value)

	jwt, err := jose.NewSignedJWT(claims, signer)
	if err != nil {
		t.Fatalf("Cannot generate token: %v", err)
		return ""
	}
	return jwt.Encode()
}

func (op *oidcProvider) generateGoodToken(t *testing.T, iss, sub, aud string, usernameClaim, value string) string {
	return op.generateToken(t, iss, sub, aud, usernameClaim, value, time.Now(), time.Now().Add(time.Hour))
}

func (op *oidcProvider) generateMalformedToken(t *testing.T, iss, sub, aud string, usernameClaim, value string) string {
	return op.generateToken(t, iss, sub, aud, usernameClaim, value, time.Now(), time.Now().Add(time.Hour)) + "randombits"
}

func (op *oidcProvider) generateExpiredToken(t *testing.T, iss, sub, aud string, usernameClaim, value string) string {
	return op.generateToken(t, iss, sub, aud, usernameClaim, value, time.Now().Add(-2*time.Hour), time.Now().Add(-1*time.Hour))
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

	_, err := New("https://foo/bar", "client-foo", "", "sub")
	if !reflect.DeepEqual(err, expectErr) {
		t.Errorf("Expecting %v, but got %v", expectErr, err)
	}
}

func TestOIDCDiscoveryNoKeyEndpoint(t *testing.T) {
	var err error
	expectErr := fmt.Errorf("OIDC provider must provide 'jwks_uri' for public key discovery")

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
	defer srv.Close()

	op.pcfg = oidc.ProviderConfig{
		Issuer: srv.URL,
	}

	_, err = New(srv.URL, "client-foo", cert, "sub")
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
	defer srv.Close()

	op.pcfg = oidc.ProviderConfig{
		Issuer:       srv.URL,
		KeysEndpoint: srv.URL + "/keys",
	}

	expectErr := fmt.Errorf("'oidc-issuer-url' (%q) has invalid scheme (%q), require 'https'", srv.URL, "http")

	_, err := New(srv.URL, "client-foo", "", "sub")
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
	defer tlsSrv.Close()

	op.pcfg = oidc.ProviderConfig{
		Issuer:       tlsSrv.URL,
		KeysEndpoint: tlsSrv.URL + "/keys",
	}

	// Create a client using cert2, should fail.
	_, err = New(tlsSrv.URL, "client-foo", cert2, "sub")
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
	defer srv.Close()

	op.pcfg = oidc.ProviderConfig{
		Issuer:       srv.URL,
		KeysEndpoint: srv.URL + "/keys",
	}

	tests := []struct {
		userClaim string
		token     string
		userInfo  user.Info
		verified  bool
		err       string
	}{
		{
			"sub",
			op.generateGoodToken(t, srv.URL, "client-foo", "client-foo", "sub", "user-foo"),
			&user.DefaultInfo{Name: fmt.Sprintf("%s#%s", srv.URL, "user-foo")},
			true,
			"",
		},
		{
			// Use user defined claim (email here).
			"email",
			op.generateGoodToken(t, srv.URL, "client-foo", "client-foo", "email", "foo@example.com"),
			&user.DefaultInfo{Name: "foo@example.com"},
			true,
			"",
		},
		{
			"sub",
			op.generateMalformedToken(t, srv.URL, "client-foo", "client-foo", "sub", "user-foo"),
			nil,
			false,
			"malformed JWS, unable to decode signature",
		},
		{
			// Invalid 'aud'.
			"sub",
			op.generateGoodToken(t, srv.URL, "client-foo", "client-bar", "sub", "user-foo"),
			nil,
			false,
			"oidc: JWT claims invalid: invalid claim value: 'aud'",
		},
		{
			// Invalid issuer.
			"sub",
			op.generateGoodToken(t, "http://foo-bar.com", "client-foo", "client-foo", "sub", "user-foo"),
			nil,
			false,
			"oidc: JWT claims invalid: invalid claim value: 'iss'.",
		},
		{
			"sub",
			op.generateExpiredToken(t, srv.URL, "client-foo", "client-foo", "sub", "user-foo"),
			nil,
			false,
			"oidc: JWT claims invalid: token is expired",
		},
	}

	for i, tt := range tests {
		client, err := New(srv.URL, "client-foo", cert, tt.userClaim)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
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
	}
}
