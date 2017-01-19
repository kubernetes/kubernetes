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

package testing

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
	"net/url"
	"os"
	"path"
	"path/filepath"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
	"github.com/coreos/go-oidc/oidc"
)

// NewOIDCProvider provides a bare minimum OIDC IdP Server useful for testing.
func NewOIDCProvider(t *testing.T, issuerPath string) *OIDCProvider {
	privKey, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("Cannot create OIDC Provider: %v", err)
		return nil
	}

	op := &OIDCProvider{
		Mux:        http.NewServeMux(),
		PrivKey:    privKey,
		issuerPath: issuerPath,
	}

	op.Mux.HandleFunc(path.Join(issuerPath, "/.well-known/openid-configuration"), op.handleConfig)
	op.Mux.HandleFunc(path.Join(issuerPath, "/keys"), op.handleKeys)

	return op
}

type OIDCProvider struct {
	Mux        *http.ServeMux
	PCFG       oidc.ProviderConfig
	PrivKey    *key.PrivateKey
	issuerPath string
}

func (op *OIDCProvider) ServeTLSWithKeyPair(cert, key string) (*httptest.Server, error) {
	srv := httptest.NewUnstartedServer(op.Mux)

	srv.TLS = &tls.Config{Certificates: make([]tls.Certificate, 1)}
	var err error
	srv.TLS.Certificates[0], err = tls.LoadX509KeyPair(cert, key)
	if err != nil {
		return nil, fmt.Errorf("Cannot load cert/key pair: %v", err)
	}
	srv.StartTLS()

	// The issuer's URL is extended by an optional path. This ensures that the plugin can
	// handle issuers that use a non-root path for discovery (see kubernetes/kubernetes#29749).
	srv.URL = srv.URL + op.issuerPath

	u, err := url.Parse(srv.URL)
	if err != nil {
		return nil, err
	}
	pathFor := func(p string) *url.URL {
		u2 := *u // Shallow copy.
		u2.Path = path.Join(u2.Path, p)
		return &u2
	}

	op.PCFG = oidc.ProviderConfig{
		Issuer:                  u,
		AuthEndpoint:            pathFor("/auth"),
		TokenEndpoint:           pathFor("/token"),
		KeysEndpoint:            pathFor("/keys"),
		ResponseTypesSupported:  []string{"code"},
		SubjectTypesSupported:   []string{"public"},
		IDTokenSigningAlgValues: []string{"RS256"},
	}
	return srv, nil
}

func (op *OIDCProvider) handleConfig(w http.ResponseWriter, req *http.Request) {
	b, err := json.Marshal(&op.PCFG)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(b)
}

func (op *OIDCProvider) handleKeys(w http.ResponseWriter, req *http.Request) {
	keys := struct {
		Keys []jose.JWK `json:"keys"`
	}{
		Keys: []jose.JWK{op.PrivKey.JWK()},
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

// generateSelfSignedCert generates a self-signed cert/key pairs and writes to the certPath/keyPath.
// This method is mostly identical to crypto.GenerateSelfSignedCert except for the 'IsCA' and 'KeyUsage'
// in the certificate template. (Maybe we can merge these two methods).
func GenerateSelfSignedCert(t *testing.T, host, certPath, keyPath string) {
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
