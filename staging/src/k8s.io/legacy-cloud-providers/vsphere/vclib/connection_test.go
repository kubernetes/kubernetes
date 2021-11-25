/*
Copyright 2018 The Kubernetes Authors.

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

package vclib_test

import (
	"context"
	"crypto/sha1"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"

	"k8s.io/legacy-cloud-providers/vsphere/vclib"
)

func createTestServer(
	t *testing.T,
	caCertPath string,
	serverCertPath string,
	serverKeyPath string,
	handler http.HandlerFunc,
) (*httptest.Server, string) {
	caCertPEM, err := ioutil.ReadFile(caCertPath)
	if err != nil {
		t.Fatalf("Could not read ca cert from file")
	}

	serverCert, err := tls.LoadX509KeyPair(serverCertPath, serverKeyPath)
	if err != nil {
		t.Fatalf("Could not load server cert and server key from files: %#v", err)
	}

	certPool := x509.NewCertPool()
	if ok := certPool.AppendCertsFromPEM(caCertPEM); !ok {
		t.Fatalf("Cannot add CA to CAPool")
	}

	server := httptest.NewUnstartedServer(http.HandlerFunc(handler))
	server.TLS = &tls.Config{
		Certificates: []tls.Certificate{
			serverCert,
		},
		RootCAs: certPool,
	}

	// calculate the leaf certificate's fingerprint
	if len(server.TLS.Certificates) < 1 || len(server.TLS.Certificates[0].Certificate) < 1 {
		t.Fatal("Expected server.TLS.Certificates not to be empty")
	}
	x509LeafCert := server.TLS.Certificates[0].Certificate[0]
	var tpString string
	for i, b := range sha1.Sum(x509LeafCert) {
		if i > 0 {
			tpString += ":"
		}
		tpString += fmt.Sprintf("%02X", b)
	}

	return server, tpString
}

func TestWithValidCaCert(t *testing.T) {
	handler, verifyConnectionWasMade := getRequestVerifier(t)

	server, _ := createTestServer(t, "./testdata/ca.pem", "./testdata/server.pem", "./testdata/server.key", handler)
	server.StartTLS()
	u := mustParseURL(t, server.URL)

	connection := &vclib.VSphereConnection{
		Hostname: u.Hostname(),
		Port:     u.Port(),
		CACert:   "./testdata/ca.pem",
	}

	// Ignoring error here, because we only care about the TLS connection
	_, _ = connection.NewClient(context.Background())

	verifyConnectionWasMade()
}

func TestWithVerificationWithWrongThumbprint(t *testing.T) {
	handler, _ := getRequestVerifier(t)

	server, _ := createTestServer(t, "./testdata/ca.pem", "./testdata/server.pem", "./testdata/server.key", handler)
	server.StartTLS()
	u := mustParseURL(t, server.URL)

	connection := &vclib.VSphereConnection{
		Hostname:   u.Hostname(),
		Port:       u.Port(),
		Thumbprint: "obviously wrong",
	}

	_, err := connection.NewClient(context.Background())

	if msg := err.Error(); !strings.Contains(msg, "thumbprint does not match") {
		t.Fatalf("Expected wrong thumbprint error, got '%s'", msg)
	}
}

func TestWithVerificationWithoutCaCertOrThumbprint(t *testing.T) {
	handler, _ := getRequestVerifier(t)

	server, _ := createTestServer(t, "./testdata/ca.pem", "./testdata/server.pem", "./testdata/server.key", handler)
	server.StartTLS()
	u := mustParseURL(t, server.URL)

	connection := &vclib.VSphereConnection{
		Hostname: u.Hostname(),
		Port:     u.Port(),
	}

	_, err := connection.NewClient(context.Background())

	verifyWrappedX509UnkownAuthorityErr(t, err)
}

func TestWithValidThumbprint(t *testing.T) {
	handler, verifyConnectionWasMade := getRequestVerifier(t)

	server, thumbprint :=
		createTestServer(t, "./testdata/ca.pem", "./testdata/server.pem", "./testdata/server.key", handler)
	server.StartTLS()
	u := mustParseURL(t, server.URL)

	connection := &vclib.VSphereConnection{
		Hostname:   u.Hostname(),
		Port:       u.Port(),
		Thumbprint: thumbprint,
	}

	// Ignoring error here, because we only care about the TLS connection
	_, _ = connection.NewClient(context.Background())

	verifyConnectionWasMade()
}

func TestWithInvalidCaCertPath(t *testing.T) {
	connection := &vclib.VSphereConnection{
		Hostname: "should-not-matter",
		Port:     "27015", // doesn't matter, but has to be a valid port
		CACert:   "invalid-path",
	}

	_, err := connection.NewClient(context.Background())
	if _, ok := err.(*os.PathError); !ok {
		t.Fatalf("Expected an os.PathError, got: '%s' (%#v)", err.Error(), err)
	}
}

func TestInvalidCaCert(t *testing.T) {
	connection := &vclib.VSphereConnection{
		Hostname: "should-not-matter",
		Port:     "27015", // doesn't matter, but has to be a valid port
		CACert:   "./testdata/invalid.pem",
	}

	_, err := connection.NewClient(context.Background())

	if msg := err.Error(); !strings.Contains(msg, "invalid certificate") {
		t.Fatalf("Expected invalid certificate error, got '%s'", msg)
	}
}

func verifyWrappedX509UnkownAuthorityErr(t *testing.T, err error) {
	urlErr, ok := err.(*url.Error)
	if !ok {
		t.Fatalf("Expected to receive an url.Error, got '%s' (%#v)", err.Error(), err)
	}
	x509Err, ok := urlErr.Err.(x509.UnknownAuthorityError)
	if !ok {
		t.Fatalf("Expected to receive a wrapped x509.UnknownAuthorityError, got: '%s' (%#v)", urlErr.Error(), urlErr)
	}
	if msg := x509Err.Error(); msg != "x509: certificate signed by unknown authority" {
		t.Fatalf("Expected 'signed by unknown authority' error, got: '%s'", msg)
	}
}

func getRequestVerifier(t *testing.T) (http.HandlerFunc, func()) {
	gotRequest := false

	handler := func(w http.ResponseWriter, r *http.Request) {
		gotRequest = true
	}

	checker := func() {
		if !gotRequest {
			t.Fatalf("Never saw a request, maybe TLS connection could not be established?")
		}
	}

	return handler, checker
}

func mustParseURL(t *testing.T, i string) *url.URL {
	u, err := url.Parse(i)
	if err != nil {
		t.Fatalf("Cannot parse URL: %v", err)
	}
	return u
}
