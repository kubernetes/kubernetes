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

package vclib_test

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
)

func createTestServer(t *testing.T, caCertPath, serverCertPath, serverKeyPath string, handler http.HandlerFunc) *httptest.Server {
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

	return server
}

func TestSomething(t *testing.T) {
	caCertPath := "fixtures/ca.pem"
	serverCertPath := "fixtures/server.pem"
	serverKeyPath := "fixtures/server.key"

	gotRequest := false
	handler := func(w http.ResponseWriter, r *http.Request) {
		gotRequest = true
	}

	server := createTestServer(t, caCertPath, serverCertPath, serverKeyPath, handler)
	server.StartTLS()

	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("Cannot parse URL: %v", err)
	}

	connection := &vclib.VSphereConnection{
		Hostname: u.Hostname(),
		Port:     u.Port(),
		CACert:   "fixtures/ca.pem",
	}

	// Ignoring error here, because we only care about the TLS connection
	connection.NewClient(context.Background())

	if !gotRequest {
		t.Fatalf("Never saw a request, TLS connection could not be established")
	}
}

func TestWithInvalidCaCertPath(t *testing.T) {
	connection := &vclib.VSphereConnection{
		Hostname: "should-not-matter",
		Port:     "should-not-matter",
		CACert:   "invalid-path",
	}

	_, err := connection.NewClient(context.Background())

	if err != vclib.ErrCaCertNotReadable {
		t.Fatalf("should have occoured")
	}
}

func TestInvalidCaCert(t *testing.T) {
	connection := &vclib.VSphereConnection{
		Hostname: "should-not-matter",
		Port:     "should-not-matter",
		CACert:   "fixtures/invalid.pem",
	}

	_, err := connection.NewClient(context.Background())

	if err != vclib.ErrCaCertInvalid {
		t.Fatalf("should have occoured")
	}
}

func TestUnsupportedTransport(t *testing.T) {
	notHttpTransport := new(fakeTransport)

	connection := &vclib.VSphereConnection{
		Hostname: "should-not-matter",
		Port:     "should-not-matter",
		CACert:   "fixtures/ca.pem",
	}

	err := connection.ConfigureTransportWithCA(notHttpTransport)
	if err != vclib.ErrUnsupportedTransport {
		t.Fatalf("should have occured")
	}
}

type fakeTransport struct{}

func (ft fakeTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, nil
}
