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

package main

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

const (
	testSvcName      = "test-service"
	testCertValidity = 1 * time.Hour
)

func TestCerts(t *testing.T) {
	testCases := []struct {
		cAddr     string
		ips       []net.IP
		hostnames []string
	}{
		{
			// Unfortunately, due to the limitation in the way Go
			// net/http/httptest package sets up the test HTTPS/TLS server,
			// 127.0.0.1 is the only accepted server address. So, we need to
			// generate certificates for this address.
			cAddr:     "127.0.0.1",
			ips:       []net.IP{net.ParseIP("127.0.0.1")},
			hostnames: []string{},
		},
	}

	for i, tc := range testCases {
		cks, err := certs(testSvcName, testCertValidity, tc.cAddr, tc.ips, tc.hostnames)
		if err != nil {
			t.Errorf("[%d] unexpected error generating certs: %v", i, err)
		}

		caPemCK := encodeCertKey(cks["ca"])
		roots := x509.NewCertPool()
		if ok := roots.AppendCertsFromPEM(caPemCK.cert); !ok {
			t.Errorf("[%d] failed to parse root certificate", i)
		}

		serverPemCK := encodeCertKey(cks["server"])
		s, err := fakeTLSServer(roots, serverPemCK)
		if err != nil {
			t.Errorf("[%d] unexpected error starting TLS server: %v", i, err)
		}
		defer s.Close()

		// Setup HTTPS client
		kcfgPemCK := encodeCertKey(cks["kubeconfig"])
		kcfgCert, err := tls.X509KeyPair(kcfgPemCK.cert, kcfgPemCK.key)
		if err != nil {
			t.Errorf("[%d] unexpected error parsing kubeconfig certificates: %v", i, err)
		}
		tlsCfg := &tls.Config{
			Certificates: []tls.Certificate{kcfgCert},
			RootCAs:      roots,
		}
		tlsCfg.BuildNameToCertificate()
		tr := &http.Transport{
			TLSClientConfig: tlsCfg,
		}
		client := &http.Client{Transport: tr}
		resp, err := client.Get(s.URL)
		if err != nil {
			t.Errorf("[%d] unexpected error while sending GET request to the server: %v", i, err)
		}
		defer resp.Body.Close()

		greeting, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("[%d] unexpected error reading server response: %v", i, err)
		}
		if want := "Hello, certificate test!\n"; string(greeting) != want {
			t.Errorf("[%d] want %q, got %q", i, want, greeting)
		}
	}
}

func fakeTLSServer(roots *x509.CertPool, serverCK *certKey) (*httptest.Server, error) {
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Hello, certificate test!")
	}))

	serverCert, err := tls.X509KeyPair(serverCK.cert, serverCK.key)
	if err != nil {
		return nil, err
	}

	s.TLS.Certificates = []tls.Certificate{serverCert}
	s.TLS.RootCAs = roots
	s.TLS.ClientAuth = tls.RequireAndVerifyClientCert
	s.TLS.ClientCAs = roots
	s.TLS.InsecureSkipVerify = false
	return s, nil
}
