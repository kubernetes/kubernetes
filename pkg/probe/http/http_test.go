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

package http

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/probe"
)

const FailureCode int = -1

func TestHTTPProbeChecker(t *testing.T) {
	handleReq := func(s int, body string) func(w http.ResponseWriter, r *http.Request) {
		return func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(s)
			w.Write([]byte(body))
		}
	}

	// Echo handler that returns the contents of request headers in the body
	headerEchoHandler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		output := ""
		for k, arr := range r.Header {
			for _, v := range arr {
				output += fmt.Sprintf("%s: %s\n", k, v)
			}
		}
		w.Write([]byte(output))
	}

	redirectHandler := func(s int, bad bool) func(w http.ResponseWriter, r *http.Request) {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/" {
				http.Redirect(w, r, "/new", s)
			} else if bad && r.URL.Path == "/new" {
				w.WriteHeader(http.StatusInternalServerError)
			}
		}
	}

	prober := NewHttpProber()
	testCases := []struct {
		handler    func(w http.ResponseWriter, r *http.Request)
		reqHeaders http.Header
		health     probe.Result
		accBody    string
		notBody    string
	}{
		// The probe will be filled in below.  This is primarily testing that an HTTP GET happens.
		{
			handler: handleReq(http.StatusOK, "ok body"),
			health:  probe.Success,
			accBody: "ok body",
		},
		{
			handler: headerEchoHandler,
			reqHeaders: http.Header{
				"X-Muffins-Or-Cupcakes": {"muffins"},
			},
			health:  probe.Success,
			accBody: "X-Muffins-Or-Cupcakes: muffins",
		},
		{
			handler: headerEchoHandler,
			reqHeaders: http.Header{
				"User-Agent": {"foo/1.0"},
			},
			health:  probe.Success,
			accBody: "User-Agent: foo/1.0",
		},
		{
			handler: headerEchoHandler,
			reqHeaders: http.Header{
				"User-Agent": {""},
			},
			health:  probe.Success,
			notBody: "User-Agent",
		},
		{
			handler:    headerEchoHandler,
			reqHeaders: http.Header{},
			health:     probe.Success,
			accBody:    "User-Agent: kube-probe/",
		},
		{
			// Echo handler that returns the contents of Host in the body
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(200)
				w.Write([]byte(r.Host))
			},
			reqHeaders: http.Header{
				"Host": {"muffins.cupcakes.org"},
			},
			health:  probe.Success,
			accBody: "muffins.cupcakes.org",
		},
		{
			handler: handleReq(FailureCode, "fail body"),
			health:  probe.Failure,
		},
		{
			handler: handleReq(http.StatusInternalServerError, "fail body"),
			health:  probe.Failure,
		},
		{
			handler: func(w http.ResponseWriter, r *http.Request) {
				time.Sleep(3 * time.Second)
			},
			health: probe.Failure,
		},
		{
			handler: redirectHandler(http.StatusMovedPermanently, false), // 301
			health:  probe.Success,
		},
		{
			handler: redirectHandler(http.StatusMovedPermanently, true), // 301
			health:  probe.Failure,
		},
		{
			handler: redirectHandler(http.StatusFound, false), // 302
			health:  probe.Success,
		},
		{
			handler: redirectHandler(http.StatusFound, true), // 302
			health:  probe.Failure,
		},
		{
			handler: redirectHandler(http.StatusTemporaryRedirect, false), // 307
			health:  probe.Success,
		},
		{
			handler: redirectHandler(http.StatusTemporaryRedirect, true), // 307
			health:  probe.Failure,
		},
		{
			handler: redirectHandler(http.StatusPermanentRedirect, false), // 308
			health:  probe.Success,
		},
		{
			handler: redirectHandler(http.StatusPermanentRedirect, true), // 308
			health:  probe.Failure,
		},
	}
	for i, test := range testCases {
		func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				test.handler(w, r)
			}))
			defer server.Close()
			u, err := url.Parse(server.URL)
			if err != nil {
				t.Errorf("case %d: unexpected error: %v", i, err)
			}
			_, port, err := net.SplitHostPort(u.Host)
			if err != nil {
				t.Errorf("case %d: unexpected error: %v", i, err)
			}
			_, err = strconv.Atoi(port)
			if err != nil {
				t.Errorf("case %d: unexpected error: %v", i, err)
			}
			health, output, err := prober.Probe(u, test.reqHeaders, 1*time.Second)
			if test.health == probe.Unknown && err == nil {
				t.Errorf("case %d: expected error", i)
			}
			if test.health != probe.Unknown && err != nil {
				t.Errorf("case %d: unexpected error: %v", i, err)
			}
			if health != test.health {
				t.Errorf("case %d: expected %v, got %v", i, test.health, health)
			}
			if health != probe.Failure && test.health != probe.Failure {
				if !strings.Contains(output, test.accBody) {
					t.Errorf("Expected response body to contain %v, got %v", test.accBody, output)
				}
				if test.notBody != "" && strings.Contains(output, test.notBody) {
					t.Errorf("Expected response not to contain %v, got %v", test.notBody, output)
				}
			}
		}()
	}
}

func TestHTTPSProbeChecker(t *testing.T) {
	httpsServer := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Hello, client")
		if r.TLS != nil {
			if r.TLS.PeerCertificates[0].EmailAddresses[0] != "a@a.com" {
				t.Errorf("client sends a certificate that does not have correct email address. Expect a@a.com, got %s",
					r.TLS.PeerCertificates[0].EmailAddresses[0])
			}
		} else {
			t.Error("request is not from TLS enabled connection")
		}
	}))

	var currentPath string
	currentPath, err := os.Getwd()
	if err != nil {
		glog.Errorf("fail to get path of current directory %v", err)
	}
	certBytes, err := ioutil.ReadFile(filepath.Join(currentPath, "kubelet-ca.crt"))
	if err != nil {
		t.Errorf("Unable to read cert.pem: %v", err)
	}
	certPool := x509.NewCertPool()
	if ok := certPool.AppendCertsFromPEM(certBytes); !ok {
		t.Error("Unable to add certificate to certificate pool")
	}

	tlsConfig := &tls.Config{
		// Reject any TLS certificate that cannot be validated
		ClientAuth: tls.RequireAndVerifyClientCert,
		// Ensure that we only use our "CA" to validate certificates
		ClientCAs: certPool,
		// PFS because we can
		CipherSuites: []uint16{tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384, tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		// Force it server side
		PreferServerCipherSuites: true,
		// TLS 1.2 because we can
		MinVersion: tls.VersionTLS12,
	}
	tlsConfig.BuildNameToCertificate()
	httpsServer.TLS = tlsConfig

	httpsServer.StartTLS()
	defer httpsServer.Close()

	u, err := url.Parse(httpsServer.URL)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	httpsProber := NewHTTPSProber(currentPath)
	health, output, err := httpsProber.Probe(u, http.Header{}, 1*time.Second)
	if health != probe.Success {
		t.Errorf("Unexpected probe response, got %v", health)
	}
	if !strings.Contains(output, "Hello, client") {
		t.Errorf("Unexpected probe response output, got %v", output)
	}
}
