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

package spdy

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/elazarl/goproxy"

	"k8s.io/apimachinery/pkg/util/httpstream"
)

// be sure to unset environment variable https_proxy (if exported) before testing, otherwise the testing will fail unexpectedly.
func TestRoundTripAndNewConnection(t *testing.T) {
	for _, redirect := range []bool{false, true} {
		t.Run(fmt.Sprintf("redirect = %t", redirect), func(t *testing.T) {
			localhostPool := x509.NewCertPool()
			if !localhostPool.AppendCertsFromPEM(localhostCert) {
				t.Errorf("error setting up localhostCert pool")
			}

			httpsServerInvalidHostname := func(h http.Handler) *httptest.Server {
				cert, err := tls.X509KeyPair(exampleCert, exampleKey)
				if err != nil {
					t.Errorf("https (invalid hostname): proxy_test: %v", err)
				}
				ts := httptest.NewUnstartedServer(h)
				ts.TLS = &tls.Config{
					Certificates: []tls.Certificate{cert},
				}
				ts.StartTLS()
				return ts
			}

			httpsServerValidHostname := func(h http.Handler) *httptest.Server {
				cert, err := tls.X509KeyPair(localhostCert, localhostKey)
				if err != nil {
					t.Errorf("https (valid hostname): proxy_test: %v", err)
				}
				ts := httptest.NewUnstartedServer(h)
				ts.TLS = &tls.Config{
					Certificates: []tls.Certificate{cert},
				}
				ts.StartTLS()
				return ts
			}

			testCases := map[string]struct {
				serverFunc             func(http.Handler) *httptest.Server
				proxyServerFunc        func(http.Handler) *httptest.Server
				proxyAuth              *url.Userinfo
				clientTLS              *tls.Config
				serverConnectionHeader string
				serverUpgradeHeader    string
				serverStatusCode       int
				shouldError            bool
			}{
				"no headers": {
					serverFunc:             httptest.NewServer,
					serverConnectionHeader: "",
					serverUpgradeHeader:    "",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            true,
				},
				"no upgrade header": {
					serverFunc:             httptest.NewServer,
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            true,
				},
				"no connection header": {
					serverFunc:             httptest.NewServer,
					serverConnectionHeader: "",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            true,
				},
				"no switching protocol status code": {
					serverFunc:             httptest.NewServer,
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusForbidden,
					shouldError:            true,
				},
				"http": {
					serverFunc:             httptest.NewServer,
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"https (invalid hostname + InsecureSkipVerify)": {
					serverFunc:             httpsServerInvalidHostname,
					clientTLS:              &tls.Config{InsecureSkipVerify: true},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"https (invalid hostname + hostname verification)": {
					serverFunc:             httpsServerInvalidHostname,
					clientTLS:              &tls.Config{InsecureSkipVerify: false},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            true,
				},
				"https (valid hostname + RootCAs)": {
					serverFunc:             httpsServerValidHostname,
					clientTLS:              &tls.Config{RootCAs: localhostPool},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"proxied http->http": {
					serverFunc:             httptest.NewServer,
					proxyServerFunc:        httptest.NewServer,
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"proxied https (invalid hostname + InsecureSkipVerify) -> http": {
					serverFunc:             httptest.NewServer,
					proxyServerFunc:        httpsServerInvalidHostname,
					clientTLS:              &tls.Config{InsecureSkipVerify: true},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"proxied https with auth (invalid hostname + InsecureSkipVerify) -> http": {
					serverFunc:             httptest.NewServer,
					proxyServerFunc:        httpsServerInvalidHostname,
					proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
					clientTLS:              &tls.Config{InsecureSkipVerify: true},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"proxied https (invalid hostname + hostname verification) -> http": {
					serverFunc:             httptest.NewServer,
					proxyServerFunc:        httpsServerInvalidHostname,
					clientTLS:              &tls.Config{InsecureSkipVerify: false},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            true, // fails because the client doesn't trust the proxy
				},
				"proxied https (valid hostname + RootCAs) -> http": {
					serverFunc:             httptest.NewServer,
					proxyServerFunc:        httpsServerValidHostname,
					clientTLS:              &tls.Config{RootCAs: localhostPool},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"proxied https with auth (valid hostname + RootCAs) -> http": {
					serverFunc:             httptest.NewServer,
					proxyServerFunc:        httpsServerValidHostname,
					proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
					clientTLS:              &tls.Config{RootCAs: localhostPool},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"proxied https (invalid hostname + InsecureSkipVerify) -> https (invalid hostname)": {
					serverFunc:             httpsServerInvalidHostname,
					proxyServerFunc:        httpsServerInvalidHostname,
					clientTLS:              &tls.Config{InsecureSkipVerify: true},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false, // works because the test proxy ignores TLS errors
				},
				"proxied https with auth (invalid hostname + InsecureSkipVerify) -> https (invalid hostname)": {
					serverFunc:             httpsServerInvalidHostname,
					proxyServerFunc:        httpsServerInvalidHostname,
					proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
					clientTLS:              &tls.Config{InsecureSkipVerify: true},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false, // works because the test proxy ignores TLS errors
				},
				"proxied https (invalid hostname + hostname verification) -> https (invalid hostname)": {
					serverFunc:             httpsServerInvalidHostname,
					proxyServerFunc:        httpsServerInvalidHostname,
					clientTLS:              &tls.Config{InsecureSkipVerify: false},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            true, // fails because the client doesn't trust the proxy
				},
				"proxied https (valid hostname + RootCAs) -> https (valid hostname + RootCAs)": {
					serverFunc:             httpsServerValidHostname,
					proxyServerFunc:        httpsServerValidHostname,
					clientTLS:              &tls.Config{RootCAs: localhostPool},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
				"proxied https with auth (valid hostname + RootCAs) -> https (valid hostname + RootCAs)": {
					serverFunc:             httpsServerValidHostname,
					proxyServerFunc:        httpsServerValidHostname,
					proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
					clientTLS:              &tls.Config{RootCAs: localhostPool},
					serverConnectionHeader: "Upgrade",
					serverUpgradeHeader:    "SPDY/3.1",
					serverStatusCode:       http.StatusSwitchingProtocols,
					shouldError:            false,
				},
			}

			for k, testCase := range testCases {
				server := testCase.serverFunc(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
					if testCase.shouldError {
						if e, a := httpstream.HeaderUpgrade, req.Header.Get(httpstream.HeaderConnection); e != a {
							t.Fatalf("%s: Expected connection=upgrade header, got '%s", k, a)
						}

						w.Header().Set(httpstream.HeaderConnection, testCase.serverConnectionHeader)
						w.Header().Set(httpstream.HeaderUpgrade, testCase.serverUpgradeHeader)
						w.WriteHeader(testCase.serverStatusCode)

						return
					}

					streamCh := make(chan httpstream.Stream)

					responseUpgrader := NewResponseUpgrader()
					spdyConn := responseUpgrader.UpgradeResponse(w, req, func(s httpstream.Stream, replySent <-chan struct{}) error {
						streamCh <- s
						return nil
					})
					if spdyConn == nil {
						t.Fatalf("%s: unexpected nil spdyConn", k)
					}
					defer spdyConn.Close()

					stream := <-streamCh
					io.Copy(stream, stream)
				}))
				defer server.Close()

				serverURL, err := url.Parse(server.URL)
				if err != nil {
					t.Fatalf("%s: Error creating request: %s", k, err)
				}
				req, err := http.NewRequest("GET", server.URL, nil)
				if err != nil {
					t.Fatalf("%s: Error creating request: %s", k, err)
				}

				spdyTransport := NewSpdyRoundTripper(testCase.clientTLS, redirect)

				var proxierCalled bool
				var proxyCalledWithHost string
				var proxyCalledWithAuth bool
				var proxyCalledWithAuthHeader string
				if testCase.proxyServerFunc != nil {
					proxyHandler := goproxy.NewProxyHttpServer()

					proxyHandler.OnRequest().HandleConnectFunc(func(host string, ctx *goproxy.ProxyCtx) (*goproxy.ConnectAction, string) {
						proxyCalledWithHost = host

						proxyAuthHeaderName := "Proxy-Authorization"
						_, proxyCalledWithAuth = ctx.Req.Header[proxyAuthHeaderName]
						proxyCalledWithAuthHeader = ctx.Req.Header.Get(proxyAuthHeaderName)
						return goproxy.OkConnect, host
					})

					proxy := testCase.proxyServerFunc(proxyHandler)

					spdyTransport.proxier = func(proxierReq *http.Request) (*url.URL, error) {
						proxierCalled = true
						proxyURL, err := url.Parse(proxy.URL)
						if err != nil {
							return nil, err
						}
						proxyURL.User = testCase.proxyAuth
						return proxyURL, nil
					}
					defer proxy.Close()
				}

				client := &http.Client{Transport: spdyTransport}

				resp, err := client.Do(req)
				var conn httpstream.Connection
				if err == nil {
					conn, err = spdyTransport.NewConnection(resp)
				}
				haveErr := err != nil
				if e, a := testCase.shouldError, haveErr; e != a {
					t.Fatalf("%s: shouldError=%t, got %t: %v", k, e, a, err)
				}
				if testCase.shouldError {
					continue
				}
				defer conn.Close()

				if resp.StatusCode != http.StatusSwitchingProtocols {
					t.Fatalf("%s: expected http 101 switching protocols, got %d", k, resp.StatusCode)
				}

				stream, err := conn.CreateStream(http.Header{})
				if err != nil {
					t.Fatalf("%s: error creating client stream: %s", k, err)
				}

				n, err := stream.Write([]byte("hello"))
				if err != nil {
					t.Fatalf("%s: error writing to stream: %s", k, err)
				}
				if n != 5 {
					t.Fatalf("%s: Expected to write 5 bytes, but actually wrote %d", k, n)
				}

				b := make([]byte, 5)
				n, err = stream.Read(b)
				if err != nil {
					t.Fatalf("%s: error reading from stream: %s", k, err)
				}
				if n != 5 {
					t.Fatalf("%s: Expected to read 5 bytes, but actually read %d", k, n)
				}
				if e, a := "hello", string(b[0:n]); e != a {
					t.Fatalf("%s: expected '%s', got '%s'", k, e, a)
				}

				if testCase.proxyServerFunc != nil {
					if !proxierCalled {
						t.Fatalf("%s: Expected to use a proxy but proxier in SpdyRoundTripper wasn't called", k)
					}
					if proxyCalledWithHost != serverURL.Host {
						t.Fatalf("%s: Expected to see a call to the proxy for backend %q, got %q", k, serverURL.Host, proxyCalledWithHost)
					}
				}

				var expectedProxyAuth string
				if testCase.proxyAuth != nil {
					encodedCredentials := base64.StdEncoding.EncodeToString([]byte(testCase.proxyAuth.String()))
					expectedProxyAuth = "Basic " + encodedCredentials
				}
				if len(expectedProxyAuth) == 0 && proxyCalledWithAuth {
					t.Fatalf("%s: Proxy authorization unexpected, got %q", k, proxyCalledWithAuthHeader)
				}
				if proxyCalledWithAuthHeader != expectedProxyAuth {
					t.Fatalf("%s: Expected to see a call to the proxy with credentials %q, got %q", k, testCase.proxyAuth, proxyCalledWithAuthHeader)
				}
			}
		})
	}
}

func TestRoundTripRedirects(t *testing.T) {
	tests := []struct {
		redirects     int32
		expectSuccess bool
	}{
		{0, true},
		{1, true},
		{10, true},
		{11, false},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("with %d redirects", test.redirects), func(t *testing.T) {
			var redirects int32 = 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if redirects < test.redirects {
					redirects = atomic.AddInt32(&redirects, 1)
					http.Redirect(w, req, "redirect", http.StatusFound)
					return
				}
				streamCh := make(chan httpstream.Stream)

				responseUpgrader := NewResponseUpgrader()
				spdyConn := responseUpgrader.UpgradeResponse(w, req, func(s httpstream.Stream, replySent <-chan struct{}) error {
					streamCh <- s
					return nil
				})
				if spdyConn == nil {
					t.Fatalf("unexpected nil spdyConn")
				}
				defer spdyConn.Close()

				stream := <-streamCh
				io.Copy(stream, stream)
			}))
			defer server.Close()

			req, err := http.NewRequest("GET", server.URL, nil)
			if err != nil {
				t.Fatalf("Error creating request: %s", err)
			}

			spdyTransport := NewSpdyRoundTripper(nil, true)
			client := &http.Client{Transport: spdyTransport}

			resp, err := client.Do(req)
			if test.expectSuccess {
				if err != nil {
					t.Fatalf("error calling Do: %v", err)
				}
			} else {
				if err == nil {
					t.Fatalf("expecting an error")
				} else if !strings.Contains(err.Error(), "too many redirects") {
					t.Fatalf("expecting too many redirects, got %v", err)
				}
				return
			}

			conn, err := spdyTransport.NewConnection(resp)
			if err != nil {
				t.Fatalf("error calling NewConnection: %v", err)
			}
			defer conn.Close()

			if resp.StatusCode != http.StatusSwitchingProtocols {
				t.Fatalf("expected http 101 switching protocols, got %d", resp.StatusCode)
			}

			stream, err := conn.CreateStream(http.Header{})
			if err != nil {
				t.Fatalf("error creating client stream: %s", err)
			}

			n, err := stream.Write([]byte("hello"))
			if err != nil {
				t.Fatalf("error writing to stream: %s", err)
			}
			if n != 5 {
				t.Fatalf("Expected to write 5 bytes, but actually wrote %d", n)
			}

			b := make([]byte, 5)
			n, err = stream.Read(b)
			if err != nil {
				t.Fatalf("error reading from stream: %s", err)
			}
			if n != 5 {
				t.Fatalf("Expected to read 5 bytes, but actually read %d", n)
			}
			if e, a := "hello", string(b[0:n]); e != a {
				t.Fatalf("expected '%s', got '%s'", e, a)
			}
		})
	}
}

// exampleCert was generated from crypto/tls/generate_cert.go with the following command:
//    go run generate_cert.go  --rsa-bits 512 --host example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var exampleCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBdzCCASGgAwIBAgIRAOVTAdPnfbS5V85mfS90TfIwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgC
QQCoVSqeu8TBvF+70T7Jm4340YQNhds6IxjRoifenYodAO1dnKGrcbF266DJGunh
nIjQH7B12tduhl0fLK4Ezf7/AgMBAAGjUDBOMA4GA1UdDwEB/wQEAwICpDATBgNV
HSUEDDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MBYGA1UdEQQPMA2CC2V4
YW1wbGUuY29tMA0GCSqGSIb3DQEBCwUAA0EAk1kVa5uZ/AzwYDVcS9bpM/czwjjV
xq3VeSCfmNa2uNjbFvodmCRwZOHUvipAMGCUCV6j5vMrJ8eMj8tCQ36W9A==
-----END CERTIFICATE-----`)

var exampleKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBOgIBAAJBAKhVKp67xMG8X7vRPsmbjfjRhA2F2zojGNGiJ96dih0A7V2coatx
sXbroMka6eGciNAfsHXa126GXR8srgTN/v8CAwEAAQJASdzdD7vKsUwMIejGCUb1
fAnLTPfAY3lFCa+CmR89nE22dAoRDv+5RbnBsZ58BazPNJHrsVPRlfXB3OQmSQr0
SQIhANoJhs+xOJE/i8nJv0uAbzKyiD1YkvRkta0GpUOULyAVAiEAxaQus3E/SuqD
P7y5NeJnE7X6XkyC35zrsJRkz7orE8MCIHdDjsI8pjyNDeGqwUCDWE/a6DrmIDwe
emHSqMN2YvChAiEAnxLCM9NWaenOsaIoP+J1rDuvw+4499nJKVqGuVrSCRkCIEqK
4KSchPMc3x8M/uhw9oWTtKFmjA/PPh0FsWCdKrEy
-----END RSA PRIVATE KEY-----`)

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//     go run generate_cert.go  --rsa-bits 512 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBjzCCATmgAwIBAgIRAKpi2WmTcFrVjxrl5n5YDUEwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgC
QQC9fEbRszP3t14Gr4oahV7zFObBI4TfA5i7YnlMXeLinb7MnvT4bkfOJzE6zktn
59zP7UiHs3l4YOuqrjiwM413AgMBAAGjaDBmMA4GA1UdDwEB/wQEAwICpDATBgNV
HSUEDDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MC4GA1UdEQQnMCWCC2V4
YW1wbGUuY29thwR/AAABhxAAAAAAAAAAAAAAAAAAAAABMA0GCSqGSIb3DQEBCwUA
A0EAUsVE6KMnza/ZbodLlyeMzdo7EM/5nb5ywyOxgIOCf0OOLHsPS9ueGLQX9HEG
//yjTXuhNcUugExIjM/AIwAZPQ==
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBAL18RtGzM/e3XgavihqFXvMU5sEjhN8DmLtieUxd4uKdvsye9Phu
R84nMTrOS2fn3M/tSIezeXhg66quOLAzjXcCAwEAAQJBAKcRxH9wuglYLBdI/0OT
BLzfWPZCEw1vZmMR2FF1Fm8nkNOVDPleeVGTWoOEcYYlQbpTmkGSxJ6ya+hqRi6x
goECIQDx3+X49fwpL6B5qpJIJMyZBSCuMhH4B7JevhGGFENi3wIhAMiNJN5Q3UkL
IuSvv03kaPR5XVQ99/UeEetUgGvBcABpAiBJSBzVITIVCGkGc7d+RCf49KTCIklv
bGWObufAR8Ni4QIgWpILjW8dkGg8GOUZ0zaNA6Nvt6TIv2UWGJ4v5PoV98kCIQDx
rIiZs5QbKdycsv9gQJzwQAogC8o04X3Zz3dsoX+h4A==
-----END RSA PRIVATE KEY-----`)
