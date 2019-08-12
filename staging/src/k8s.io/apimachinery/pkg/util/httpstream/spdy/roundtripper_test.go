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

				spdyTransport := NewSpdyRoundTripper(testCase.clientTLS, redirect, redirect)

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
		{9, true},
		{10, false},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("with %d redirects", test.redirects), func(t *testing.T) {
			var redirects int32 = 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if redirects < test.redirects {
					atomic.AddInt32(&redirects, 1)
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

			spdyTransport := NewSpdyRoundTripper(nil, true, true)
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
//    go run generate_cert.go  --rsa-bits 1024 --host example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var exampleCert = []byte(`-----BEGIN CERTIFICATE-----
MIIB+zCCAWSgAwIBAgIQArqTHmCaW6G843kgXgy12DANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMCAXDTcwMDEwMTAwMDAwMFoYDzIwODQwMTI5MTYw
MDAwWjASMRAwDgYDVQQKEwdBY21lIENvMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCB
iQKBgQDSt6+lUEp2Fk8OcLuPz3Z9lrPSDFnfLz30RrA3DaZlVgeNVzyrOsxQuRp+
9wgljadxVEDMY69x8NDJnC0mem7kYaIt+gyEtxAPqo7wrsqT17g9MGBQbthtpFeZ
jEPrL9aqAhY9O8kFN2iWkEf8kU2+MGwsaoK/icQH+eyFQ+/VuQIDAQABo1AwTjAO
BgNVHQ8BAf8EBAMCAqQwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDwYDVR0TAQH/BAUw
AwEB/zAWBgNVHREEDzANggtleGFtcGxlLmNvbTANBgkqhkiG9w0BAQsFAAOBgQAy
fpch5gzwsQucZ1pIAj4qZ3wku3mJiXzUtjHBiTkYpwcCMvH2JxNZWTzGQKSO7eJH
hbmHPOfUbr6UazRiVqJuRJ6oI1iHnTFJxELuIx/mM+YThzdZjlq9Dn8VxkZwMpOI
ru5O2VXdMHW/wpK4kCy+FI+VazpHHyPUSMHFVr0Wjw==
-----END CERTIFICATE-----`)

var exampleKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBANK3r6VQSnYWTw5w
u4/Pdn2Ws9IMWd8vPfRGsDcNpmVWB41XPKs6zFC5Gn73CCWNp3FUQMxjr3Hw0Mmc
LSZ6buRhoi36DIS3EA+qjvCuypPXuD0wYFBu2G2kV5mMQ+sv1qoCFj07yQU3aJaQ
R/yRTb4wbCxqgr+JxAf57IVD79W5AgMBAAECgYEAuOSGUZbnD0C586DFYwWWIdK3
TCqcPTJ1uT7BZj0q8SYQkFuol1KLbpVNA3T9B/6Imu9jwDQEAQVeHllUYLvzSggt
n8C3bytUfKYNQbj0729EapygQ0Xda5ZTZYIjz312mfwdIIs8A5/taBQU5j3ku9Lg
PPinOXZqiYAMNpTHswECQQDoaIXHTdzPGB9KSMc2Htg2xbRptJ5aYAYic1VXiXDO
XB2XzVYiUrQ/+Bs2gyjtoJyfOWjoN1qlDdN4V7ETSnAhAkEA6Bt/GQoPjb3BE/CQ
ZU6c9+VaY2RWoFemiE+rxRt78Av5F+0c5KufYpJNUktd/1NJUsiyNJHkYFnpOU7R
OICSmQJBAOB3443l9DjJcZ9Lv6zUCbyNI31dB/z99a7cejb79ko5yhNOLb0k6BdI
yO/TqnoowF1BE8QFgrUcL31yJQMeyEECQCJM9fJoVzYWJbNhqKUgAfhsb3giut6F
NXoNdA/z6NPnoQ8VHmD4r9wsTLrtol16HGrcd+Fm8f3/K4Upjaew8HkCQEQLBYeI
VR3mybfS4TQE+4jX/PrgOAXGhKjdmtPqqaqk5KAfZUwR86kFXtVFH/TwKGQAwL1T
awwC4qga/9zIa6U=
-----END RSA PRIVATE KEY-----`)

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//     go run generate_cert.go  --rsa-bits 1024 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIICEzCCAXygAwIBAgIQZYovfgbbbhbli5vN0HAfKzANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMCAXDTcwMDEwMTAwMDAwMFoYDzIwODQwMTI5MTYw
MDAwWjASMRAwDgYDVQQKEwdBY21lIENvMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCB
iQKBgQDm4Srs2LoP4+Yotx7M+itbWen7IbdHVPmIfguzzzFEzXJlc/ipeQ8uze15
gWFwBbY2Cvdy1LU3oItO8X1/75Cx/66B+tdhENExe6w5gZPqNPXhf9ei2vJ0jdEu
MedteXu9AqAKJBU23H5HBlaTr2irNCBGc77K2pQt0a9DLSdMCQIDAQABo2gwZjAO
BgNVHQ8BAf8EBAMCAqQwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDwYDVR0TAQH/BAUw
AwEB/zAuBgNVHREEJzAlggtleGFtcGxlLmNvbYcEfwAAAYcQAAAAAAAAAAAAAAAA
AAAAATANBgkqhkiG9w0BAQsFAAOBgQDDG08vFeL8353MsLbRwqUZ6uwa+w1SnMFu
+gjgJhVcPhS7n4H8J8wanAjzikomOZVfUdkz5n2PE6ShQyXeu7LAN63abvWVcfyl
g7RVq3/Pryhah21lyOxVr11EjsCaEeiGO1WuzOEdIOFD9BXJEhg+HRN9gxv/HrRg
fHSFpMgCwA==
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBAObhKuzYug/j5ii3
Hsz6K1tZ6fsht0dU+Yh+C7PPMUTNcmVz+Kl5Dy7N7XmBYXAFtjYK93LUtTegi07x
fX/vkLH/roH612EQ0TF7rDmBk+o09eF/16La8nSN0S4x5215e70CoAokFTbcfkcG
VpOvaKs0IEZzvsralC3Rr0MtJ0wJAgMBAAECgYBN4IG0Jl6MYZkO/sW66l+Zjrin
5vWFcBpDehDEdAzwYkRGCFpF//mpFfkWVRfiy2pszEIvT6RYwSR8WmS0tMAfRkE/
apJ7w9v7ghhIMXKuO/ohLyHi5hgPWy1L4+gje4YB+TsZftcDxEVklIplUv8eC9cU
NBP49S/tKLaLg+baCQJBAOuEFAfglYZKlXZ9d8mSPAOTCEvV7e/RFDI8w8OUdcyE
zSB5kx0lS3DFY5AirmpPswB1lupdxec1B9FWSE/CoU8CQQD69duGx5DM7oSCw8Wo
x5KljuMxs4mbfcXEGS+oP++khEWoa5evW+m3EzrxLVHDgYG+pMdy3UROXIzmHARm
63cnAkBWSHs2L5dYLbb4RBtAo+yMuq9NaUDUnVqy1QQ7gQZvOTAVd7Tn9qPe2tIR
GkOf+zbvMiVqE5TPkeQdU2kGn52NAkEAtzRyTSM1BxX8sIWAr2T6HliAbREXHOcl
T7HfQ6FhLaXOQFRDSKX9qUOlnNkrvmC1udoLLERxkA8qYPYFFKlCswJBANQJf9j5
mhgfW8Z7iyQboufgSUq4UYJPpevEfLWAg6809sWHhzeg8AHOH8rap1Z97PUjeeWf
XbCRvoe8v2wSoo0=
-----END RSA PRIVATE KEY-----`)
