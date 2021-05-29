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

				spdyTransport := NewRoundTripper(testCase.clientTLS, redirect, redirect)

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

			spdyTransport := NewRoundTripper(nil, true, true)
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
//    go run generate_cert.go  --rsa-bits 2048 --host example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var exampleCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDADCCAeigAwIBAgIQVHG3Fn9SdWayyLOZKCW1vzANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMCAXDTcwMDEwMTAwMDAwMFoYDzIwODQwMTI5MTYw
MDAwWjASMRAwDgYDVQQKEwdBY21lIENvMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A
MIIBCgKCAQEArTCu9fiIclNgDdWHphewM+JW55dCb5yYGlJgCBvwbOx547M9p+tn
zm9QOhsdZDHDZsG9tqnWxE2Nc1HpIJyOlfYsOoonpEoG/Ep6nnK91ngj0bn/JlNy
+i/bwU4r97MOukvnOIQez9/D9jAJaOX2+b8/d4lRz9BsqiwJyg+ynZ5tVVYj7aMi
vXnd6HOnJmtqutOtr3beucJnkd6XbwRkLUcAYATT+ZihOWRbTuKqhCg6zGkJOoUG
f8sX61JjoilxiURA//ftGVbdTCU3DrmGmardp5NNOHbumMYU8Vhmqgx1Bqxb+9he
7G42uW5YWYK/GqJzgVPjjlB2dOGj9KrEWQIDAQABo1AwTjAOBgNVHQ8BAf8EBAMC
AqQwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDwYDVR0TAQH/BAUwAwEB/zAWBgNVHREE
DzANggtleGFtcGxlLmNvbTANBgkqhkiG9w0BAQsFAAOCAQEAig4AIi9xWs1+pLES
eeGGdSDoclplFpcbXANnsYYFyLf+8pcWgVi2bOmb2gXMbHFkB07MA82wRJAUTaA+
2iNXVQMhPCoA7J6ADUbww9doJX2S9HGyArhiV/MhHtE8txzMn2EKNLdhhk3N9rmV
x/qRbWAY1U2z4BpdrAR87Fe81Nlj7h45csW9K+eS+NgXipiNTIfEShKgCFM8EdxL
1WXg7r9AvYV3TNDPWTjLsm1rQzzZQ7Uvcf6deWiNodZd8MOT/BFLclDPTK6cF2Hr
UU4dq6G4kCwMSxWE4cM3HlZ4u1dyIt47VbkP0rtvkBCXx36y+NXYA5lzntchNFZP
uvEQdw==
-----END CERTIFICATE-----`)

var exampleKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEArTCu9fiIclNgDdWHphewM+JW55dCb5yYGlJgCBvwbOx547M9
p+tnzm9QOhsdZDHDZsG9tqnWxE2Nc1HpIJyOlfYsOoonpEoG/Ep6nnK91ngj0bn/
JlNy+i/bwU4r97MOukvnOIQez9/D9jAJaOX2+b8/d4lRz9BsqiwJyg+ynZ5tVVYj
7aMivXnd6HOnJmtqutOtr3beucJnkd6XbwRkLUcAYATT+ZihOWRbTuKqhCg6zGkJ
OoUGf8sX61JjoilxiURA//ftGVbdTCU3DrmGmardp5NNOHbumMYU8Vhmqgx1Bqxb
+9he7G42uW5YWYK/GqJzgVPjjlB2dOGj9KrEWQIDAQABAoIBAQClt4CiYaaF5ltx
wVDjz6TNcJUBUs3CKE+uWAYFnF5Ii1nyU876Pxj8Aaz9fHZ6Kde0GkwiXY7gFOj1
YHo2tzcELSKS/SEDZcYbYFTGCjq13g1AH74R+SV6WZLn+5m8kPvVrM1ZWap188H5
bmuCkRDqVmIvShkbRW7EwhC35J9fiuW3majC/sjmsxtxyP6geWmu4f5/Ttqahcdb
osPZIgIIPzqAkNtkLTi7+meHYI9wlrGhL7XZTwnJ1Oc/Y67zzmbthLYB5YFSLUew
rXT58jtSjX4gbiQyheBSrWxW08QE4qYg6jJlAdffHhWv72hJW2MCXhuXp8gJs/Do
XLRHGwSBAoGBAMdNtsbe4yae/QeHUPGxNW0ipa0yoTF6i+VYoxvqiRMzDM3+3L8k
dgI1rr4330SivqDahMA/odWtM/9rVwJI2B2QhZLMHA0n9ytH007OO9TghgVB12nN
xosRYBpKdHXyyvV/MUZl7Jux6zKIzRDWOkF95VVYPcAaxJqd1E5/jJ6JAoGBAN51
QrebA1w/jfydeqQTz1sK01sbO4HYj4qGfo/JarVqGEkm1azeBBPPRnHz3jNKnCkM
S4PpqRDased3NIcViXlAgoqPqivZ8mQa/Rb146l7WaTErASHsZ023OGrxsr/Ed6N
P3GrmvxVJjebaFNaQ9sP80dLkpgeas0t2TY8iQNRAoGATOcnx8TpUVW3vNfx29DN
FLdxxkrq9/SZVn3FMlhlXAsuva3B799ZybB9JNjaRdmmRNsMrkHfaFvU3JHGmRMS
kRXa9LHdgRYSwZiNaLMbUyDvlce6HxFPswmZU4u3NGvi9KeHk+pwSgN1BaLTvdNr
1ymE/FF4QlAR3LdZ3JBK6kECgYEA0wW4/CJ31ZIURoW8SNjh4iMqy0nR8SJVR7q9
Y/hU2TKDRyEnoIwaohAFayNCrLUh3W5kVAXa8roB+OgDVAECH5sqOfZ+HorofD19
x8II7ESujLZj1whBXDkm3ovsT7QWZ17lyBZZNvQvBKDPHgKKS8udowv1S4fPGENd
wS07a4ECgYEAwLSbmMIVJme0jFjsp5d1wOGA2Qi2ZwGIAVlsbnJtygrU/hSBfnu8
VfyJSCgg3fPe7kChWKlfcOebVKSb68LKRsz1Lz1KdbY0HOJFp/cT4lKmDAlRY9gq
LB4rdf46lV0mUkvd2/oofIbTrzukjQSnyfLawb/2uJGV1IkTcZcn9CI=
-----END RSA PRIVATE KEY-----`)

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//     go run generate_cert.go  --rsa-bits 2048 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDGTCCAgGgAwIBAgIRALL5AZcefF4kkYV1SEG6YrMwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzCCASIwDQYJKoZIhvcNAQEBBQADggEP
ADCCAQoCggEBALQ/FHcyVwdFHxARbbD2KBtDUT7Eni+8ioNdjtGcmtXqBv45EC1C
JOqqGJTroFGJ6Q9kQIZ9FqH5IJR2fOOJD9kOTueG4Vt1JY1rj1Kbpjefu8XleZ5L
SBwIWVnN/lEsEbuKmj7N2gLt5AH3zMZiBI1mg1u9Z5ZZHYbCiTpBrwsq6cTlvR9g
dyo1YkM5hRESCzsrL0aUByoo0qRMD8ZsgANJwgsiO0/M6idbxDwv1BnGwGmRYvOE
Hxpy3v0Jg7GJYrvnpnifJTs4nw91N5X9pXxR7FFzi/6HTYDWRljvTb0w6XciKYAz
bWZ0+cJr5F7wB7ovlbm7HrQIR7z7EIIu2d8CAwEAAaNoMGYwDgYDVR0PAQH/BAQD
AgKkMBMGA1UdJQQMMAoGCCsGAQUFBwMBMA8GA1UdEwEB/wQFMAMBAf8wLgYDVR0R
BCcwJYILZXhhbXBsZS5jb22HBH8AAAGHEAAAAAAAAAAAAAAAAAAAAAEwDQYJKoZI
hvcNAQELBQADggEBAFPPWopNEJtIA2VFAQcqN6uJK+JVFOnjGRoCrM6Xgzdm0wxY
XCGjsxY5dl+V7KzdGqu858rCaq5osEBqypBpYAnS9C38VyCDA1vPS1PsN8SYv48z
DyBwj+7R2qar0ADBhnhWxvYO9M72lN/wuCqFKYMeFSnJdQLv3AsrrHe9lYqOa36s
8wxSwVTFTYXBzljPEnSaaJMPqFD8JXaZK1ryJPkO5OsCNQNGtatNiWAf3DcmwHAT
MGYMzP0u4nw47aRz9shB8w+taPKHx2BVwE1m/yp3nHVioOjXqA1fwRQVGclCJSH1
D2iq3hWVHRENgjTjANBPICLo9AZ4JfN6PH19mnU=
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAtD8UdzJXB0UfEBFtsPYoG0NRPsSeL7yKg12O0Zya1eoG/jkQ
LUIk6qoYlOugUYnpD2RAhn0WofkglHZ844kP2Q5O54bhW3UljWuPUpumN5+7xeV5
nktIHAhZWc3+USwRu4qaPs3aAu3kAffMxmIEjWaDW71nllkdhsKJOkGvCyrpxOW9
H2B3KjViQzmFERILOysvRpQHKijSpEwPxmyAA0nCCyI7T8zqJ1vEPC/UGcbAaZFi
84QfGnLe/QmDsYliu+emeJ8lOzifD3U3lf2lfFHsUXOL/odNgNZGWO9NvTDpdyIp
gDNtZnT5wmvkXvAHui+VubsetAhHvPsQgi7Z3wIDAQABAoIBAGmw93IxjYCQ0ncc
kSKMJNZfsdtJdaxuNRZ0nNNirhQzR2h403iGaZlEpmdkhzxozsWcto1l+gh+SdFk
bTUK4MUZM8FlgO2dEqkLYh5BcMT7ICMZvSfJ4v21E5eqR68XVUqQKoQbNvQyxFk3
EddeEGdNrkb0GDK8DKlBlzAW5ep4gjG85wSTjR+J+muUv3R0BgLBFSuQnIDM/IMB
LWqsja/QbtB7yppe7jL5u8UCFdZG8BBKT9fcvFIu5PRLO3MO0uOI7LTc8+W1Xm23
uv+j3SY0+v+6POjK0UlJFFi/wkSPTFIfrQO1qFBkTDQHhQ6q/7GnILYYOiGbIRg2
NNuP52ECgYEAzXEoy50wSYh8xfFaBuxbm3ruuG2W49jgop7ZfoFrPWwOQKAZS441
VIwV4+e5IcA6KkuYbtGSdTYqK1SMkgnUyD/VevwAqH5TJoEIGu0pDuKGwVuwqioZ
frCIAV5GllKyUJ55VZNbRr2vY2fCsWbaCSCHETn6C16DNuTCe5C0JBECgYEA4JqY
5GpNbMG8fOt4H7hU0Fbm2yd6SHJcQ3/9iimef7xG6ajxsYrIhg1ft+3IPHMjVI0+
9brwHDnWg4bOOx/VO4VJBt6Dm/F33bndnZRkuIjfSNpLM51P+EnRdaFVHOJHwKqx
uF69kihifCAG7YATgCveeXImzBUSyZUz9UrETu8CgYARNBimdFNG1RcdvEg9rC0/
p9u1tfecvNySwZqU7WF9kz7eSonTueTdX521qAHowaAdSpdJMGODTTXaywm6cPhQ
jIfj9JZZhbqQzt1O4+08Qdvm9TamCUB5S28YLjza+bHU7nBaqixKkDfPqzCyilpX
yVGGL8SwjwmN3zop/sQXAQKBgC0JMsESQ6YcDsRpnrOVjYQc+LtW5iEitTdfsaID
iGGKihmOI7B66IxgoCHMTws39wycKdSyADVYr5e97xpR3rrJlgQHmBIrz+Iow7Q2
LiAGaec8xjl6QK/DdXmFuQBKqyKJ14rljFODP4QuE9WJid94bGqjpf3j99ltznZP
4J8HAoGAJb4eb4lu4UGwifDzqfAPzLGCoi0fE1/hSx34lfuLcc1G+LEu9YDKoOVJ
9suOh0b5K/bfEy9KrVMBBriduvdaERSD8S3pkIQaitIz0B029AbE4FLFf9lKQpP2
KR8NJEkK99Vh/tew6jAMll70xFrE7aF8VLXJVE7w4sQzuvHxl9Q=
-----END RSA PRIVATE KEY-----
`)
