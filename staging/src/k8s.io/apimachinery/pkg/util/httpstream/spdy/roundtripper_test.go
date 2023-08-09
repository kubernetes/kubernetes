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
	"context"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/armon/go-socks5"
	"github.com/elazarl/goproxy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/httpstream"
)

type serverHandlerConfig struct {
	shouldError      bool
	statusCode       int
	connectionHeader string
	upgradeHeader    string
}

func serverHandler(t *testing.T, config serverHandlerConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		if config.shouldError {
			if e, a := httpstream.HeaderUpgrade, req.Header.Get(httpstream.HeaderConnection); e != a {
				t.Fatalf("expected connection=upgrade header, got '%s", a)
			}

			w.Header().Set(httpstream.HeaderConnection, config.connectionHeader)
			w.Header().Set(httpstream.HeaderUpgrade, config.upgradeHeader)
			w.WriteHeader(config.statusCode)

			return
		}

		streamCh := make(chan httpstream.Stream)

		responseUpgrader := NewResponseUpgrader()
		spdyConn := responseUpgrader.UpgradeResponse(w, req, func(s httpstream.Stream, replySent <-chan struct{}) error {
			streamCh <- s
			return nil
		})
		if spdyConn == nil {
			t.Fatal("unexpected nil spdyConn")
		}
		defer spdyConn.Close()

		stream := <-streamCh
		io.Copy(stream, stream)
	}
}

type serverFunc func(http.Handler) *httptest.Server

func httpsServerInvalidHostname(t *testing.T) serverFunc {
	return func(h http.Handler) *httptest.Server {
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
}

func httpsServerValidHostname(t *testing.T) serverFunc {
	return func(h http.Handler) *httptest.Server {
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
}

func localhostCertPool(t *testing.T) *x509.CertPool {
	localhostPool := x509.NewCertPool()

	if !localhostPool.AppendCertsFromPEM(localhostCert) {
		t.Errorf("error setting up localhostCert pool")
	}
	return localhostPool
}

// be sure to unset environment variable https_proxy (if exported) before testing, otherwise the testing will fail unexpectedly.
func TestRoundTripAndNewConnection(t *testing.T) {
	localhostPool := localhostCertPool(t)

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
			serverFunc:             httpsServerInvalidHostname(t),
			clientTLS:              &tls.Config{InsecureSkipVerify: true},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
		"https (invalid hostname + hostname verification)": {
			serverFunc:             httpsServerInvalidHostname(t),
			clientTLS:              &tls.Config{InsecureSkipVerify: false},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            true,
		},
		"https (valid hostname + RootCAs)": {
			serverFunc:             httpsServerValidHostname(t),
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
			proxyServerFunc:        httpsServerInvalidHostname(t),
			clientTLS:              &tls.Config{InsecureSkipVerify: true},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
		"proxied https with auth (invalid hostname + InsecureSkipVerify) -> http": {
			serverFunc:             httptest.NewServer,
			proxyServerFunc:        httpsServerInvalidHostname(t),
			proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
			clientTLS:              &tls.Config{InsecureSkipVerify: true},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
		"proxied https (invalid hostname + hostname verification) -> http": {
			serverFunc:             httptest.NewServer,
			proxyServerFunc:        httpsServerInvalidHostname(t),
			clientTLS:              &tls.Config{InsecureSkipVerify: false},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            true, // fails because the client doesn't trust the proxy
		},
		"proxied https (valid hostname + RootCAs) -> http": {
			serverFunc:             httptest.NewServer,
			proxyServerFunc:        httpsServerValidHostname(t),
			clientTLS:              &tls.Config{RootCAs: localhostPool},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
		"proxied https with auth (valid hostname + RootCAs) -> http": {
			serverFunc:             httptest.NewServer,
			proxyServerFunc:        httpsServerValidHostname(t),
			proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
			clientTLS:              &tls.Config{RootCAs: localhostPool},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
		"proxied https (invalid hostname + InsecureSkipVerify) -> https (invalid hostname)": {
			serverFunc:             httpsServerInvalidHostname(t),
			proxyServerFunc:        httpsServerInvalidHostname(t),
			clientTLS:              &tls.Config{InsecureSkipVerify: true},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false, // works because the test proxy ignores TLS errors
		},
		"proxied https with auth (invalid hostname + InsecureSkipVerify) -> https (invalid hostname)": {
			serverFunc:             httpsServerInvalidHostname(t),
			proxyServerFunc:        httpsServerInvalidHostname(t),
			proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
			clientTLS:              &tls.Config{InsecureSkipVerify: true},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false, // works because the test proxy ignores TLS errors
		},
		"proxied https (invalid hostname + hostname verification) -> https (invalid hostname)": {
			serverFunc:             httpsServerInvalidHostname(t),
			proxyServerFunc:        httpsServerInvalidHostname(t),
			clientTLS:              &tls.Config{InsecureSkipVerify: false},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            true, // fails because the client doesn't trust the proxy
		},
		"proxied https (valid hostname + RootCAs) -> https (valid hostname + RootCAs)": {
			serverFunc:             httpsServerValidHostname(t),
			proxyServerFunc:        httpsServerValidHostname(t),
			clientTLS:              &tls.Config{RootCAs: localhostPool},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
		"proxied https with auth (valid hostname + RootCAs) -> https (valid hostname + RootCAs)": {
			serverFunc:             httpsServerValidHostname(t),
			proxyServerFunc:        httpsServerValidHostname(t),
			proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
			clientTLS:              &tls.Config{RootCAs: localhostPool},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
		"proxied valid https, proxy auth with chars that percent escape -> valid https": {
			serverFunc:             httpsServerValidHostname(t),
			proxyServerFunc:        httpsServerValidHostname(t),
			proxyAuth:              url.UserPassword("proxy user", "proxypasswd%"),
			clientTLS:              &tls.Config{RootCAs: localhostPool},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
	}

	for k, testCase := range testCases {
		t.Run(k, func(t *testing.T) {
			server := testCase.serverFunc(serverHandler(
				t, serverHandlerConfig{
					shouldError:      testCase.shouldError,
					statusCode:       testCase.serverStatusCode,
					connectionHeader: testCase.serverConnectionHeader,
					upgradeHeader:    testCase.serverUpgradeHeader,
				},
			))
			defer server.Close()

			serverURL, err := url.Parse(server.URL)
			if err != nil {
				t.Fatalf("error creating request: %s", err)
			}
			req, err := http.NewRequest("GET", server.URL, nil)
			if err != nil {
				t.Fatalf("error creating request: %s", err)
			}

			spdyTransport := NewRoundTripper(testCase.clientTLS)

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
				t.Fatalf("shouldError=%t, got %t: %v", e, a, err)
			}
			if testCase.shouldError {
				return
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
				t.Fatalf("expected to write 5 bytes, but actually wrote %d", n)
			}

			b := make([]byte, 5)
			n, err = stream.Read(b)
			if err != nil {
				t.Fatalf("error reading from stream: %s", err)
			}
			if n != 5 {
				t.Fatalf("expected to read 5 bytes, but actually read %d", n)
			}
			if e, a := "hello", string(b[0:n]); e != a {
				t.Fatalf("expected '%s', got '%s'", e, a)
			}

			if testCase.proxyServerFunc != nil {
				if !proxierCalled {
					t.Fatal("expected to use a proxy but proxier in SpdyRoundTripper wasn't called")
				}
				if proxyCalledWithHost != serverURL.Host {
					t.Fatalf("expected to see a call to the proxy for backend %q, got %q", serverURL.Host, proxyCalledWithHost)
				}
			}

			if testCase.proxyAuth != nil {
				expectedUsername := testCase.proxyAuth.Username()
				expectedPassword, _ := testCase.proxyAuth.Password()
				username, password, ok := (&http.Request{Header: http.Header{"Authorization": []string{proxyCalledWithAuthHeader}}}).BasicAuth()
				if !ok {
					t.Fatalf("invalid proxy auth header %s", proxyCalledWithAuthHeader)
				}
				if username != expectedUsername || password != expectedPassword {
					t.Fatalf("expected proxy auth \"%s:%s\", got \"%s:%s\"", expectedUsername, expectedPassword, username, password)
				}
			} else if proxyCalledWithAuth {
				t.Fatalf("proxy authorization unexpected, got %q", proxyCalledWithAuthHeader)
			}
		})
	}
}

type Interceptor struct {
	Authorization       socks5.AuthContext
	proxyCalledWithHost *string
}

func (i *Interceptor) GetAuthContext() (int, map[string]string) {
	return int(i.Authorization.Method), i.Authorization.Payload
}

func (i *Interceptor) Rewrite(ctx context.Context, req *socks5.Request) (context.Context, *socks5.AddrSpec) {
	*i.proxyCalledWithHost = req.DestAddr.Address()
	i.Authorization = socks5.AuthContext(*req.AuthContext)
	return ctx, req.DestAddr
}

// be sure to unset environment variable https_proxy (if exported) before testing, otherwise the testing will fail unexpectedly.
func TestRoundTripSocks5AndNewConnection(t *testing.T) {
	localhostPool := localhostCertPool(t)

	socks5Server := func(creds *socks5.StaticCredentials, interceptor *Interceptor) *socks5.Server {
		var conf *socks5.Config
		if creds != nil {
			authenticator := socks5.UserPassAuthenticator{Credentials: creds}
			conf = &socks5.Config{
				AuthMethods: []socks5.Authenticator{authenticator},
				Rewriter:    interceptor,
			}
		} else {
			conf = &socks5.Config{Rewriter: interceptor}
		}

		ts, err := socks5.New(conf)
		if err != nil {
			t.Errorf("failed to create sock5 server: %v", err)
		}
		return ts
	}

	testCases := map[string]struct {
		clientTLS              *tls.Config
		proxyAuth              *url.Userinfo
		serverConnectionHeader string
		serverFunc             serverFunc
		serverStatusCode       int
		serverUpgradeHeader    string
		shouldError            bool
	}{
		"proxied without auth -> http": {
			serverFunc:             httptest.NewServer,
			serverConnectionHeader: "Upgrade",
			serverStatusCode:       http.StatusSwitchingProtocols,
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            false,
		},
		"proxied with invalid auth -> http": {
			serverFunc:             httptest.NewServer,
			proxyAuth:              url.UserPassword("invalid", "auth"),
			serverConnectionHeader: "Upgrade",
			serverStatusCode:       http.StatusSwitchingProtocols,
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            true,
		},
		"proxied with valid auth -> http": {
			serverFunc:             httptest.NewServer,
			proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
			serverConnectionHeader: "Upgrade",
			serverStatusCode:       http.StatusSwitchingProtocols,
			serverUpgradeHeader:    "SPDY/3.1",
			shouldError:            false,
		},
		"proxied with valid auth -> https (invalid hostname + InsecureSkipVerify)": {
			serverFunc:             httpsServerInvalidHostname(t),
			proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
			clientTLS:              &tls.Config{InsecureSkipVerify: true},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
		"proxied with valid auth -> https (invalid hostname + hostname verification)": {
			serverFunc:             httpsServerInvalidHostname(t),
			proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
			clientTLS:              &tls.Config{InsecureSkipVerify: false},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            true,
		},
		"proxied with valid auth -> https (valid hostname + RootCAs)": {
			serverFunc:             httpsServerValidHostname(t),
			proxyAuth:              url.UserPassword("proxyuser", "proxypasswd"),
			clientTLS:              &tls.Config{RootCAs: localhostPool},
			serverConnectionHeader: "Upgrade",
			serverUpgradeHeader:    "SPDY/3.1",
			serverStatusCode:       http.StatusSwitchingProtocols,
			shouldError:            false,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			server := testCase.serverFunc(serverHandler(
				t, serverHandlerConfig{
					shouldError:      testCase.shouldError,
					statusCode:       testCase.serverStatusCode,
					connectionHeader: testCase.serverConnectionHeader,
					upgradeHeader:    testCase.serverUpgradeHeader,
				},
			))
			defer server.Close()

			req, err := http.NewRequest("GET", server.URL, nil)
			if err != nil {
				t.Fatalf("error creating request: %s", err)
			}

			spdyTransport := NewRoundTripper(testCase.clientTLS)
			var proxierCalled bool
			var proxyCalledWithHost string

			interceptor := &Interceptor{proxyCalledWithHost: &proxyCalledWithHost}

			proxyHandler := socks5Server(nil, interceptor)

			if testCase.proxyAuth != nil {
				proxyHandler = socks5Server(&socks5.StaticCredentials{
					"proxyuser": "proxypasswd", // Socks5 server static credentials when client authentication is expected
				}, interceptor)
			}

			closed := make(chan struct{})
			isClosed := func() bool {
				select {
				case <-closed:
					return true
				default:
					return false
				}
			}

			l, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatalf("socks5Server: proxy_test: Listen: %v", err)
			}
			defer l.Close()

			go func(shoulderror bool) {
				conn, err := l.Accept()
				if err != nil {
					if isClosed() {
						return
					}

					t.Errorf("error accepting connection: %s", err)
				}

				if err := proxyHandler.ServeConn(conn); err != nil && !shoulderror {
					// If the connection request is closed before the channel is closed
					// the test will fail with a ServeConn error. Since the test only return
					// early if expects shouldError=true, the channel is closed at the end of
					// the test, just before all the deferred connections Close() are executed.
					if isClosed() {
						return
					}

					t.Errorf("ServeConn error: %s", err)
				}
			}(testCase.shouldError)
			spdyTransport.proxier = func(proxierReq *http.Request) (*url.URL, error) {
				proxierCalled = true
				return &url.URL{
					Scheme: "socks5",
					Host:   net.JoinHostPort("127.0.0.1", strconv.Itoa(l.Addr().(*net.TCPAddr).Port)),
					User:   testCase.proxyAuth,
				}, nil
			}

			client := &http.Client{Transport: spdyTransport}

			resp, err := client.Do(req)
			haveErr := err != nil
			if e, a := testCase.shouldError, haveErr; e != a {
				t.Fatalf("shouldError=%t, got %t: %v", e, a, err)
			}
			if testCase.shouldError {
				return
			}

			conn, err := spdyTransport.NewConnection(resp)
			haveErr = err != nil
			if e, a := testCase.shouldError, haveErr; e != a {
				t.Fatalf("shouldError=%t, got %t: %v", e, a, err)
			}
			if testCase.shouldError {
				return
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
				t.Fatalf("expected to write 5 bytes, but actually wrote %d", n)
			}

			b := make([]byte, 5)
			n, err = stream.Read(b)
			if err != nil {
				t.Fatalf("error reading from stream: %s", err)
			}
			if n != 5 {
				t.Fatalf("expected to read 5 bytes, but actually read %d", n)
			}
			if e, a := "hello", string(b[0:n]); e != a {
				t.Fatalf("expected '%s', got '%s'", e, a)
			}

			if !proxierCalled {
				t.Fatal("xpected to use a proxy but proxier in SpdyRoundTripper wasn't called")
			}

			serverURL, err := url.Parse(server.URL)
			if err != nil {
				t.Fatalf("error creating request: %s", err)
			}
			if proxyCalledWithHost != serverURL.Host {
				t.Fatalf("expected to see a call to the proxy for backend %q, got %q", serverURL.Host, proxyCalledWithHost)
			}

			authMethod, authUser := interceptor.GetAuthContext()

			if testCase.proxyAuth != nil {
				expectedSocks5AuthMethod := 2
				expectedSocks5AuthUser := "proxyuser"

				if expectedSocks5AuthMethod != authMethod {
					t.Fatalf("socks5 Proxy authorization unexpected, got %d, expected %d", authMethod, expectedSocks5AuthMethod)
				}

				if expectedSocks5AuthUser != authUser["Username"] {
					t.Fatalf("socks5 Proxy authorization user unexpected, got %q, expected %q", authUser["Username"], expectedSocks5AuthUser)
				}
			} else {
				if authMethod != 0 {
					t.Fatalf("proxy authentication method unexpected, got %d", authMethod)
				}
				if len(authUser) != 0 {
					t.Fatalf("unexpected proxy user: %v", authUser)
				}
			}

			// The channel must be closed before any of the connections are closed
			close(closed)
		})
	}
}

func TestRoundTripPassesContextToDialer(t *testing.T) {
	urls := []string{"http://127.0.0.1:1233/", "https://127.0.0.1:1233/"}
	for _, u := range urls {
		t.Run(u, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
			require.NoError(t, err)
			spdyTransport := NewRoundTripper(&tls.Config{})
			_, err = spdyTransport.Dial(req)
			assert.EqualError(t, err, "dial tcp 127.0.0.1:1233: operation was canceled")
		})
	}
}

// exampleCert was generated from crypto/tls/generate_cert.go with the following command:
//
//	go run generate_cert.go  --rsa-bits 2048 --host example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
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
//
//	go run generate_cert.go  --rsa-bits 2048 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
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
