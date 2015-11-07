/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"compress/gzip"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"golang.org/x/net/websocket"
	"k8s.io/kubernetes/pkg/api/rest"
)

func TestProxy(t *testing.T) {
	table := []struct {
		method          string
		path            string
		reqBody         string
		respBody        string
		respContentType string
		reqNamespace    string
	}{
		{"GET", "/some/dir", "", "answer", "text/css", "default"},
		{"GET", "/some/dir", "", "<html><head></head><body>answer</body></html>", "text/html", "default"},
		{"POST", "/some/other/dir", "question", "answer", "text/css", "default"},
		{"PUT", "/some/dir/id", "different question", "answer", "text/css", "default"},
		{"DELETE", "/some/dir/id", "", "ok", "text/css", "default"},
		{"GET", "/some/dir/id", "", "answer", "text/css", "other"},
		{"GET", "/trailing/slash/", "", "answer", "text/css", "default"},
		{"GET", "/", "", "answer", "text/css", "default"},
	}

	for _, item := range table {
		proxyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			gotBody, err := ioutil.ReadAll(req.Body)
			if err != nil {
				t.Errorf("%v - unexpected error %v", item.method, err)
			}
			if e, a := item.reqBody, string(gotBody); e != a {
				t.Errorf("%v - expected %v, got %v", item.method, e, a)
			}
			if e, a := item.path, req.URL.Path; e != a {
				t.Errorf("%v - expected %v, got %v", item.method, e, a)
			}
			w.Header().Set("Content-Type", item.respContentType)
			var out io.Writer = w
			if strings.Contains(req.Header.Get("Accept-Encoding"), "gzip") {
				// The proxier can ask for gzip'd data; we need to provide it with that
				// in order to test our processing of that data.
				w.Header().Set("Content-Encoding", "gzip")
				gzw := gzip.NewWriter(w)
				out = gzw
				defer gzw.Close()
			}
			fmt.Fprint(out, item.respBody)
		}))
		defer proxyServer.Close()

		serverURL, _ := url.Parse(proxyServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			expectedResourceNamespace: item.reqNamespace,
		}

		namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})
		namespaceServer := httptest.NewServer(namespaceHandler)
		defer namespaceServer.Close()

		// test each supported URL pattern for finding the redirection resource in the proxy in a particular namespace
		serverPatterns := []struct {
			server           *httptest.Server
			proxyTestPattern string
		}{
			{namespaceServer, "/api/version2/proxy/namespaces/" + item.reqNamespace + "/foo/id" + item.path},
		}

		for _, serverPattern := range serverPatterns {
			server := serverPattern.server
			proxyTestPattern := serverPattern.proxyTestPattern
			req, err := http.NewRequest(
				item.method,
				server.URL+proxyTestPattern,
				strings.NewReader(item.reqBody),
			)
			if err != nil {
				t.Errorf("%v - unexpected error %v", item.method, err)
				continue
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Errorf("%v - unexpected error %v", item.method, err)
				continue
			}
			gotResp, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Errorf("%v - unexpected error %v", item.method, err)
			}
			resp.Body.Close()
			if e, a := item.respBody, string(gotResp); e != a {
				t.Errorf("%v - expected %v, got %v. url: %#v", item.method, e, a, req.URL)
			}
		}
	}
}

func TestProxyUpgrade(t *testing.T) {

	localhostPool := x509.NewCertPool()
	if !localhostPool.AppendCertsFromPEM(localhostCert) {
		t.Errorf("error setting up localhostCert pool")
	}

	testcases := map[string]struct {
		ServerFunc     func(http.Handler) *httptest.Server
		ProxyTransport http.RoundTripper
	}{
		"http": {
			ServerFunc:     httptest.NewServer,
			ProxyTransport: nil,
		},
		"https (invalid hostname + InsecureSkipVerify)": {
			ServerFunc: func(h http.Handler) *httptest.Server {
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
			},
			ProxyTransport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}},
		},
		"https (valid hostname + RootCAs)": {
			ServerFunc: func(h http.Handler) *httptest.Server {
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
			},
			ProxyTransport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: localhostPool}},
		},
		"https (valid hostname + RootCAs + custom dialer)": {
			ServerFunc: func(h http.Handler) *httptest.Server {
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
			},
			ProxyTransport: &http.Transport{Dial: net.Dial, TLSClientConfig: &tls.Config{RootCAs: localhostPool}},
		},
	}

	for k, tc := range testcases {

		backendServer := tc.ServerFunc(websocket.Handler(func(ws *websocket.Conn) {
			defer ws.Close()
			body := make([]byte, 5)
			ws.Read(body)
			ws.Write([]byte("hello " + string(body)))
		}))
		defer backendServer.Close()

		serverURL, _ := url.Parse(backendServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			resourceLocationTransport: tc.ProxyTransport,
			expectedResourceNamespace: "myns",
		}

		namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})

		server := httptest.NewServer(namespaceHandler)
		defer server.Close()

		ws, err := websocket.Dial("ws://"+server.Listener.Addr().String()+"/api/version2/proxy/namespaces/myns/foo/123", "", "http://127.0.0.1/")
		if err != nil {
			t.Errorf("%s: websocket dial err: %s", k, err)
			continue
		}
		defer ws.Close()

		if _, err := ws.Write([]byte("world")); err != nil {
			t.Errorf("%s: write err: %s", k, err)
			continue
		}

		response := make([]byte, 20)
		n, err := ws.Read(response)
		if err != nil {
			t.Errorf("%s: read err: %s", k, err)
			continue
		}
		if e, a := "hello world", string(response[0:n]); e != a {
			t.Errorf("%s: expected '%#v', got '%#v'", k, e, a)
			continue
		}
	}
}

func TestRedirectOnMissingTrailingSlash(t *testing.T) {
	table := []struct {
		// The requested path
		path string
		// The path requested on the proxy server.
		proxyServerPath string
		// query string
		query string
	}{
		{"/trailing/slash/", "/trailing/slash/", ""},
		{"/", "/", "test1=value1&test2=value2"},
		// "/" should be added at the end.
		{"", "/", "test1=value1&test2=value2"},
		// "/" should not be added at a non-root path.
		{"/some/path", "/some/path", ""},
	}

	for _, item := range table {
		proxyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if req.URL.Path != item.proxyServerPath {
				t.Errorf("Unexpected request on path: %s, expected path: %s, item: %v", req.URL.Path, item.proxyServerPath, item)
			}
			if req.URL.RawQuery != item.query {
				t.Errorf("Unexpected query on url: %s, expected: %s", req.URL.RawQuery, item.query)
			}
		}))
		defer proxyServer.Close()

		serverURL, _ := url.Parse(proxyServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			expectedResourceNamespace: "ns",
		}

		handler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})
		server := httptest.NewServer(handler)
		defer server.Close()

		proxyTestPattern := "/api/version2/proxy/namespaces/ns/foo/id" + item.path
		req, err := http.NewRequest(
			"GET",
			server.URL+proxyTestPattern+"?"+item.query,
			strings.NewReader(""),
		)
		if err != nil {
			t.Errorf("unexpected error %v", err)
			continue
		}
		// Note: We are using a default client here, that follows redirects.
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Errorf("unexpected error %v", err)
			continue
		}
		if resp.StatusCode != http.StatusOK {
			t.Errorf("Unexpected errorCode: %v, expected: 200. Response: %v, item: %v", resp.StatusCode, resp, item)
		}
	}
}

// exampleCert was generated from crypto/tls/generate_cert.go with the following command:
//    go run generate_cert.go  --rsa-bits 512 --host example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var exampleCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBcjCCAR6gAwIBAgIQBOUTYowZaENkZi0faI9DgTALBgkqhkiG9w0BAQswEjEQ
MA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2MDAw
MFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQCZ
xfR3sgeHBraGFfF/24tTn4PRVAHOf2UOOxSQRs+aYjNqimFqf/SRIblQgeXdBJDR
gVK5F1Js2zwlehw0bHxRAgMBAAGjUDBOMA4GA1UdDwEB/wQEAwIApDATBgNVHSUE
DDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MBYGA1UdEQQPMA2CC2V4YW1w
bGUuY29tMAsGCSqGSIb3DQEBCwNBAI/mfBB8dm33IpUl+acSyWfL6gX5Wc0FFyVj
dKeesE1XBuPX1My/rzU6Oy/YwX7LOL4FaeNUS6bbL4axSLPKYSs=
-----END CERTIFICATE-----`)

var exampleKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBOgIBAAJBAJnF9HeyB4cGtoYV8X/bi1Ofg9FUAc5/ZQ47FJBGz5piM2qKYWp/
9JEhuVCB5d0EkNGBUrkXUmzbPCV6HDRsfFECAwEAAQJBAJLH9yPuButniACTn5L5
IJQw1mWQt6zBw9eCo41YWkA0866EgjC53aPZaRjXMp0uNJGdIsys2V5rCOOLWN2C
ODECIQDICHsi8QQQ9wpuJy8X5l8MAfxHL+DIqI84wQTeVM91FQIhAMTME8A18/7h
1Ad6drdnxAkuC0tX6Sx0LDozrmen+HFNAiAlcEDrt0RVkIcpOrg7tuhPLQf0oudl
Zvb3Xlj069awSQIgcT15E/43w2+RASifzVNhQ2MCTr1sSA8lL+xzK+REmnUCIBhQ
j4139pf8Re1J50zBxS/JlQfgDQi9sO9pYeiHIxNs
-----END RSA PRIVATE KEY-----`)

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//     go run generate_cert.go  --rsa-bits 512 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIBdzCCASOgAwIBAgIBADALBgkqhkiG9w0BAQUwEjEQMA4GA1UEChMHQWNtZSBD
bzAeFw03MDAxMDEwMDAwMDBaFw00OTEyMzEyMzU5NTlaMBIxEDAOBgNVBAoTB0Fj
bWUgQ28wWjALBgkqhkiG9w0BAQEDSwAwSAJBAN55NcYKZeInyTuhcCwFMhDHCmwa
IUSdtXdcbItRB/yfXGBhiex00IaLXQnSU+QZPRZWYqeTEbFSgihqi1PUDy8CAwEA
AaNoMGYwDgYDVR0PAQH/BAQDAgCkMBMGA1UdJQQMMAoGCCsGAQUFBwMBMA8GA1Ud
EwEB/wQFMAMBAf8wLgYDVR0RBCcwJYILZXhhbXBsZS5jb22HBH8AAAGHEAAAAAAA
AAAAAAAAAAAAAAEwCwYJKoZIhvcNAQEFA0EAAoQn/ytgqpiLcZu9XKbCJsJcvkgk
Se6AbGXgSlq+ZCEVo0qIwSgeBqmsJxUu7NCSOwVJLYNEBO2DtIxoYVk+MA==
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIBPAIBAAJBAN55NcYKZeInyTuhcCwFMhDHCmwaIUSdtXdcbItRB/yfXGBhiex0
0IaLXQnSU+QZPRZWYqeTEbFSgihqi1PUDy8CAwEAAQJBAQdUx66rfh8sYsgfdcvV
NoafYpnEcB5s4m/vSVe6SU7dCK6eYec9f9wpT353ljhDUHq3EbmE4foNzJngh35d
AekCIQDhRQG5Li0Wj8TM4obOnnXUXf1jRv0UkzE9AHWLG5q3AwIhAPzSjpYUDjVW
MCUXgckTpKCuGwbJk7424Nb8bLzf3kllAiA5mUBgjfr/WtFSJdWcPQ4Zt9KTMNKD
EUO0ukpTwEIl6wIhAMbGqZK3zAAFdq8DD2jPx+UJXnh0rnOkZBzDtJ6/iN69AiEA
1Aq8MJgTaYsDQWyU/hDq5YkDJc9e9DSCvUIzqxQWMQE=
-----END RSA PRIVATE KEY-----`)
