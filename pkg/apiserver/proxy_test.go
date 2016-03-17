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
	"bytes"
	"compress/gzip"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/net/websocket"
	"k8s.io/kubernetes/pkg/api/rest"
	utilnet "k8s.io/kubernetes/pkg/util/net"
)

func TestProxyRequestContentLengthAndTransferEncoding(t *testing.T) {
	chunk := func(data []byte) []byte {
		out := &bytes.Buffer{}
		chunker := httputil.NewChunkedWriter(out)
		for _, b := range data {
			if _, err := chunker.Write([]byte{b}); err != nil {
				panic(err)
			}
		}
		chunker.Close()
		out.Write([]byte("\r\n"))
		return out.Bytes()
	}

	zip := func(data []byte) []byte {
		out := &bytes.Buffer{}
		zipper := gzip.NewWriter(out)
		if _, err := zipper.Write(data); err != nil {
			panic(err)
		}
		zipper.Close()
		return out.Bytes()
	}

	sampleData := []byte("abcde")

	table := map[string]struct {
		reqHeaders http.Header
		reqBody    []byte

		expectedHeaders http.Header
		expectedBody    []byte
	}{
		"content-length": {
			reqHeaders: http.Header{
				"Content-Length": []string{"5"},
			},
			reqBody: sampleData,

			expectedHeaders: http.Header{
				"Content-Length":    []string{"5"},
				"Content-Encoding":  nil, // none set
				"Transfer-Encoding": nil, // none set
			},
			expectedBody: sampleData,
		},

		"content-length + identity transfer-encoding": {
			reqHeaders: http.Header{
				"Content-Length":    []string{"5"},
				"Transfer-Encoding": []string{"identity"},
			},
			reqBody: sampleData,

			expectedHeaders: http.Header{
				"Content-Length":    []string{"5"},
				"Content-Encoding":  nil, // none set
				"Transfer-Encoding": nil, // gets removed
			},
			expectedBody: sampleData,
		},

		"content-length + gzip content-encoding": {
			reqHeaders: http.Header{
				"Content-Length":   []string{strconv.Itoa(len(zip(sampleData)))},
				"Content-Encoding": []string{"gzip"},
			},
			reqBody: zip(sampleData),

			expectedHeaders: http.Header{
				"Content-Length":    []string{strconv.Itoa(len(zip(sampleData)))},
				"Content-Encoding":  []string{"gzip"},
				"Transfer-Encoding": nil, // none set
			},
			expectedBody: zip(sampleData),
		},

		"chunked transfer-encoding": {
			reqHeaders: http.Header{
				"Transfer-Encoding": []string{"chunked"},
			},
			reqBody: chunk(sampleData),

			expectedHeaders: http.Header{
				"Content-Length":    nil, // none set
				"Content-Encoding":  nil, // none set
				"Transfer-Encoding": nil, // Transfer-Encoding gets removed
			},
			expectedBody: sampleData, // sample data is unchunked
		},

		"chunked transfer-encoding + gzip content-encoding": {
			reqHeaders: http.Header{
				"Content-Encoding":  []string{"gzip"},
				"Transfer-Encoding": []string{"chunked"},
			},
			reqBody: chunk(zip(sampleData)),

			expectedHeaders: http.Header{
				"Content-Length":    nil, // none set
				"Content-Encoding":  []string{"gzip"},
				"Transfer-Encoding": nil, // gets removed
			},
			expectedBody: zip(sampleData), // sample data is unchunked, but content-encoding is preserved
		},

		// "Transfer-Encoding: gzip" is not supported by go
		// See http/transfer.go#fixTransferEncoding (https://golang.org/src/net/http/transfer.go#L427)
		// Once it is supported, this test case should succeed
		//
		// "gzip+chunked transfer-encoding": {
		// 	reqHeaders: http.Header{
		// 		"Transfer-Encoding": []string{"chunked,gzip"},
		// 	},
		// 	reqBody: chunk(zip(sampleData)),
		//
		// 	expectedHeaders: http.Header{
		// 		"Content-Length":    nil, // no content-length headers
		// 		"Transfer-Encoding": nil, // Transfer-Encoding gets removed
		// 	},
		// 	expectedBody: sampleData,
		// },
	}

	successfulResponse := "backend passed tests"
	for k, item := range table {
		// Start the downstream server
		downstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			// Verify headers
			for header, v := range item.expectedHeaders {
				if !reflect.DeepEqual(v, req.Header[header]) {
					t.Errorf("%s: Expected headers for %s to be %v, got %v", k, header, v, req.Header[header])
				}
			}

			// Read body
			body, err := ioutil.ReadAll(req.Body)
			if err != nil {
				t.Errorf("%s: unexpected error %v", k, err)
			}
			req.Body.Close()

			// Verify length
			if req.ContentLength > 0 && req.ContentLength != int64(len(body)) {
				t.Errorf("%s: ContentLength was %d, len(data) was %d", k, req.ContentLength, len(body))
			}

			// Verify content
			if !bytes.Equal(item.expectedBody, body) {
				t.Errorf("%s: Expected %q, got %q", k, string(item.expectedBody), string(body))
			}

			// Write successful response
			w.Write([]byte(successfulResponse))
		}))
		// TODO: Uncomment when fix #19254
		// defer downstreamServer.Close()

		// Start the proxy server
		serverURL, _ := url.Parse(downstreamServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			expectedResourceNamespace: "default",
		}
		namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})
		server := httptest.NewServer(namespaceHandler)
		// TODO: Uncomment when fix #19254
		// defer server.Close()

		// Dial the proxy server
		conn, err := net.Dial(server.Listener.Addr().Network(), server.Listener.Addr().String())
		if err != nil {
			t.Errorf("%s: unexpected error %v", err)
			continue
		}
		defer conn.Close()

		// Add standard http 1.1 headers
		if item.reqHeaders == nil {
			item.reqHeaders = http.Header{}
		}
		item.reqHeaders.Add("Connection", "close")
		item.reqHeaders.Add("Host", server.Listener.Addr().String())

		// We directly write to the connection to bypass the Go library's manipulation of the Request.Header.
		// Write the request headers
		post := fmt.Sprintf("POST /%s/%s/%s/proxy/namespaces/default/foo/id/some/dir HTTP/1.1\r\n", prefix, newGroupVersion.Group, newGroupVersion.Version)
		if _, err := fmt.Fprint(conn, post); err != nil {
			t.Fatalf("%s: unexpected error %v", err)
		}
		for header, values := range item.reqHeaders {
			for _, value := range values {
				if _, err := fmt.Fprintf(conn, "%s: %s\r\n", header, value); err != nil {
					t.Fatalf("%s: unexpected error %v", err)
				}
			}
		}
		// Header separator
		if _, err := fmt.Fprint(conn, "\r\n"); err != nil {
			t.Fatalf("%s: unexpected error %v", err)
		}
		// Body
		if _, err := conn.Write(item.reqBody); err != nil {
			t.Fatalf("%s: unexpected error %v", err)
		}

		// Read response
		response, err := ioutil.ReadAll(conn)
		if err != nil {
			t.Errorf("%s: unexpected error %v", err)
			continue
		}
		if !strings.HasSuffix(string(response), successfulResponse) {
			t.Errorf("%s: Did not get successful response: %s", k, string(response))
			continue
		}
	}
}

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
		downstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
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
		// TODO: Uncomment when fix #19254
		// defer downstreamServer.Close()

		serverURL, _ := url.Parse(downstreamServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			expectedResourceNamespace: item.reqNamespace,
		}

		namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})
		namespaceServer := httptest.NewServer(namespaceHandler)
		// TODO: Uncomment when fix #19254
		// defer namespaceServer.Close()

		// test each supported URL pattern for finding the redirection resource in the proxy in a particular namespace
		serverPatterns := []struct {
			server           *httptest.Server
			proxyTestPattern string
		}{
			{namespaceServer, "/" + prefix + "/" + newGroupVersion.Group + "/" + newGroupVersion.Version + "/proxy/namespaces/" + item.reqNamespace + "/foo/id" + item.path},
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
			ProxyTransport: utilnet.SetTransportDefaults(&http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}),
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
			ProxyTransport: utilnet.SetTransportDefaults(&http.Transport{TLSClientConfig: &tls.Config{RootCAs: localhostPool}}),
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
			ProxyTransport: utilnet.SetTransportDefaults(&http.Transport{Dial: net.Dial, TLSClientConfig: &tls.Config{RootCAs: localhostPool}}),
		},
	}

	for k, tc := range testcases {

		backendServer := tc.ServerFunc(websocket.Handler(func(ws *websocket.Conn) {
			defer ws.Close()
			body := make([]byte, 5)
			ws.Read(body)
			ws.Write([]byte("hello " + string(body)))
		}))
		// TODO: Uncomment when fix #19254
		// defer backendServer.Close()

		serverURL, _ := url.Parse(backendServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			resourceLocationTransport: tc.ProxyTransport,
			expectedResourceNamespace: "myns",
		}

		namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})

		server := httptest.NewServer(namespaceHandler)
		// TODO: Uncomment when fix #19254
		// defer server.Close()

		ws, err := websocket.Dial("ws://"+server.Listener.Addr().String()+"/"+prefix+"/"+newGroupVersion.Group+"/"+newGroupVersion.Version+"/proxy/namespaces/myns/foo/123", "", "http://127.0.0.1/")
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
		downstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if req.URL.Path != item.proxyServerPath {
				t.Errorf("Unexpected request on path: %s, expected path: %s, item: %v", req.URL.Path, item.proxyServerPath, item)
			}
			if req.URL.RawQuery != item.query {
				t.Errorf("Unexpected query on url: %s, expected: %s", req.URL.RawQuery, item.query)
			}
		}))
		// TODO: Uncomment when fix #19254
		// defer downstreamServer.Close()

		serverURL, _ := url.Parse(downstreamServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			expectedResourceNamespace: "ns",
		}

		handler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})
		server := httptest.NewServer(handler)
		// TODO: Uncomment when fix #19254
		// defer server.Close()

		proxyTestPattern := "/" + prefix + "/" + newGroupVersion.Group + "/" + newGroupVersion.Version + "/proxy/namespaces/ns/foo/id" + item.path
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
