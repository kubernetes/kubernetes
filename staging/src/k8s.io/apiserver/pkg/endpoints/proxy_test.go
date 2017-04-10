/*
Copyright 2014 The Kubernetes Authors.

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

package endpoints

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
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/registry/rest"
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
		defer downstreamServer.Close()

		// Start the proxy server
		serverURL, _ := url.Parse(downstreamServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			expectedResourceNamespace: "default",
		}
		namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})
		server := newTestServer(namespaceHandler)
		defer server.Close()

		// Dial the proxy server
		conn, err := net.Dial(server.Listener.Addr().Network(), server.Listener.Addr().String())
		if err != nil {
			t.Errorf("%s: unexpected error %v", k, err)
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
			t.Fatalf("%s: unexpected error %v", k, err)
		}
		for header, values := range item.reqHeaders {
			for _, value := range values {
				if _, err := fmt.Fprintf(conn, "%s: %s\r\n", header, value); err != nil {
					t.Fatalf("%s: unexpected error %v", k, err)
				}
			}
		}
		// Header separator
		if _, err := fmt.Fprint(conn, "\r\n"); err != nil {
			t.Fatalf("%s: unexpected error %v", k, err)
		}
		// Body
		if _, err := conn.Write(item.reqBody); err != nil {
			t.Fatalf("%s: unexpected error %v", k, err)
		}

		// Read response
		response, err := ioutil.ReadAll(conn)
		if err != nil {
			t.Errorf("%s: unexpected error %v", k, err)
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
		defer downstreamServer.Close()

		serverURL, _ := url.Parse(downstreamServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			expectedResourceNamespace: item.reqNamespace,
		}

		namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})
		namespaceServer := newTestServer(namespaceHandler)
		defer namespaceServer.Close()

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
		defer backendServer.Close()

		serverURL, _ := url.Parse(backendServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			resourceLocationTransport: tc.ProxyTransport,
			expectedResourceNamespace: "myns",
		}

		namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})

		server := newTestServer(namespaceHandler)
		defer server.Close()

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
		defer downstreamServer.Close()

		serverURL, _ := url.Parse(downstreamServer.URL)
		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          serverURL,
			expectedResourceNamespace: "ns",
		}

		handler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})
		server := newTestServer(handler)
		defer server.Close()

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
