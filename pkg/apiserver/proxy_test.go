/*
Copyright 2014 Google Inc. All rights reserved.

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
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"golang.org/x/net/html"
	"golang.org/x/net/websocket"
)

func parseURLOrDie(inURL string) *url.URL {
	parsed, err := url.Parse(inURL)
	if err != nil {
		panic(err)
	}
	return parsed
}

// fmtHTML parses and re-emits 'in', effectively canonicalizing it.
func fmtHTML(in string) string {
	doc, err := html.Parse(strings.NewReader(in))
	if err != nil {
		panic(err)
	}
	out := &bytes.Buffer{}
	if err := html.Render(out, doc); err != nil {
		panic(err)
	}
	return string(out.Bytes())
}

func TestProxyTransport(t *testing.T) {
	testTransport := &proxyTransport{
		proxyScheme:      "http",
		proxyHost:        "foo.com",
		proxyPathPrepend: "/proxy/minion/minion1:10250",
	}
	testTransport2 := &proxyTransport{
		proxyScheme:      "https",
		proxyHost:        "foo.com",
		proxyPathPrepend: "/proxy/minion/minion1:8080",
	}
	type Item struct {
		input        string
		sourceURL    string
		transport    *proxyTransport
		output       string
		contentType  string
		forwardedURI string
		redirect     string
		redirectWant string
	}

	table := map[string]Item{
		"normal": {
			input:        `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			sourceURL:    "http://myminion.com/logs/log.log",
			transport:    testTransport,
			output:       `<pre><a href="http://foo.com/proxy/minion/minion1:10250/logs/kubelet.log">kubelet.log</a><a href="http://foo.com/proxy/minion/minion1:10250/google.log">google.log</a></pre>`,
			contentType:  "text/html",
			forwardedURI: "/proxy/minion/minion1:10250/logs/log.log",
		},
		"trailing slash": {
			input:        `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log/">google.log</a></pre>`,
			sourceURL:    "http://myminion.com/logs/log.log",
			transport:    testTransport,
			output:       `<pre><a href="http://foo.com/proxy/minion/minion1:10250/logs/kubelet.log">kubelet.log</a><a href="http://foo.com/proxy/minion/minion1:10250/google.log/">google.log</a></pre>`,
			contentType:  "text/html",
			forwardedURI: "/proxy/minion/minion1:10250/logs/log.log",
		},
		"content-type charset": {
			input:        `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			sourceURL:    "http://myminion.com/logs/log.log",
			transport:    testTransport,
			output:       `<pre><a href="http://foo.com/proxy/minion/minion1:10250/logs/kubelet.log">kubelet.log</a><a href="http://foo.com/proxy/minion/minion1:10250/google.log">google.log</a></pre>`,
			contentType:  "text/html; charset=utf-8",
			forwardedURI: "/proxy/minion/minion1:10250/logs/log.log",
		},
		"content-type passthrough": {
			input:        `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			sourceURL:    "http://myminion.com/logs/log.log",
			transport:    testTransport,
			output:       `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			contentType:  "text/plain",
			forwardedURI: "/proxy/minion/minion1:10250/logs/log.log",
		},
		"subdir": {
			input:        `<a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a>`,
			sourceURL:    "http://myminion.com/whatever/apt/somelog.log",
			transport:    testTransport2,
			output:       `<a href="https://foo.com/proxy/minion/minion1:8080/whatever/apt/kubelet.log">kubelet.log</a><a href="https://foo.com/proxy/minion/minion1:8080/google.log">google.log</a>`,
			contentType:  "text/html",
			forwardedURI: "/proxy/minion/minion1:8080/whatever/apt/somelog.log",
		},
		"image": {
			input:        `<pre><img src="kubernetes.jpg"/></pre>`,
			sourceURL:    "http://myminion.com/",
			transport:    testTransport,
			output:       `<pre><img src="http://foo.com/proxy/minion/minion1:10250/kubernetes.jpg"/></pre>`,
			contentType:  "text/html",
			forwardedURI: "/proxy/minion/minion1:10250/",
		},
		"abs": {
			input:        `<script src="http://google.com/kubernetes.js"/>`,
			sourceURL:    "http://myminion.com/any/path/",
			transport:    testTransport,
			output:       `<script src="http://google.com/kubernetes.js"/>`,
			contentType:  "text/html",
			forwardedURI: "/proxy/minion/minion1:10250/any/path/",
		},
		"abs but same host": {
			input:        `<script src="http://myminion.com/kubernetes.js"/>`,
			sourceURL:    "http://myminion.com/any/path/",
			transport:    testTransport,
			output:       `<script src="http://foo.com/proxy/minion/minion1:10250/kubernetes.js"/>`,
			contentType:  "text/html",
			forwardedURI: "/proxy/minion/minion1:10250/any/path/",
		},
		"redirect rel": {
			sourceURL:    "http://myminion.com/redirect",
			transport:    testTransport,
			redirect:     "/redirected/target/",
			redirectWant: "http://foo.com/proxy/minion/minion1:10250/redirected/target/",
			forwardedURI: "/proxy/minion/minion1:10250/redirect",
		},
		"redirect abs same host": {
			sourceURL:    "http://myminion.com/redirect",
			transport:    testTransport,
			redirect:     "http://myminion.com/redirected/target/",
			redirectWant: "http://foo.com/proxy/minion/minion1:10250/redirected/target/",
			forwardedURI: "/proxy/minion/minion1:10250/redirect",
		},
		"redirect abs other host": {
			sourceURL:    "http://myminion.com/redirect",
			transport:    testTransport,
			redirect:     "http://example.com/redirected/target/",
			redirectWant: "http://example.com/redirected/target/",
			forwardedURI: "/proxy/minion/minion1:10250/redirect",
		},
	}

	testItem := func(name string, item *Item) {
		// Canonicalize the html so we can diff.
		item.input = fmtHTML(item.input)
		item.output = fmtHTML(item.output)

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Check request headers.
			if got, want := r.Header.Get("X-Forwarded-Uri"), item.forwardedURI; got != want {
				t.Errorf("%v: X-Forwarded-Uri = %q, want %q", name, got, want)
			}
			if got, want := r.Header.Get("X-Forwarded-Host"), item.transport.proxyHost; got != want {
				t.Errorf("%v: X-Forwarded-Host = %q, want %q", name, got, want)
			}
			if got, want := r.Header.Get("X-Forwarded-Proto"), item.transport.proxyScheme; got != want {
				t.Errorf("%v: X-Forwarded-Proto = %q, want %q", name, got, want)
			}

			// Send response.
			if item.redirect != "" {
				http.Redirect(w, r, item.redirect, http.StatusMovedPermanently)
				return
			}
			w.Header().Set("Content-Type", item.contentType)
			fmt.Fprint(w, item.input)
		}))
		defer server.Close()

		// Replace source URL with our test server address.
		sourceURL := parseURLOrDie(item.sourceURL)
		serverURL := parseURLOrDie(server.URL)
		item.input = strings.Replace(item.input, sourceURL.Host, serverURL.Host, -1)
		item.redirect = strings.Replace(item.redirect, sourceURL.Host, serverURL.Host, -1)
		sourceURL.Host = serverURL.Host

		req, err := http.NewRequest("GET", sourceURL.String(), nil)
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", name, err)
			return
		}
		resp, err := item.transport.RoundTrip(req)
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", name, err)
			return
		}
		if item.redirect != "" {
			// Check that redirect URLs get rewritten properly.
			if got, want := resp.Header.Get("Location"), item.redirectWant; got != want {
				t.Errorf("%v: Location header = %q, want %q", name, got, want)
			}
			return
		}
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", name, err)
			return
		}
		if e, a := item.output, string(body); e != a {
			t.Errorf("%v: expected %v, but got %v", name, e, a)
		}
	}

	for name, item := range table {
		testItem(name, &item)
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
		legacyNamespaceHandler := handle(map[string]rest.Storage{"foo": simpleStorage})
		legacyNamespaceServer := httptest.NewServer(legacyNamespaceHandler)
		defer legacyNamespaceServer.Close()

		// test each supported URL pattern for finding the redirection resource in the proxy in a particular namespace
		serverPatterns := []struct {
			server           *httptest.Server
			proxyTestPattern string
		}{
			{namespaceServer, "/api/version2/proxy/namespaces/" + item.reqNamespace + "/foo/id" + item.path},
			{legacyNamespaceServer, "/api/version/proxy/foo/id" + item.path + "?namespace=" + item.reqNamespace},
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
	backendServer := httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
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
		expectedResourceNamespace: "myns",
	}

	namespaceHandler := handleNamespaced(map[string]rest.Storage{"foo": simpleStorage})

	server := httptest.NewServer(namespaceHandler)
	defer server.Close()

	ws, err := websocket.Dial("ws://"+server.Listener.Addr().String()+"/api/version2/proxy/namespaces/myns/foo/123", "", "http://127.0.0.1/")
	if err != nil {
		t.Fatalf("websocket dial err: %s", err)
	}
	defer ws.Close()

	if _, err := ws.Write([]byte("world")); err != nil {
		t.Fatalf("write err: %s", err)
	}

	response := make([]byte, 20)
	n, err := ws.Read(response)
	if err != nil {
		t.Fatalf("read err: %s", err)
	}
	if e, a := "hello world", string(response[0:n]); e != a {
		t.Fatalf("expected '%#v', got '%#v'", e, a)
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
