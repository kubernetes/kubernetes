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

	"golang.org/x/net/html"
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
		proxyPathPrepend: "/proxy/minion/minion1:10250/",
	}
	testTransport2 := &proxyTransport{
		proxyScheme:      "https",
		proxyHost:        "foo.com",
		proxyPathPrepend: "/proxy/minion/minion1:8080/",
	}

	table := map[string]struct {
		input       string
		sourceURL   string
		transport   *proxyTransport
		output      string
		contentType string
	}{
		"normal": {
			input:       `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			sourceURL:   "http://myminion.com/logs/log.log",
			transport:   testTransport,
			output:      `<pre><a href="http://foo.com/proxy/minion/minion1:10250/logs/kubelet.log">kubelet.log</a><a href="http://foo.com/proxy/minion/minion1:10250/logs/google.log">google.log</a></pre>`,
			contentType: "text/html",
		},
		"content-type charset": {
			input:       `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			sourceURL:   "http://myminion.com/logs/log.log",
			transport:   testTransport,
			output:      `<pre><a href="http://foo.com/proxy/minion/minion1:10250/logs/kubelet.log">kubelet.log</a><a href="http://foo.com/proxy/minion/minion1:10250/logs/google.log">google.log</a></pre>`,
			contentType: "text/html; charset=utf-8",
		},
		"content-type passthrough": {
			input:       `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			sourceURL:   "http://myminion.com/logs/log.log",
			transport:   testTransport,
			output:      `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			contentType: "text/plain",
		},
		"subdir": {
			input:       `<a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a>`,
			sourceURL:   "http://myminion.com/whatever/apt/somelog.log",
			transport:   testTransport2,
			output:      `<a href="https://foo.com/proxy/minion/minion1:8080/whatever/apt/kubelet.log">kubelet.log</a><a href="https://foo.com/proxy/minion/minion1:8080/whatever/apt/google.log">google.log</a>`,
			contentType: "text/html",
		},
		"image": {
			input:       `<pre><img src="kubernetes.jpg"/></pre>`,
			sourceURL:   "http://myminion.com/",
			transport:   testTransport,
			output:      `<pre><img src="http://foo.com/proxy/minion/minion1:10250/kubernetes.jpg"/></pre>`,
			contentType: "text/html",
		},
		"abs": {
			input:       `<script src="http://google.com/kubernetes.js"/>`,
			sourceURL:   "http://myminion.com/any/path/",
			transport:   testTransport,
			output:      `<script src="http://google.com/kubernetes.js"/>`,
			contentType: "text/html",
		},
		"abs but same host": {
			input:       `<script src="http://myminion.com/kubernetes.js"/>`,
			sourceURL:   "http://myminion.com/any/path/",
			transport:   testTransport,
			output:      `<script src="http://foo.com/proxy/minion/minion1:10250/kubernetes.js"/>`,
			contentType: "text/html",
		},
	}

	for name, item := range table {
		// Canonicalize the html so we can diff.
		item.input = fmtHTML(item.input)
		item.output = fmtHTML(item.output)

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set("Content-Type", item.contentType)
			fmt.Fprint(w, item.input)
		}))
		// Replace source URL with our test server address.
		sourceURL := parseURLOrDie(item.sourceURL)
		serverURL := parseURLOrDie(server.URL)
		item.input = strings.Replace(item.input, sourceURL.Host, serverURL.Host, -1)
		sourceURL.Host = serverURL.Host

		req, err := http.NewRequest("GET", sourceURL.String(), nil)
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", name, err)
			continue
		}
		resp, err := item.transport.RoundTrip(req)
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", name, err)
			continue
		}
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", name, err)
			continue
		}
		if e, a := item.output, string(body); e != a {
			t.Errorf("%v: expected %v, but got %v", name, e, a)
		}
		server.Close()
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

		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          proxyServer.URL,
			expectedResourceNamespace: item.reqNamespace,
		}
		handler := Handle(map[string]RESTStorage{
			"foo": simpleStorage,
		}, codec, "/prefix", "version", selfLinker, admissionControl)
		server := httptest.NewServer(handler)
		defer server.Close()

		// test each supported URL pattern for finding the redirection resource in the proxy in a particular namespace
		proxyTestPatterns := []string{
			"/prefix/version/proxy/foo/id" + item.path + "?namespace=" + item.reqNamespace,
			"/prefix/version/proxy/ns/" + item.reqNamespace + "/foo/id" + item.path,
		}
		for _, proxyTestPattern := range proxyTestPatterns {
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
				t.Errorf("%v - expected %v, got %v", item.method, e, a)
			}
		}
	}
}
