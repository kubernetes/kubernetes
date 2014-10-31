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
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"code.google.com/p/go.net/html"
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

func TestProxyTransport_fixLinks(t *testing.T) {
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
		input     string
		sourceURL string
		transport *proxyTransport
		output    string
	}{
		"normal": {
			input:     `<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`,
			sourceURL: "http://myminion.com/logs/log.log",
			transport: testTransport,
			output:    `<pre><a href="http://foo.com/proxy/minion/minion1:10250/logs/kubelet.log">kubelet.log</a><a href="http://foo.com/proxy/minion/minion1:10250/logs/google.log">google.log</a></pre>`,
		},
		"subdir": {
			input:     `<a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a>`,
			sourceURL: "http://myminion.com/whatever/apt/somelog.log",
			transport: testTransport2,
			output:    `<a href="https://foo.com/proxy/minion/minion1:8080/whatever/apt/kubelet.log">kubelet.log</a><a href="https://foo.com/proxy/minion/minion1:8080/whatever/apt/google.log">google.log</a>`,
		},
		"image": {
			input:     `<pre><img src="kubernetes.jpg"/></pre>`,
			sourceURL: "http://myminion.com/",
			transport: testTransport,
			output:    `<pre><img src="http://foo.com/proxy/minion/minion1:10250/kubernetes.jpg"/></pre>`,
		},
		"abs": {
			input:     `<script src="http://google.com/kubernetes.js"/>`,
			sourceURL: "http://myminion.com/any/path/",
			transport: testTransport,
			output:    `<script src="http://google.com/kubernetes.js"/>`,
		},
		"abs but same host": {
			input:     `<script src="http://myminion.com/kubernetes.js"/>`,
			sourceURL: "http://myminion.com/any/path/",
			transport: testTransport,
			output:    `<script src="http://foo.com/proxy/minion/minion1:10250/kubernetes.js"/>`,
		},
	}

	for name, item := range table {
		// Canonicalize the html so we can diff.
		item.input = fmtHTML(item.input)
		item.output = fmtHTML(item.output)
		req := &http.Request{
			Method: "GET",
			URL:    parseURLOrDie(item.sourceURL),
		}
		resp := &http.Response{
			Status:     "200 OK",
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(strings.NewReader(item.input)),
			Close:      true,
		}
		updatedResp, err := item.transport.fixLinks(req, resp)
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", name, err)
			continue
		}
		body, err := ioutil.ReadAll(updatedResp.Body)
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", name, err)
			continue
		}
		if e, a := item.output, string(body); e != a {
			t.Errorf("%v: expected %v, but got %v", name, e, a)
		}
	}
}

func TestProxy(t *testing.T) {
	table := []struct {
		method       string
		path         string
		reqBody      string
		respBody     string
		reqNamespace string
	}{
		{"GET", "/some/dir", "", "answer", "default"},
		{"POST", "/some/other/dir", "question", "answer", "default"},
		{"PUT", "/some/dir/id", "different question", "answer", "default"},
		{"DELETE", "/some/dir/id", "", "ok", "default"},
		{"GET", "/some/dir/id?namespace=other", "", "answer", "other"},
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
			fmt.Fprint(w, item.respBody)
		}))
		defer proxyServer.Close()

		simpleStorage := &SimpleRESTStorage{
			errors:                    map[string]error{},
			resourceLocation:          proxyServer.URL,
			expectedResourceNamespace: item.reqNamespace,
		}
		handler := Handle(map[string]RESTStorage{
			"foo": simpleStorage,
		}, codec, "/prefix/version", selfLinker)
		server := httptest.NewServer(handler)
		defer server.Close()

		req, err := http.NewRequest(
			item.method,
			server.URL+"/prefix/version/proxy/foo/id"+item.path,
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
