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

package proxy

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/client-go/rest"
)

func TestAccept(t *testing.T) {
	tests := []struct {
		name          string
		acceptPaths   string
		rejectPaths   string
		acceptHosts   string
		rejectMethods string
		path          string
		host          string
		method        string
		expectAccept  bool
	}{

		{
			name:          "test1",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "",
			host:          "127.0.0.1",
			method:        "GET",
			expectAccept:  true,
		},
		{
			name:          "test2",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/pods",
			host:          "127.0.0.1",
			method:        "GET",
			expectAccept:  true,
		},
		{
			name:          "test3",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/pods",
			host:          "localhost",
			method:        "GET",
			expectAccept:  true,
		},
		{
			name:          "test4",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/foo",
			host:          "localhost",
			method:        "GET",
			expectAccept:  true,
		},
		{
			name:          "test5",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/attachfoo",
			host:          "localhost",
			method:        "GET",
			expectAccept:  true,
		},
		{
			name:          "test7",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/execfoo",
			host:          "localhost",
			method:        "GET",
			expectAccept:  true,
		},
		{
			name:          "test8",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/foo/exec",
			host:          "127.0.0.1",
			method:        "GET",
			expectAccept:  false,
		},
		{
			name:          "test9",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/foo/attach",
			host:          "127.0.0.1",
			method:        "GET",
			expectAccept:  false,
		},
		{
			name:          "test10",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/pods",
			host:          "evil.com",
			method:        "GET",
			expectAccept:  false,
		},
		{
			name:          "test11",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/pods",
			host:          "localhost.evil.com",
			method:        "GET",
			expectAccept:  false,
		},
		{
			name:          "test12",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/pods",
			host:          "127a0b0c1",
			method:        "GET",
			expectAccept:  false,
		},
		{
			name:          "test13",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/ui",
			host:          "localhost",
			method:        "GET",
			expectAccept:  true,
		},
		{
			name:          "test14",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/pods",
			host:          "localhost",
			method:        "POST",
			expectAccept:  true,
		},
		{
			name:          "test15",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/somepod",
			host:          "localhost",
			method:        "PUT",
			expectAccept:  true,
		},
		{
			name:          "test16",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/somepod",
			host:          "localhost",
			method:        "PATCH",
			expectAccept:  true,
		},
		{
			name:          "test17",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: "GET",
			path:          "/api/v1/pods",
			host:          "127.0.0.1",
			method:        "GET",
			expectAccept:  false,
		},
		{
			name:          "test18",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: "POST",
			path:          "/api/v1/pods",
			host:          "localhost",
			method:        "POST",
			expectAccept:  false,
		},
		{
			name:          "test19",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: "PUT",
			path:          "/api/v1/namespaces/default/pods/somepod",
			host:          "localhost",
			method:        "PUT",
			expectAccept:  false,
		},
		{
			name:          "test20",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: "PATCH",
			path:          "/api/v1/namespaces/default/pods/somepod",
			host:          "localhost",
			method:        "PATCH",
			expectAccept:  false,
		},
		{
			name:          "test21",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: "POST,PUT,PATCH",
			path:          "/api/v1/namespaces/default/pods/somepod",
			host:          "localhost",
			method:        "PATCH",
			expectAccept:  false,
		},
		{
			name:          "test22",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: "POST,PUT,PATCH",
			path:          "/api/v1/namespaces/default/pods/somepod",
			host:          "localhost",
			method:        "PUT",
			expectAccept:  false,
		},
		{
			name:          "test23",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   DefaultPathRejectRE,
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/somepod/exec",
			host:          "localhost",
			method:        "POST",
			expectAccept:  false,
		},
		{
			name:          "test24",
			acceptPaths:   DefaultPathAcceptRE,
			rejectPaths:   "",
			acceptHosts:   DefaultHostAcceptRE,
			rejectMethods: DefaultMethodRejectRE,
			path:          "/api/v1/namespaces/default/pods/somepod/exec",
			host:          "localhost",
			method:        "POST",
			expectAccept:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := &FilterServer{
				AcceptPaths:   MakeRegexpArrayOrDie(tt.acceptPaths),
				RejectPaths:   MakeRegexpArrayOrDie(tt.rejectPaths),
				AcceptHosts:   MakeRegexpArrayOrDie(tt.acceptHosts),
				RejectMethods: MakeRegexpArrayOrDie(tt.rejectMethods),
			}
			accept := filter.accept(tt.method, tt.path, tt.host)
			if accept != tt.expectAccept {
				t.Errorf("expected: %v, got %v for %#v", tt.expectAccept, accept, tt)
			}
		})
	}
}

func TestRegexpMatch(t *testing.T) {
	tests := []struct {
		name        string
		str         string
		regexps     string
		expectMatch bool
	}{
		{
			name:        "test1",
			str:         "foo",
			regexps:     "bar,.*",
			expectMatch: true,
		},
		{
			name:        "test2",
			str:         "foo",
			regexps:     "bar,fo.*",
			expectMatch: true,
		},
		{
			name:        "test3",
			str:         "bar",
			regexps:     "bar,fo.*",
			expectMatch: true,
		},
		{
			name:        "test4",
			str:         "baz",
			regexps:     "bar,fo.*",
			expectMatch: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			match := matchesRegexp(tt.str, MakeRegexpArrayOrDie(tt.regexps))
			if tt.expectMatch != match {
				t.Errorf("expected: %v, found: %v, for %s and %v", tt.expectMatch, match, tt.str, tt.regexps)
			}
		})
	}
}

func TestFileServing(t *testing.T) {
	const (
		fname = "test.txt"
		data  = "This is test data"
	)
	dir, err := ioutil.TempDir("", "data")
	if err != nil {
		t.Fatalf("error creating tmp dir: %v", err)
	}
	defer os.RemoveAll(dir)
	if err := ioutil.WriteFile(filepath.Join(dir, fname), []byte(data), 0755); err != nil {
		t.Fatalf("error writing tmp file: %v", err)
	}

	const prefix = "/foo/"
	handler := newFileHandler(prefix, dir)
	server := httptest.NewServer(handler)
	defer server.Close()

	url := server.URL + prefix + fname
	res, err := http.Get(url)
	if err != nil {
		t.Fatalf("http.Get(%q) error: %v", url, err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		t.Errorf("res.StatusCode = %d; want %d", res.StatusCode, http.StatusOK)
	}
	b, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("error reading resp body: %v", err)
	}
	if string(b) != data {
		t.Errorf("have %q; want %q", string(b), data)
	}
}

func newProxy(target *url.URL) http.Handler {
	p := proxy.NewUpgradeAwareHandler(target, http.DefaultTransport, false, false, &responder{})
	p.UseRequestLocation = true
	return p
}

func TestAPIRequests(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(w, "%s %s %s", r.Method, r.RequestURI, string(b))
	}))
	defer ts.Close()

	// httptest.NewServer should always generate a valid URL.
	target, _ := url.Parse(ts.URL)
	target.Path = "/"
	proxy := newProxy(target)

	tests := []struct{ name, method, body string }{
		{"test1", "GET", ""},
		{"test2", "DELETE", ""},
		{"test3", "POST", "test payload"},
		{"test4", "PUT", "test payload"},
	}

	const path = "/api/test?fields=ID%3Dfoo&labels=key%3Dvalue"
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := http.NewRequest(tt.method, path, strings.NewReader(tt.body))
			if err != nil {
				t.Errorf("error creating request: %v", err)
				return
			}
			w := httptest.NewRecorder()
			proxy.ServeHTTP(w, r)
			if w.Code != http.StatusOK {
				t.Errorf("%d: proxy.ServeHTTP w.Code = %d; want %d", i, w.Code, http.StatusOK)
			}
			want := strings.Join([]string{tt.method, path, tt.body}, " ")
			if w.Body.String() != want {
				t.Errorf("%d: response body = %q; want %q", i, w.Body.String(), want)
			}
		})
	}
}

func TestPathHandling(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, r.URL.Path)
	}))
	defer ts.Close()

	table := []struct {
		name       string
		prefix     string
		reqPath    string
		expectPath string
		appendPath bool
	}{
		{"test1", "/api/", "/metrics", "404 page not found\n", false},
		{"test2", "/api/", "/api/metrics", "/api/metrics", false},
		{"test3", "/api/", "/api/v1/pods/", "/api/v1/pods/", false},
		{"test4", "/", "/metrics", "/metrics", false},
		{"test5", "/", "/api/v1/pods/", "/api/v1/pods/", false},
		{"test6", "/custom/", "/metrics", "404 page not found\n", false},
		{"test7", "/custom/", "/api/metrics", "404 page not found\n", false},
		{"test8", "/custom/", "/api/v1/pods/", "404 page not found\n", false},
		{"test9", "/custom/", "/custom/api/metrics", "/api/metrics", false},
		{"test10", "/custom/", "/custom/api/v1/pods/", "/api/v1/pods/", false},
		{"test11", "/custom/", "/custom/api/v1/services/", "/api/v1/services/", true},
	}

	cc := &rest.Config{
		Host: ts.URL,
	}

	for _, tt := range table {
		t.Run(tt.name, func(t *testing.T) {
			p, err := NewServer("", tt.prefix, "/not/used/for/this/test", nil, cc, 0, tt.appendPath)
			if err != nil {
				t.Fatalf("%#v: %v", tt, err)
			}
			pts := httptest.NewServer(p.handler)
			defer pts.Close()

			r, err := http.Get(pts.URL + tt.reqPath)
			if err != nil {
				t.Fatalf("%#v: %v", tt, err)
			}
			body, err := ioutil.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
				t.Fatalf("%#v: %v", tt, err)
			}
			if e, a := tt.expectPath, string(body); e != a {
				t.Errorf("%#v: Wanted %q, got %q", tt, e, a)
			}
		})
	}
}

func TestExtractHost(t *testing.T) {
	fixtures := map[string]string{
		"localhost:8085": "localhost",
		"marmalade":      "marmalade",
	}
	for header, expected := range fixtures {
		host := extractHost(header)
		if host != expected {
			t.Fatalf("%s != %s", host, expected)
		}
	}
}
