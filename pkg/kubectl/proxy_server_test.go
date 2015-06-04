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

package kubectl

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func TestFileServing(t *testing.T) {
	const (
		fname = "test.txt"
		data  = "This is test data"
	)
	dir, err := ioutil.TempDir("", "data")
	if err != nil {
		t.Fatalf("error creating tmp dir: %v", err)
	}
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
	proxy := newProxyServer(target)

	tests := []struct{ method, body string }{
		{"GET", ""},
		{"DELETE", ""},
		{"POST", "test payload"},
		{"PUT", "test payload"},
	}

	const path = "/api/test?fields=ID%3Dfoo&labels=key%3Dvalue"
	for i, tt := range tests {
		r, err := http.NewRequest(tt.method, path, strings.NewReader(tt.body))
		if err != nil {
			t.Errorf("error creating request: %v", err)
			continue
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
	}
}

func TestPathHandling(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, r.URL.Path)
	}))
	defer ts.Close()

	table := []struct {
		prefix     string
		reqPath    string
		expectPath string
	}{
		{"/api/", "/metrics", "404 page not found\n"},
		{"/api/", "/api/metrics", "/api/metrics"},
		{"/api/", "/api/v1/pods/", "/api/v1/pods/"},
		{"/", "/metrics", "/metrics"},
		{"/", "/api/v1/pods/", "/api/v1/pods/"},
		{"/custom/", "/metrics", "404 page not found\n"},
		{"/custom/", "/api/metrics", "404 page not found\n"},
		{"/custom/", "/api/v1/pods/", "404 page not found\n"},
		{"/custom/", "/custom/api/metrics", "/api/metrics"},
		{"/custom/", "/custom/api/v1/pods/", "/api/v1/pods/"},
	}

	cc := &client.Config{
		Host: ts.URL,
	}

	for _, item := range table {
		func() {
			p, err := NewProxyServer("", item.prefix, "/not/used/for/this/test", cc)
			if err != nil {
				t.Fatalf("%#v: %v", item, err)
			}
			pts := httptest.NewServer(p.mux)
			defer pts.Close()

			r, err := http.Get(pts.URL + item.reqPath)
			if err != nil {
				t.Fatalf("%#v: %v", item, err)
			}
			body, err := ioutil.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
				t.Fatalf("%#v: %v", item, err)
			}
			if e, a := item.expectPath, string(body); e != a {
				t.Errorf("%#v: Wanted %q, got %q", item, e, a)
			}
		}()
	}
}
