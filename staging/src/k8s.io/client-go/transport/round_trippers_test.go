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

package transport

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"strings"
	"testing"
)

type testRoundTripper struct {
	Request  *http.Request
	Response *http.Response
	Err      error
}

func (rt *testRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.Request = req
	return rt.Response, rt.Err
}

func TestBearerAuthRoundTripper(t *testing.T) {
	rt := &testRoundTripper{}
	req := &http.Request{}
	NewBearerAuthRoundTripper("test", rt).RoundTrip(req)
	if rt.Request == nil {
		t.Fatalf("unexpected nil request: %v", rt)
	}
	if rt.Request == req {
		t.Fatalf("round tripper should have copied request object: %#v", rt.Request)
	}
	if rt.Request.Header.Get("Authorization") != "Bearer test" {
		t.Errorf("unexpected authorization header: %#v", rt.Request)
	}
}

func TestBasicAuthRoundTripper(t *testing.T) {
	for n, tc := range map[string]struct {
		user string
		pass string
	}{
		"basic":   {user: "user", pass: "pass"},
		"no pass": {user: "user"},
	} {
		rt := &testRoundTripper{}
		req := &http.Request{}
		NewBasicAuthRoundTripper(tc.user, tc.pass, rt).RoundTrip(req)
		if rt.Request == nil {
			t.Fatalf("%s: unexpected nil request: %v", n, rt)
		}
		if rt.Request == req {
			t.Fatalf("%s: round tripper should have copied request object: %#v", n, rt.Request)
		}
		if user, pass, found := rt.Request.BasicAuth(); !found || user != tc.user || pass != tc.pass {
			t.Errorf("%s: unexpected authorization header: %#v", n, rt.Request)
		}
	}
}

func TestUserAgentRoundTripper(t *testing.T) {
	rt := &testRoundTripper{}
	req := &http.Request{
		Header: make(http.Header),
	}
	req.Header.Set("User-Agent", "other")
	NewUserAgentRoundTripper("test", rt).RoundTrip(req)
	if rt.Request == nil {
		t.Fatalf("unexpected nil request: %v", rt)
	}
	if rt.Request != req {
		t.Fatalf("round tripper should not have copied request object: %#v", rt.Request)
	}
	if rt.Request.Header.Get("User-Agent") != "other" {
		t.Errorf("unexpected user agent header: %#v", rt.Request)
	}

	req = &http.Request{}
	NewUserAgentRoundTripper("test", rt).RoundTrip(req)
	if rt.Request == nil {
		t.Fatalf("unexpected nil request: %v", rt)
	}
	if rt.Request == req {
		t.Fatalf("round tripper should have copied request object: %#v", rt.Request)
	}
	if rt.Request.Header.Get("User-Agent") != "test" {
		t.Errorf("unexpected user agent header: %#v", rt.Request)
	}
}

func TestImpersonationRoundTripper(t *testing.T) {
	tcs := []struct {
		name                string
		impersonationConfig ImpersonationConfig
		expected            map[string][]string
	}{
		{
			name: "all",
			impersonationConfig: ImpersonationConfig{
				UserName: "user",
				Groups:   []string{"one", "two"},
				Extra: map[string][]string{
					"first":  {"A", "a"},
					"second": {"B", "b"},
				},
			},
			expected: map[string][]string{
				ImpersonateUserHeader:                       {"user"},
				ImpersonateGroupHeader:                      {"one", "two"},
				ImpersonateUserExtraHeaderPrefix + "First":  {"A", "a"},
				ImpersonateUserExtraHeaderPrefix + "Second": {"B", "b"},
			},
		},
	}

	for _, tc := range tcs {
		rt := &testRoundTripper{}
		req := &http.Request{
			Header: make(http.Header),
		}
		NewImpersonatingRoundTripper(tc.impersonationConfig, rt).RoundTrip(req)

		for k, v := range rt.Request.Header {
			expected, ok := tc.expected[k]
			if !ok {
				t.Errorf("%v missing %v=%v", tc.name, k, v)
				continue
			}
			if !reflect.DeepEqual(expected, v) {
				t.Errorf("%v expected %v: %v, got %v", tc.name, k, expected, v)
			}
		}
		for k, v := range tc.expected {
			expected, ok := rt.Request.Header[k]
			if !ok {
				t.Errorf("%v missing %v=%v", tc.name, k, v)
				continue
			}
			if !reflect.DeepEqual(expected, v) {
				t.Errorf("%v expected %v: %v, got %v", tc.name, k, expected, v)
			}
		}
	}
}

func TestAuthProxyRoundTripper(t *testing.T) {
	for n, tc := range map[string]struct {
		username string
		groups   []string
		extra    map[string][]string
	}{
		"allfields": {
			username: "user",
			groups:   []string{"groupA", "groupB"},
			extra: map[string][]string{
				"one": {"alpha", "bravo"},
				"two": {"charlie", "delta"},
			},
		},
	} {
		rt := &testRoundTripper{}
		req := &http.Request{}
		NewAuthProxyRoundTripper(tc.username, tc.groups, tc.extra, rt).RoundTrip(req)
		if rt.Request == nil {
			t.Errorf("%s: unexpected nil request: %v", n, rt)
			continue
		}
		if rt.Request == req {
			t.Errorf("%s: round tripper should have copied request object: %#v", n, rt.Request)
			continue
		}

		actualUsernames, ok := rt.Request.Header["X-Remote-User"]
		if !ok {
			t.Errorf("%s missing value", n)
			continue
		}
		if e, a := []string{tc.username}, actualUsernames; !reflect.DeepEqual(e, a) {
			t.Errorf("%s expected %v, got %v", n, e, a)
			continue
		}
		actualGroups, ok := rt.Request.Header["X-Remote-Group"]
		if !ok {
			t.Errorf("%s missing value", n)
			continue
		}
		if e, a := tc.groups, actualGroups; !reflect.DeepEqual(e, a) {
			t.Errorf("%s expected %v, got %v", n, e, a)
			continue
		}

		actualExtra := map[string][]string{}
		for key, values := range rt.Request.Header {
			if strings.HasPrefix(strings.ToLower(key), strings.ToLower("X-Remote-Extra-")) {
				extraKey := strings.ToLower(key[len("X-Remote-Extra-"):])
				actualExtra[extraKey] = append(actualExtra[key], values...)
			}
		}
		if e, a := tc.extra, actualExtra; !reflect.DeepEqual(e, a) {
			t.Errorf("%s expected %v, got %v", n, e, a)
			continue
		}
	}
}

func TestCacheRoundTripper(t *testing.T) {
	rt := &testRoundTripper{}
	cacheDir, err := ioutil.TempDir("", "cache-rt")
	defer os.RemoveAll(cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewCacheRoundTripper(cacheDir, rt)

	// First call, caches the response
	req := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Host: "localhost"},
	}
	rt.Response = &http.Response{
		Header:     http.Header{"ETag": []string{`"123456"`}},
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("Content"))),
		StatusCode: http.StatusOK,
	}
	resp, err := cache.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	content, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "Content" {
		t.Errorf(`Expected Body to be "Content", got %q`, string(content))
	}

	// Second call, returns cached response
	req = &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Host: "localhost"},
	}
	rt.Response = &http.Response{
		StatusCode: http.StatusNotModified,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("Other Content"))),
	}

	resp, err = cache.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}

	// Read body and make sure we have the initial content
	content, err = ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "Content" {
		t.Errorf("Invalid content read from cache %q", string(content))
	}
}
