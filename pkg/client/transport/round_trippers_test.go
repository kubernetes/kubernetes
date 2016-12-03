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
	"net/http"
	"reflect"
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
