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
	"net/url"
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

func TestMaskValue(t *testing.T) {
	tcs := []struct {
		key      string
		value    string
		expected string
	}{
		{
			key:      "Authorization",
			value:    "Basic YWxhZGRpbjpvcGVuc2VzYW1l",
			expected: "Basic <masked>",
		},
		{
			key:      "Authorization",
			value:    "basic",
			expected: "basic",
		},
		{
			key:      "Authorization",
			value:    "Basic",
			expected: "Basic",
		},
		{
			key:      "Authorization",
			value:    "Bearer cn389ncoiwuencr",
			expected: "Bearer <masked>",
		},
		{
			key:      "Authorization",
			value:    "Bearer",
			expected: "Bearer",
		},
		{
			key:      "Authorization",
			value:    "bearer",
			expected: "bearer",
		},
		{
			key:      "Authorization",
			value:    "bearer ",
			expected: "bearer",
		},
		{
			key:      "Authorization",
			value:    "Negotiate cn389ncoiwuencr",
			expected: "Negotiate <masked>",
		},
		{
			key:      "ABC",
			value:    "Negotiate cn389ncoiwuencr",
			expected: "Negotiate cn389ncoiwuencr",
		},
		{
			key:      "Authorization",
			value:    "Negotiate",
			expected: "Negotiate",
		},
		{
			key:      "Authorization",
			value:    "Negotiate ",
			expected: "Negotiate",
		},
		{
			key:      "Authorization",
			value:    "negotiate",
			expected: "negotiate",
		},
		{
			key:      "Authorization",
			value:    "abc cn389ncoiwuencr",
			expected: "<masked>",
		},
		{
			key:      "Authorization",
			value:    "",
			expected: "",
		},
	}
	for _, tc := range tcs {
		maskedValue := maskValue(tc.key, tc.value)
		if tc.expected != maskedValue {
			t.Errorf("unexpected value %s, given %s.", maskedValue, tc.value)
		}
	}
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
		{
			name: "escape handling",
			impersonationConfig: ImpersonationConfig{
				UserName: "user",
				Extra: map[string][]string{
					"test.example.com/thing.thing": {"A", "a"},
				},
			},
			expected: map[string][]string{
				ImpersonateUserHeader: {"user"},
				ImpersonateUserExtraHeaderPrefix + `Test.example.com%2fthing.thing`: {"A", "a"},
			},
		},
		{
			name: "double escape handling",
			impersonationConfig: ImpersonationConfig{
				UserName: "user",
				Extra: map[string][]string{
					"test.example.com/thing.thing%20another.thing": {"A", "a"},
				},
			},
			expected: map[string][]string{
				ImpersonateUserHeader: {"user"},
				ImpersonateUserExtraHeaderPrefix + `Test.example.com%2fthing.thing%2520another.thing`: {"A", "a"},
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
		username      string
		groups        []string
		extra         map[string][]string
		expectedExtra map[string][]string
	}{
		"allfields": {
			username: "user",
			groups:   []string{"groupA", "groupB"},
			extra: map[string][]string{
				"one": {"alpha", "bravo"},
				"two": {"charlie", "delta"},
			},
			expectedExtra: map[string][]string{
				"one": {"alpha", "bravo"},
				"two": {"charlie", "delta"},
			},
		},
		"escaped extra": {
			username: "user",
			groups:   []string{"groupA", "groupB"},
			extra: map[string][]string{
				"one":             {"alpha", "bravo"},
				"example.com/two": {"charlie", "delta"},
			},
			expectedExtra: map[string][]string{
				"one":               {"alpha", "bravo"},
				"example.com%2ftwo": {"charlie", "delta"},
			},
		},
		"double escaped extra": {
			username: "user",
			groups:   []string{"groupA", "groupB"},
			extra: map[string][]string{
				"one":                     {"alpha", "bravo"},
				"example.com/two%20three": {"charlie", "delta"},
			},
			expectedExtra: map[string][]string{
				"one":                         {"alpha", "bravo"},
				"example.com%2ftwo%2520three": {"charlie", "delta"},
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
		if e, a := tc.expectedExtra, actualExtra; !reflect.DeepEqual(e, a) {
			t.Errorf("%s expected %v, got %v", n, e, a)
			continue
		}
	}
}

// TestHeaderEscapeRoundTrip tests to see if foo == url.PathUnescape(headerEscape(foo))
// This behavior is important for client -> API server transmission of extra values.
func TestHeaderEscapeRoundTrip(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name string
		key  string
	}{
		{
			name: "alpha",
			key:  "alphabetical",
		},
		{
			name: "alphanumeric",
			key:  "alph4num3r1c",
		},
		{
			name: "percent encoded",
			key:  "percent%20encoded",
		},
		{
			name: "almost percent encoded",
			key:  "almost%zzpercent%xxencoded",
		},
		{
			name: "illegal char & percent encoding",
			key:  "example.com/percent%20encoded",
		},
		{
			name: "weird unicode stuff",
			key:  "example.com/ᛒᚥᛏᛖᚥᚢとロビン",
		},
		{
			name: "header legal chars",
			key:  "abc123!#$+.-_*\\^`~|'",
		},
		{
			name: "legal path, illegal header",
			key:  "@=:",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			escaped := headerKeyEscape(tc.key)
			unescaped, err := url.PathUnescape(escaped)
			if err != nil {
				t.Fatalf("url.PathUnescape(%q) returned error: %v", escaped, err)
			}
			if tc.key != unescaped {
				t.Errorf("url.PathUnescape(headerKeyEscape(%q)) returned %q, wanted %q", tc.key, unescaped, tc.key)
			}
		})
	}
}
