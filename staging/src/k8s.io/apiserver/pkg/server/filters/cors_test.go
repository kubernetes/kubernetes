/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestCORSAllowedOrigins(t *testing.T) {
	tests := []struct {
		name           string
		allowedOrigins []string
		origins        []string
		allowed        bool
	}{
		{
			name:           "allowed origins list is empty",
			allowedOrigins: []string{},
			origins:        []string{"example.com"},
			allowed:        false,
		},
		{
			name:           "origin request header not set",
			allowedOrigins: []string{"example.com"},
			origins:        []string{""},
			allowed:        false,
		},
		{
			name:           "allowed regexp is a match",
			allowedOrigins: []string{"example.com"},
			origins:        []string{"http://example.com", "example.com"},
			allowed:        true,
		},
		{
			name:           "allowed regexp is not a match",
			allowedOrigins: []string{"example.com"},
			origins:        []string{"http://not-allowed.com", "not-allowed.com"},
			allowed:        false,
		},
		{
			name:           "allowed list with multiple regex",
			allowedOrigins: []string{"not-matching.com", "example.com"},
			origins:        []string{"http://example.com", "example.com"},
			allowed:        true,
		},
		{
			name:           "wildcard matching",
			allowedOrigins: []string{".*"},
			origins:        []string{"http://example.com", "example.com"},
			allowed:        true,
		},
	}

	for _, test := range tests {
		for _, origin := range test.origins {
			name := fmt.Sprintf("%s/origin/%s", test.name, origin)
			t.Run(name, func(t *testing.T) {
				var handlerInvoked int
				handler := WithCORS(
					http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
						handlerInvoked++
					}),
					test.allowedOrigins, nil, nil, nil, "true",
				)
				var response *http.Response
				func() {
					server := httptest.NewServer(handler)
					defer server.Close()

					request, err := http.NewRequest("GET", server.URL+"/version", nil)
					if err != nil {
						t.Errorf("unexpected error: %v", err)
					}
					request.Header.Set("Origin", origin)
					client := http.Client{}
					response, err = client.Do(request)
					if err != nil {
						t.Errorf("unexpected error: %v", err)
					}
				}()
				if handlerInvoked != 1 {
					t.Errorf("Expected the handler to be invoked once, but got: %d", handlerInvoked)
				}

				if test.allowed {
					if !reflect.DeepEqual(origin, response.Header.Get("Access-Control-Allow-Origin")) {
						t.Errorf("Expected %#v, Got %#v", origin, response.Header.Get("Access-Control-Allow-Origin"))
					}

					if response.Header.Get("Access-Control-Allow-Credentials") == "" {
						t.Errorf("Expected Access-Control-Allow-Credentials header to be set")
					}

					if response.Header.Get("Access-Control-Allow-Headers") == "" {
						t.Errorf("Expected Access-Control-Allow-Headers header to be set")
					}

					if response.Header.Get("Access-Control-Allow-Methods") == "" {
						t.Errorf("Expected Access-Control-Allow-Methods header to be set")
					}

					if response.Header.Get("Access-Control-Expose-Headers") != "Date" {
						t.Errorf("Expected Date in Access-Control-Expose-Headers header")
					}
				} else {
					if response.Header.Get("Access-Control-Allow-Origin") != "" {
						t.Errorf("Expected Access-Control-Allow-Origin header to not be set")
					}

					if response.Header.Get("Access-Control-Allow-Credentials") != "" {
						t.Errorf("Expected Access-Control-Allow-Credentials header to not be set")
					}

					if response.Header.Get("Access-Control-Allow-Headers") != "" {
						t.Errorf("Expected Access-Control-Allow-Headers header to not be set")
					}

					if response.Header.Get("Access-Control-Allow-Methods") != "" {
						t.Errorf("Expected Access-Control-Allow-Methods header to not be set")
					}

					if response.Header.Get("Access-Control-Expose-Headers") == "Date" {
						t.Errorf("Expected Date in Access-Control-Expose-Headers header")
					}
				}
			})
		}
	}
}

func TestCORSAllowedMethods(t *testing.T) {
	tests := []struct {
		allowedMethods []string
		method         string
		allowed        bool
	}{
		{nil, "POST", true},
		{nil, "GET", true},
		{nil, "OPTIONS", true},
		{nil, "PUT", true},
		{nil, "DELETE", true},
		{nil, "PATCH", true},
		{[]string{"GET", "POST"}, "PATCH", false},
	}

	allowsMethod := func(res *http.Response, method string) bool {
		allowedMethods := strings.Split(res.Header.Get("Access-Control-Allow-Methods"), ",")
		for _, allowedMethod := range allowedMethods {
			if strings.TrimSpace(allowedMethod) == method {
				return true
			}
		}
		return false
	}

	for _, test := range tests {
		handler := WithCORS(
			http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
			[]string{".*"}, test.allowedMethods, nil, nil, "true",
		)
		var response *http.Response
		func() {
			server := httptest.NewServer(handler)
			defer server.Close()

			request, err := http.NewRequest(test.method, server.URL+"/version", nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			request.Header.Set("Origin", "allowed.com")
			client := http.Client{}
			response, err = client.Do(request)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		}()

		methodAllowed := allowsMethod(response, test.method)
		switch {
		case test.allowed && !methodAllowed:
			t.Errorf("Expected %v to be allowed, Got only %#v", test.method, response.Header.Get("Access-Control-Allow-Methods"))
		case !test.allowed && methodAllowed:
			t.Errorf("Unexpected allowed method %v, Expected only %#v", test.method, response.Header.Get("Access-Control-Allow-Methods"))
		}
	}

}

func TestCORSWithMultipleOrigins(t *testing.T) {
	tests := []struct {
		name                string
		allowedOrigins      []string
		origin              func(*http.Request)
		allowOriginExpected string
	}{
		{
			name:           "multiple origins in one Origin header",
			allowedOrigins: []string{"foo.com"},
			origin: func(r *http.Request) {
				r.Header.Set("Origin", "http://foo.com http://bar.com")
			},
			// this fails today since req.Header.Get("Origin") returns
			//  "http://foo.com http://bar.com"
			// and the CORS filter sends
			//  "access-control-allow-origin" = "http://foo.com http://bar.com"
			// and the browser will fail
			allowOriginExpected: "http://foo.com",
		},
		{
			name:           "multiple origins in one Origin header",
			allowedOrigins: []string{"bar.com"},
			origin: func(r *http.Request) {
				r.Header.Set("Origin", "http://foo.com http://bar.com")
				r.Header.Add("Origin", "http://baz.com")
			},
			// this fails today, same as above
			allowOriginExpected: "http://bar.com",
		},
		{
			name:           "multiple Origin headers",
			allowedOrigins: []string{"foo.com"},
			origin: func(r *http.Request) {
				r.Header.Set("Origin", "http://foo.com")
				r.Header.Add("Origin", "http://bar.com")
			},
			// the first matching origin should be returned
			allowOriginExpected: "http://foo.com",
		},
		{
			name:           "multiple Origin headers",
			allowedOrigins: []string{"bar.com"},
			origin: func(r *http.Request) {
				r.Header.Set("Origin", "http://foo.com")
				r.Header.Add("Origin", "http://bar.com")
			},
			// this fails today since the CORS filter uses
			//  req.Header.Get("Origin")
			// which will return http://foo.com
			allowOriginExpected: "http://bar.com",
		},
		{
			name:           "multiple Origin headers",
			allowedOrigins: []string{"bar.com", "foo.com"},
			origin: func(r *http.Request) {
				r.Header.Set("Origin", "http://foo.com")
				r.Header.Add("Origin", "http://bar.com")
			},
			// the first matching origin should be returned
			allowOriginExpected: "http://foo.com",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var firstOriginsGot string
			var allOriginsGot []string
			before := func(handler http.Handler) http.HandlerFunc {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					firstOriginsGot = r.Header.Get("Origin")
					allOriginsGot = r.Header.Values("Origin")

					handler.ServeHTTP(w, r)
				})
			}
			handler := WithCORS(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}),
				test.allowedOrigins, nil, nil, nil, "true")
			handler = before(handler)

			server := httptest.NewUnstartedServer(handler)
			server.EnableHTTP2 = true
			server.StartTLS()
			defer server.Close()

			request, err := http.NewRequest("GET", server.URL+"/version", nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			test.origin(request)

			client := server.Client()
			response, err := client.Do(request)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			t.Logf(`Request - Header.Get("Origin") returned: %q Header.Values("Origin") returned: %#v`, firstOriginsGot, allOriginsGot)

			allowOriginGot := response.Header.Values("Access-Control-Allow-Origin")
			expected := []string{test.allowOriginExpected}
			if !reflect.DeepEqual(expected, allowOriginGot) {
				t.Errorf("expected Access-Control-Allow-Origin: %v, but got: %v", expected, allowOriginGot)
			}
		})
	}
}

func TestCompileRegex(t *testing.T) {
	uncompiledRegexes := []string{"endsWithMe$", "^startingWithMe"}
	regexes, err := compileRegexps(uncompiledRegexes)

	if err != nil {
		t.Errorf("Failed to compile legal regexes: '%v': %v", uncompiledRegexes, err)
	}
	if len(regexes) != len(uncompiledRegexes) {
		t.Errorf("Wrong number of regexes returned: '%v': %v", uncompiledRegexes, regexes)
	}

	if !regexes[0].MatchString("Something that endsWithMe") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[0], regexes[0])
	}
	if regexes[0].MatchString("Something that doesn't endsWithMe.") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[0], regexes[0])
	}
	if !regexes[1].MatchString("startingWithMe is very important") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[1], regexes[1])
	}
	if regexes[1].MatchString("not startingWithMe should fail") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[1], regexes[1])
	}
}
