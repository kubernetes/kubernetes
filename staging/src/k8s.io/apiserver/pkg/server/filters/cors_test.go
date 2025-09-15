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
		{
			// CVE-2022-1996: regular expression 'example.com' matches
			// 'example.com.hacker.org' because it does not pin to the start
			// and end of the host.
			// The CVE should not occur in a real kubernetes cluster since we
			// validate the regular expression specified in --cors-allowed-origins
			// to ensure that it pins to the start and end of the host name.
			name:           "regex does not pin, CVE-2022-1996 is not prevented",
			allowedOrigins: []string{"example.com"},
			origins:        []string{"http://example.com.hacker.org", "http://example.com.hacker.org:8080"},
			allowed:        true,
		},
		{
			// with a proper regular expression we can prevent CVE-2022-1996
			name:           "regex pins to start/end of the host name, CVE-2022-1996 is prevented",
			allowedOrigins: []string{`//example.com(:|$)`},
			origins:        []string{"http://example.com.hacker.org", "http://example.com.hacker.org:8080"},
			allowed:        false,
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
