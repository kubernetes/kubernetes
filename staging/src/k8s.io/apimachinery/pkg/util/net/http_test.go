// +build go1.8

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

package net

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/wait"
)

func TestGetClientIP(t *testing.T) {
	ipString := "10.0.0.1"
	ip := net.ParseIP(ipString)
	invalidIPString := "invalidIPString"
	testCases := []struct {
		Request    http.Request
		ExpectedIP net.IP
	}{
		{
			Request: http.Request{},
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {ipString},
				},
			},
			ExpectedIP: ip,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {invalidIPString},
				},
			},
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {ipString},
				},
			},
			ExpectedIP: ip,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {invalidIPString},
				},
			},
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {invalidIPString + "," + ipString},
				},
			},
			ExpectedIP: ip,
		},
		{
			Request: http.Request{
				// RemoteAddr is in the form host:port
				RemoteAddr: ipString + ":1234",
			},
			ExpectedIP: ip,
		},
		{
			Request: http.Request{
				RemoteAddr: invalidIPString,
			},
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {invalidIPString},
				},
				// RemoteAddr is in the form host:port
				RemoteAddr: ipString,
			},
			ExpectedIP: ip,
		},
	}

	for i, test := range testCases {
		if a, e := GetClientIP(&test.Request), test.ExpectedIP; reflect.DeepEqual(e, a) != true {
			t.Fatalf("test case %d failed. expected: %v, actual: %v", i, e, a)
		}
	}
}

func TestAppendForwardedForHeader(t *testing.T) {
	testCases := []struct {
		addr, forwarded, expected string
	}{
		{"1.2.3.4:8000", "", "1.2.3.4"},
		{"1.2.3.4:8000", "8.8.8.8", "8.8.8.8, 1.2.3.4"},
		{"1.2.3.4:8000", "8.8.8.8, 1.2.3.4", "8.8.8.8, 1.2.3.4, 1.2.3.4"},
		{"1.2.3.4:8000", "foo,bar", "foo,bar, 1.2.3.4"},
	}
	for i, test := range testCases {
		req := &http.Request{
			RemoteAddr: test.addr,
			Header:     make(http.Header),
		}
		if test.forwarded != "" {
			req.Header.Set("X-Forwarded-For", test.forwarded)
		}

		AppendForwardedForHeader(req)
		actual := req.Header.Get("X-Forwarded-For")
		if actual != test.expected {
			t.Errorf("[%d] Expected %q, Got %q", i, test.expected, actual)
		}
	}
}

func TestProxierWithNoProxyCIDR(t *testing.T) {
	testCases := []struct {
		name    string
		noProxy string
		url     string

		expectedDelegated bool
	}{
		{
			name:              "no env",
			url:               "https://192.168.143.1/api",
			expectedDelegated: true,
		},
		{
			name:              "no cidr",
			noProxy:           "192.168.63.1",
			url:               "https://192.168.143.1/api",
			expectedDelegated: true,
		},
		{
			name:              "hostname",
			noProxy:           "192.168.63.0/24,192.168.143.0/24",
			url:               "https://my-hostname/api",
			expectedDelegated: true,
		},
		{
			name:              "match second cidr",
			noProxy:           "192.168.63.0/24,192.168.143.0/24",
			url:               "https://192.168.143.1/api",
			expectedDelegated: false,
		},
		{
			name:              "match second cidr with host:port",
			noProxy:           "192.168.63.0/24,192.168.143.0/24",
			url:               "https://192.168.143.1:8443/api",
			expectedDelegated: false,
		},
		{
			name:              "IPv6 cidr",
			noProxy:           "2001:db8::/48",
			url:               "https://[2001:db8::1]/api",
			expectedDelegated: false,
		},
		{
			name:              "IPv6+port cidr",
			noProxy:           "2001:db8::/48",
			url:               "https://[2001:db8::1]:8443/api",
			expectedDelegated: false,
		},
		{
			name:              "IPv6, not matching cidr",
			noProxy:           "2001:db8::/48",
			url:               "https://[2001:db8:1::1]/api",
			expectedDelegated: true,
		},
		{
			name:              "IPv6+port, not matching cidr",
			noProxy:           "2001:db8::/48",
			url:               "https://[2001:db8:1::1]:8443/api",
			expectedDelegated: true,
		},
	}

	for _, test := range testCases {
		os.Setenv("NO_PROXY", test.noProxy)
		actualDelegated := false
		proxyFunc := NewProxierWithNoProxyCIDR(func(req *http.Request) (*url.URL, error) {
			actualDelegated = true
			return nil, nil
		})

		req, err := http.NewRequest("GET", test.url, nil)
		if err != nil {
			t.Errorf("%s: unexpected err: %v", test.name, err)
			continue
		}
		if _, err := proxyFunc(req); err != nil {
			t.Errorf("%s: unexpected err: %v", test.name, err)
			continue
		}

		if test.expectedDelegated != actualDelegated {
			t.Errorf("%s: expected %v, got %v", test.name, test.expectedDelegated, actualDelegated)
			continue
		}
	}
}

type fakeTLSClientConfigHolder struct {
	called bool
}

func (f *fakeTLSClientConfigHolder) TLSClientConfig() *tls.Config {
	f.called = true
	return nil
}
func (f *fakeTLSClientConfigHolder) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, nil
}

func TestTLSClientConfigHolder(t *testing.T) {
	rt := &fakeTLSClientConfigHolder{}
	TLSClientConfig(rt)

	if !rt.called {
		t.Errorf("didn't find tls config")
	}
}

func TestJoinPreservingTrailingSlash(t *testing.T) {
	tests := []struct {
		a    string
		b    string
		want string
	}{
		// All empty
		{"", "", ""},

		// Empty a
		{"", "/", "/"},
		{"", "foo", "foo"},
		{"", "/foo", "/foo"},
		{"", "/foo/", "/foo/"},

		// Empty b
		{"/", "", "/"},
		{"foo", "", "foo"},
		{"/foo", "", "/foo"},
		{"/foo/", "", "/foo/"},

		// Both populated
		{"/", "/", "/"},
		{"foo", "foo", "foo/foo"},
		{"/foo", "/foo", "/foo/foo"},
		{"/foo/", "/foo/", "/foo/foo/"},
	}
	for _, tt := range tests {
		name := fmt.Sprintf("%q+%q=%q", tt.a, tt.b, tt.want)
		t.Run(name, func(t *testing.T) {
			if got := JoinPreservingTrailingSlash(tt.a, tt.b); got != tt.want {
				t.Errorf("JoinPreservingTrailingSlash() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConnectWithRedirects(t *testing.T) {
	tests := []struct {
		desc              string
		redirects         []string
		method            string // initial request method, empty == GET
		expectError       bool
		expectedRedirects int
		newPort           bool // special case different port test
	}{{
		desc:              "relative redirects allowed",
		redirects:         []string{"/ok"},
		expectedRedirects: 1,
	}, {
		desc:              "redirects to the same host are allowed",
		redirects:         []string{"http://HOST/ok"}, // HOST replaced with server address in test
		expectedRedirects: 1,
	}, {
		desc:              "POST redirects to GET",
		method:            http.MethodPost,
		redirects:         []string{"/ok"},
		expectedRedirects: 1,
	}, {
		desc:              "PUT redirects to GET",
		method:            http.MethodPut,
		redirects:         []string{"/ok"},
		expectedRedirects: 1,
	}, {
		desc:              "DELETE redirects to GET",
		method:            http.MethodDelete,
		redirects:         []string{"/ok"},
		expectedRedirects: 1,
	}, {
		desc:              "9 redirects are allowed",
		redirects:         []string{"/1", "/2", "/3", "/4", "/5", "/6", "/7", "/8", "/9"},
		expectedRedirects: 9,
	}, {
		desc:        "10 redirects are forbidden",
		redirects:   []string{"/1", "/2", "/3", "/4", "/5", "/6", "/7", "/8", "/9", "/10"},
		expectError: true,
	}, {
		desc:        "redirect to different host are prevented",
		redirects:   []string{"http://example.com/foo"},
		expectError: true,
	}, {
		desc:        "multiple redirect to different host forbidden",
		redirects:   []string{"/1", "/2", "/3", "http://example.com/foo"},
		expectError: true,
	}, {
		desc:              "redirect to different port is allowed",
		redirects:         []string{"http://HOST/foo"},
		expectedRedirects: 1,
		newPort:           true,
	}}

	const resultString = "Test output"
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			redirectCount := 0
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				// Verify redirect request.
				if redirectCount > 0 {
					expectedURL, err := url.Parse(test.redirects[redirectCount-1])
					require.NoError(t, err, "test URL error")
					assert.Equal(t, req.URL.Path, expectedURL.Path, "unknown redirect path")
					assert.Equal(t, http.MethodGet, req.Method, "redirects must always be GET")
				}
				if redirectCount < len(test.redirects) {
					http.Redirect(w, req, test.redirects[redirectCount], http.StatusFound)
					redirectCount++
				} else if redirectCount == len(test.redirects) {
					w.Write([]byte(resultString))
				} else {
					t.Errorf("unexpected number of redirects %d to %s", redirectCount, req.URL.String())
				}
			}))
			defer s.Close()

			u, err := url.Parse(s.URL)
			require.NoError(t, err, "Error parsing server URL")
			host := u.Host

			// Special case new-port test with a secondary server.
			if test.newPort {
				s2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
					w.Write([]byte(resultString))
				}))
				defer s2.Close()
				u2, err := url.Parse(s2.URL)
				require.NoError(t, err, "Error parsing secondary server URL")

				// Sanity check: secondary server uses same hostname, different port.
				require.Equal(t, u.Hostname(), u2.Hostname(), "sanity check: same hostname")
				require.NotEqual(t, u.Port(), u2.Port(), "sanity check: different port")

				// Redirect to the secondary server.
				host = u2.Host

			}

			// Update redirect URLs with actual host.
			for i := range test.redirects {
				test.redirects[i] = strings.Replace(test.redirects[i], "HOST", host, 1)
			}

			method := test.method
			if method == "" {
				method = http.MethodGet
			}

			netdialer := &net.Dialer{
				Timeout:   wait.ForeverTestTimeout,
				KeepAlive: wait.ForeverTestTimeout,
			}
			dialer := DialerFunc(func(req *http.Request) (net.Conn, error) {
				conn, err := netdialer.Dial("tcp", req.URL.Host)
				if err != nil {
					return conn, err
				}
				if err = req.Write(conn); err != nil {
					require.NoError(t, conn.Close())
					return nil, fmt.Errorf("error sending request: %v", err)
				}
				return conn, err
			})
			conn, rawResponse, err := ConnectWithRedirects(method, u, http.Header{} /*body*/, nil, dialer, true)
			if test.expectError {
				require.Error(t, err, "expected request error")
				return
			}

			require.NoError(t, err, "unexpected request error")
			assert.NoError(t, conn.Close(), "error closing connection")

			resp, err := http.ReadResponse(bufio.NewReader(bytes.NewReader(rawResponse)), nil)
			require.NoError(t, err, "unexpected request error")

			result, err := ioutil.ReadAll(resp.Body)
			require.NoError(t, resp.Body.Close())
			if test.expectedRedirects < len(test.redirects) {
				// Expect the last redirect to be returned.
				assert.Equal(t, http.StatusFound, resp.StatusCode, "Final response is not a redirect")
				assert.Equal(t, test.redirects[len(test.redirects)-1], resp.Header.Get("Location"))
				assert.NotEqual(t, resultString, string(result), "wrong content")
			} else {
				assert.Equal(t, resultString, string(result), "stream content does not match")
			}
		})
	}
}

func TestAllowsHTTP2(t *testing.T) {
	testcases := []struct {
		Name         string
		Transport    *http.Transport
		ExpectAllows bool
	}{
		{
			Name:         "empty",
			Transport:    &http.Transport{},
			ExpectAllows: true,
		},
		{
			Name:         "empty tlsconfig",
			Transport:    &http.Transport{TLSClientConfig: &tls.Config{}},
			ExpectAllows: true,
		},
		{
			Name:         "zero-length NextProtos",
			Transport:    &http.Transport{TLSClientConfig: &tls.Config{NextProtos: []string{}}},
			ExpectAllows: true,
		},
		{
			Name:         "includes h2 in NextProtos after",
			Transport:    &http.Transport{TLSClientConfig: &tls.Config{NextProtos: []string{"http/1.1", "h2"}}},
			ExpectAllows: true,
		},
		{
			Name:         "includes h2 in NextProtos before",
			Transport:    &http.Transport{TLSClientConfig: &tls.Config{NextProtos: []string{"h2", "http/1.1"}}},
			ExpectAllows: true,
		},
		{
			Name:         "includes h2 in NextProtos between",
			Transport:    &http.Transport{TLSClientConfig: &tls.Config{NextProtos: []string{"http/1.1", "h2", "h3"}}},
			ExpectAllows: true,
		},
		{
			Name:         "excludes h2 in NextProtos",
			Transport:    &http.Transport{TLSClientConfig: &tls.Config{NextProtos: []string{"http/1.1"}}},
			ExpectAllows: false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			allows := allowsHTTP2(tc.Transport)
			if allows != tc.ExpectAllows {
				t.Errorf("expected %v, got %v", tc.ExpectAllows, allows)
			}
		})
	}
}
