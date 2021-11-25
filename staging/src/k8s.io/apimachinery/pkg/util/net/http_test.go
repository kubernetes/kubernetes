//go:build go1.8
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
	"io"
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
	netutils "k8s.io/utils/net"
)

func TestGetClientIP(t *testing.T) {
	ipString := "10.0.0.1"
	ip := netutils.ParseIPSloppy(ipString)
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
			assert.NoError(t, err)
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

func TestSourceIPs(t *testing.T) {
	tests := []struct {
		name         string
		realIP       string
		forwardedFor string
		remoteAddr   string
		expected     []string
	}{{
		name:     "no headers, missing remoteAddr",
		expected: []string{},
	}, {
		name:       "no headers, just remoteAddr host:port",
		remoteAddr: "1.2.3.4:555",
		expected:   []string{"1.2.3.4"},
	}, {
		name:       "no headers, just remoteAddr host",
		remoteAddr: "1.2.3.4",
		expected:   []string{"1.2.3.4"},
	}, {
		name:         "empty forwarded-for chain",
		forwardedFor: " ",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"1.2.3.4"},
	}, {
		name:         "invalid forwarded-for chain",
		forwardedFor: "garbage garbage values!",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"1.2.3.4"},
	}, {
		name:         "partially invalid forwarded-for chain",
		forwardedFor: "garbage garbage values!,4.5.6.7",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"4.5.6.7", "1.2.3.4"},
	}, {
		name:         "valid forwarded-for chain",
		forwardedFor: "120.120.120.126,2.2.2.2,4.5.6.7",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"120.120.120.126", "2.2.2.2", "4.5.6.7", "1.2.3.4"},
	}, {
		name:         "valid forwarded-for chain with redundant remoteAddr",
		forwardedFor: "2.2.2.2,1.2.3.4",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"2.2.2.2", "1.2.3.4"},
	}, {
		name:       "invalid Real-Ip",
		realIP:     "garbage, just garbage!",
		remoteAddr: "1.2.3.4",
		expected:   []string{"1.2.3.4"},
	}, {
		name:         "invalid Real-Ip with forwarded-for",
		realIP:       "garbage, just garbage!",
		forwardedFor: "2.2.2.2",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"2.2.2.2", "1.2.3.4"},
	}, {
		name:       "valid Real-Ip",
		realIP:     "2.2.2.2",
		remoteAddr: "1.2.3.4",
		expected:   []string{"2.2.2.2", "1.2.3.4"},
	}, {
		name:       "redundant Real-Ip",
		realIP:     "1.2.3.4",
		remoteAddr: "1.2.3.4",
		expected:   []string{"1.2.3.4"},
	}, {
		name:         "valid Real-Ip with forwarded-for",
		realIP:       "2.2.2.2",
		forwardedFor: "120.120.120.126,4.5.6.7",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"120.120.120.126", "4.5.6.7", "2.2.2.2", "1.2.3.4"},
	}, {
		name:         "redundant Real-Ip with forwarded-for",
		realIP:       "2.2.2.2",
		forwardedFor: "120.120.120.126,2.2.2.2,4.5.6.7",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"120.120.120.126", "2.2.2.2", "4.5.6.7", "1.2.3.4"},
	}, {
		name:         "full redundancy",
		realIP:       "1.2.3.4",
		forwardedFor: "1.2.3.4",
		remoteAddr:   "1.2.3.4",
		expected:     []string{"1.2.3.4"},
	}, {
		name:         "full ipv6",
		realIP:       "abcd:ef01:2345:6789:abcd:ef01:2345:6789",
		forwardedFor: "aaaa:bbbb:cccc:dddd:eeee:ffff:0:1111,0:1111:2222:3333:4444:5555:6666:7777",
		remoteAddr:   "aaaa:aaaa:aaaa:aaaa:aaaa:aaaa:aaaa:aaaa",
		expected: []string{
			"aaaa:bbbb:cccc:dddd:eeee:ffff:0:1111",
			"0:1111:2222:3333:4444:5555:6666:7777",
			"abcd:ef01:2345:6789:abcd:ef01:2345:6789",
			"aaaa:aaaa:aaaa:aaaa:aaaa:aaaa:aaaa:aaaa",
		},
	}, {
		name:         "mixed ipv4 ipv6",
		forwardedFor: "aaaa:bbbb:cccc:dddd:eeee:ffff:0:1111,1.2.3.4",
		remoteAddr:   "0:0:0:0:0:ffff:102:304", // ipv6 equivalent to 1.2.3.4
		expected: []string{
			"aaaa:bbbb:cccc:dddd:eeee:ffff:0:1111",
			"1.2.3.4",
		},
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "https://cluster.k8s.io/apis/foobars/v1/foo/bar", nil)
			req.RemoteAddr = test.remoteAddr
			if test.forwardedFor != "" {
				req.Header.Set("X-Forwarded-For", test.forwardedFor)
			}
			if test.realIP != "" {
				req.Header.Set("X-Real-Ip", test.realIP)
			}

			actualIPs := SourceIPs(req)
			actual := make([]string, len(actualIPs))
			for i, ip := range actualIPs {
				actual[i] = ip.String()
			}

			assert.Equal(t, test.expected, actual)
		})
	}
}

func TestParseWarningHeader(t *testing.T) {
	tests := []struct {
		name string

		header string

		wantResult    WarningHeader
		wantRemainder string
		wantErr       string
	}{
		// invalid cases
		{
			name:    "empty",
			header:  ``,
			wantErr: "fewer than 3 segments",
		},
		{
			name:    "bad code",
			header:  `A B`,
			wantErr: "fewer than 3 segments",
		},
		{
			name:    "short code",
			header:  `1 - "text"`,
			wantErr: "not 3 digits",
		},
		{
			name:    "bad code",
			header:  `A - "text"`,
			wantErr: "not 3 digits",
		},
		{
			name:    "invalid date quoting",
			header:  `  299 - "text\"\\\a\b\c"  "Tue, 15 Nov 1994 08:12:31 GMT `,
			wantErr: "unterminated date segment",
		},
		{
			name:    "invalid post-date",
			header:  `  299 - "text\"\\\a\b\c"  "Tue, 15 Nov 1994 08:12:31 GMT" other`,
			wantErr: "unexpected token after warn-date",
		},
		{
			name:    "agent control character",
			header:  "  299 agent\u0000name \"text\"",
			wantErr: "invalid agent",
		},
		{
			name:    "agent non-utf8 character",
			header:  "  299 agent\xc5name \"text\"",
			wantErr: "invalid agent",
		},
		{
			name:    "text control character",
			header:  "  299 - \"text\u0000\"content",
			wantErr: "invalid text",
		},
		{
			name:    "text non-utf8 character",
			header:  "  299 - \"text\xc5\"content",
			wantErr: "invalid text",
		},

		// valid cases
		{
			name:       "ok",
			header:     `299 - "text"`,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `text`},
		},
		{
			name:       "ok",
			header:     `299 - "text\"\\\a\b\c"`,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `text"\abc`},
		},
		// big code
		{
			name:       "big code",
			header:     `321 - "text"`,
			wantResult: WarningHeader{Code: 321, Agent: "-", Text: "text"},
		},
		// RFC 2047 decoding
		{
			name:       "ok, rfc 2047, iso-8859-1, q",
			header:     `299 - "=?iso-8859-1?q?this=20is=20some=20text?="`,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `this is some text`},
		},
		{
			name:       "ok, rfc 2047, utf-8, b",
			header:     `299 - "=?UTF-8?B?VGhpcyBpcyBhIGhvcnNleTog8J+Qjg==?= And =?UTF-8?B?VGhpcyBpcyBhIGhvcnNleTog8J+Qjg==?="`,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `This is a horsey: 🐎 And This is a horsey: 🐎`},
		},
		{
			name:       "ok, rfc 2047, utf-8, q",
			header:     `299 - "=?UTF-8?Q?This is a \"horsey\": =F0=9F=90=8E?="`,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `This is a "horsey": 🐎`},
		},
		{
			name:       "ok, rfc 2047, unknown charset",
			header:     `299 - "=?UTF-9?Q?This is a horsey: =F0=9F=90=8E?="`,
			wantResult: WarningHeader{Code: 299, Agent: "-", Text: `=?UTF-9?Q?This is a horsey: =F0=9F=90=8E?=`},
		},
		{
			name:       "ok with spaces",
			header:     `  299 - "text\"\\\a\b\c"  `,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `text"\abc`},
		},
		{
			name:       "ok with date",
			header:     `  299 - "text\"\\\a\b\c"  "Tue, 15 Nov 1994 08:12:31 GMT" `,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `text"\abc`},
		},
		{
			name:       "ok with date and comma",
			header:     `  299 - "text\"\\\a\b\c"  "Tue, 15 Nov 1994 08:12:31 GMT" , `,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `text"\abc`},
		},
		{
			name:       "ok with comma",
			header:     `  299 - "text\"\\\a\b\c"  , `,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `text"\abc`},
		},
		{
			name:          "ok with date and comma and remainder",
			header:        `  299 - "text\"\\\a\b\c"  "Tue, 15 Nov 1994 08:12:31 GMT" , remainder `,
			wantResult:    WarningHeader{Code: 299, Agent: `-`, Text: `text"\abc`},
			wantRemainder: "remainder",
		},
		{
			name:          "ok with comma and remainder",
			header:        `  299 - "text\"\\\a\b\c"  ,remainder text,second remainder`,
			wantResult:    WarningHeader{Code: 299, Agent: `-`, Text: `text"\abc`},
			wantRemainder: "remainder text,second remainder",
		},
		{
			name:       "ok with utf-8 content directly in warn-text",
			header:     ` 299 - "Test of Iñtërnâtiônàlizætiøn,💝🐹🌇⛔" `,
			wantResult: WarningHeader{Code: 299, Agent: `-`, Text: `Test of Iñtërnâtiônàlizætiøn,💝🐹🌇⛔`},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotResult, gotRemainder, err := ParseWarningHeader(tt.header)
			switch {
			case err == nil && len(tt.wantErr) > 0:
				t.Errorf("ParseWarningHeader() no error, expected error %q", tt.wantErr)
				return
			case err != nil && len(tt.wantErr) == 0:
				t.Errorf("ParseWarningHeader() error %q, expected no error", err)
				return
			case err != nil && len(tt.wantErr) > 0 && !strings.Contains(err.Error(), tt.wantErr):
				t.Errorf("ParseWarningHeader() error %q, expected error %q", err, tt.wantErr)
				return
			}
			if err != nil {
				return
			}
			if !reflect.DeepEqual(gotResult, tt.wantResult) {
				t.Errorf("ParseWarningHeader() gotResult = %#v, want %#v", gotResult, tt.wantResult)
			}
			if gotRemainder != tt.wantRemainder {
				t.Errorf("ParseWarningHeader() gotRemainder = %v, want %v", gotRemainder, tt.wantRemainder)
			}
		})
	}
}

func TestNewWarningHeader(t *testing.T) {
	tests := []struct {
		name string

		code  int
		agent string
		text  string

		want    string
		wantErr string
	}{
		// invalid cases
		{
			name:    "code too low",
			code:    -1,
			agent:   `-`,
			text:    `example warning`,
			wantErr: "between 0 and 999",
		},
		{
			name:    "code too high",
			code:    1000,
			agent:   `-`,
			text:    `example warning`,
			wantErr: "between 0 and 999",
		},
		{
			name:    "agent with space",
			code:    299,
			agent:   `test agent`,
			text:    `example warning`,
			wantErr: `agent must be valid`,
		},
		{
			name:    "agent with newline",
			code:    299,
			agent:   "test\nagent",
			text:    `example warning`,
			wantErr: `agent must be valid`,
		},
		{
			name:    "agent with backslash",
			code:    299,
			agent:   `test\agent`,
			text:    `example warning`,
			wantErr: `agent must be valid`,
		},
		{
			name:    "agent with quote",
			code:    299,
			agent:   `test"agent"`,
			text:    `example warning`,
			wantErr: `agent must be valid`,
		},
		{
			name:    "agent with control character",
			code:    299,
			agent:   "test\u0000agent",
			text:    `example warning`,
			wantErr: `agent must be valid`,
		},
		{
			name:    "agent with non-UTF8",
			code:    299,
			agent:   "test\xc5agent",
			text:    `example warning`,
			wantErr: `agent must be valid`,
		},
		{
			name:    "text with newline",
			code:    299,
			agent:   `-`,
			text:    "Test of new\nline",
			wantErr: "text must be valid",
		},
		{
			name:    "text with control character",
			code:    299,
			agent:   `-`,
			text:    "Test of control\u0000character",
			wantErr: "text must be valid",
		},
		{
			name:    "text with non-UTF8",
			code:    299,
			agent:   `-`,
			text:    "Test of control\xc5character",
			wantErr: "text must be valid",
		},

		{
			name:  "valid empty text",
			code:  299,
			agent: `-`,
			text:  ``,
			want:  `299 - ""`,
		},
		{
			name:  "valid empty agent",
			code:  299,
			agent: ``,
			text:  `example warning`,
			want:  `299 - "example warning"`,
		},
		{
			name:  "valid low code",
			code:  1,
			agent: `-`,
			text:  `example warning`,
			want:  `001 - "example warning"`,
		},
		{
			name:  "valid high code",
			code:  999,
			agent: `-`,
			text:  `example warning`,
			want:  `999 - "example warning"`,
		},
		{
			name:  "valid utf-8",
			code:  299,
			agent: `-`,
			text:  `Test of "Iñtërnâtiônàlizætiøn,💝🐹🌇⛔"`,
			want:  `299 - "Test of \"Iñtërnâtiônàlizætiøn,💝🐹🌇⛔\""`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewWarningHeader(tt.code, tt.agent, tt.text)

			switch {
			case err == nil && len(tt.wantErr) > 0:
				t.Fatalf("ParseWarningHeader() no error, expected error %q", tt.wantErr)
			case err != nil && len(tt.wantErr) == 0:
				t.Fatalf("ParseWarningHeader() error %q, expected no error", err)
			case err != nil && len(tt.wantErr) > 0 && !strings.Contains(err.Error(), tt.wantErr):
				t.Fatalf("ParseWarningHeader() error %q, expected error %q", err, tt.wantErr)
			}
			if err != nil {
				return
			}

			if got != tt.want {
				t.Fatalf("NewWarningHeader() = %v, want %v", got, tt.want)
			}

			roundTrip, remaining, err := ParseWarningHeader(got)
			if err != nil {
				t.Fatalf("error roundtripping: %v", err)
			}
			if len(remaining) > 0 {
				t.Fatalf("unexpected remainder roundtripping: %s", remaining)
			}
			agent := tt.agent
			if len(agent) == 0 {
				agent = "-"
			}
			expect := WarningHeader{Code: tt.code, Agent: agent, Text: tt.text}
			if roundTrip != expect {
				t.Fatalf("after round trip, want:\n%#v\ngot\n%#v", expect, roundTrip)
			}
		})
	}
}

func TestParseWarningHeaders(t *testing.T) {
	tests := []struct {
		name string

		headers []string

		want     []WarningHeader
		wantErrs []string
	}{
		{
			name:     "empty",
			headers:  []string{},
			want:     nil,
			wantErrs: []string{},
		},
		{
			name: "multi-header with error",
			headers: []string{
				`299 - "warning 1.1",299 - "warning 1.2"`,
				`299 - "warning 2", 299 - "warning unquoted`,
				` 299 - "warning 3.1" ,  299 - "warning 3.2" `,
			},
			want: []WarningHeader{
				{Code: 299, Agent: "-", Text: "warning 1.1"},
				{Code: 299, Agent: "-", Text: "warning 1.2"},
				{Code: 299, Agent: "-", Text: "warning 2"},
				{Code: 299, Agent: "-", Text: "warning 3.1"},
				{Code: 299, Agent: "-", Text: "warning 3.2"},
			},
			wantErrs: []string{"invalid warning header: invalid quoted string: missing closing quote"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, gotErrs := ParseWarningHeaders(tt.headers)

			switch {
			case len(gotErrs) != len(tt.wantErrs):
				t.Fatalf("ParseWarningHeader() got %v, expected %v", gotErrs, tt.wantErrs)
			case len(gotErrs) == len(tt.wantErrs) && len(gotErrs) > 0:
				gotErrStrings := []string{}
				for _, err := range gotErrs {
					gotErrStrings = append(gotErrStrings, err.Error())
				}
				if !reflect.DeepEqual(gotErrStrings, tt.wantErrs) {
					t.Fatalf("ParseWarningHeader() got %v, expected %v", gotErrs, tt.wantErrs)
				}
			}
			if len(gotErrs) > 0 {
				return
			}

			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseWarningHeaders() got %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestIsProbableEOF(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "with no error",
			expected: false,
		},
		{
			name:     "with EOF error",
			err:      io.EOF,
			expected: true,
		},
		{
			name:     "with unexpected EOF error",
			err:      io.ErrUnexpectedEOF,
			expected: true,
		},
		{
			name:     "with broken connection error",
			err:      fmt.Errorf("http: can't write HTTP request on broken connection"),
			expected: true,
		},
		{
			name:     "with server sent GOAWAY error",
			err:      fmt.Errorf("error foo - http2: server sent GOAWAY and closed the connection - error bar"),
			expected: true,
		},
		{
			name:     "with connection reset by peer error",
			err:      fmt.Errorf("error foo - connection reset by peer - error bar"),
			expected: true,
		},
		{
			name:     "with use of closed network connection error",
			err:      fmt.Errorf("error foo - Use of closed network connection - error bar"),
			expected: true,
		},
		{
			name: "with url error",
			err: &url.Error{
				Err: io.ErrUnexpectedEOF,
			},
			expected: true,
		},
		{
			name:     "with unrecognized error",
			err:      fmt.Errorf("error foo"),
			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := IsProbableEOF(test.err)
			assert.Equal(t, test.expected, actual)
		})
	}
}

func setEnv(key, value string) func() {
	originalValue := os.Getenv(key)
	os.Setenv(key, value)
	return func() {
		os.Setenv(key, originalValue)
	}
}

func TestReadIdleTimeoutSeconds(t *testing.T) {
	reset := setEnv("HTTP2_READ_IDLE_TIMEOUT_SECONDS", "60")
	if e, a := 60, readIdleTimeoutSeconds(); e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	reset()

	reset = setEnv("HTTP2_READ_IDLE_TIMEOUT_SECONDS", "illegal value")
	if e, a := 30, readIdleTimeoutSeconds(); e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	reset()
}

func TestPingTimeoutSeconds(t *testing.T) {
	reset := setEnv("HTTP2_PING_TIMEOUT_SECONDS", "60")
	if e, a := 60, pingTimeoutSeconds(); e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	reset()

	reset = setEnv("HTTP2_PING_TIMEOUT_SECONDS", "illegal value")
	if e, a := 15, pingTimeoutSeconds(); e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	reset()
}

func Benchmark_ParseQuotedString(b *testing.B) {
	str := `"The quick brown" fox jumps over the lazy dog`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quoted, remainder, err := parseQuotedString(str)
		if err != nil {
			b.Errorf("Unexpected error %s", err)
		}
		if quoted != "The quick brown" {
			b.Errorf("Unexpected quoted string %s", quoted)
		}
		if remainder != "fox jumps over the lazy dog" {
			b.Errorf("Unexpected remainder string %s", quoted)
		}
	}
}
