/*
Copyright 2015 The Kubernetes Authors.

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

package httpstream

import (
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

type responseWriter struct {
	header     http.Header
	statusCode *int
}

func newResponseWriter() *responseWriter {
	return &responseWriter{
		header: make(http.Header),
	}
}

func (r *responseWriter) Header() http.Header {
	return r.header
}

func (r *responseWriter) WriteHeader(code int) {
	r.statusCode = &code
}

func (r *responseWriter) Write([]byte) (int, error) {
	return 0, nil
}

func TestHandshake(t *testing.T) {
	tests := map[string]struct {
		clientProtocols  []string
		serverProtocols  []string
		expectedProtocol string
		expectError      bool
	}{
		"no common protocol": {
			clientProtocols:  []string{"c"},
			serverProtocols:  []string{"a", "b"},
			expectedProtocol: "",
			expectError:      true,
		},
		"no common protocol with comma separated list": {
			clientProtocols:  []string{"c, d"},
			serverProtocols:  []string{"a", "b"},
			expectedProtocol: "",
			expectError:      true,
		},
		"common protocol": {
			clientProtocols:  []string{"b"},
			serverProtocols:  []string{"a", "b"},
			expectedProtocol: "b",
		},
		"common protocol with comma separated list": {
			clientProtocols:  []string{"b, c"},
			serverProtocols:  []string{"a", "b"},
			expectedProtocol: "b",
		},
	}

	for name, test := range tests {
		req, err := http.NewRequest(http.MethodGet, "http://www.example.com/", nil)
		if err != nil {
			t.Fatalf("%s: error creating request: %v", name, err)
		}

		for _, p := range test.clientProtocols {
			req.Header.Add(HeaderProtocolVersion, p)
		}

		w := newResponseWriter()
		negotiated, err := Handshake(req, w, test.serverProtocols)

		// verify negotiated protocol
		if e, a := test.expectedProtocol, negotiated; e != a {
			t.Errorf("%s: protocol: expected %q, got %q", name, e, a)
		}

		if test.expectError {
			if err == nil {
				t.Errorf("%s: expected error but did not get one", name)
			}
			if w.statusCode == nil {
				t.Errorf("%s: expected w.statusCode to be set", name)
			} else if e, a := http.StatusForbidden, *w.statusCode; e != a {
				t.Errorf("%s: w.statusCode: expected %d, got %d", name, e, a)
			}
			if e, a := test.serverProtocols, w.Header()[HeaderAcceptedProtocolVersions]; !reflect.DeepEqual(e, a) {
				t.Errorf("%s: accepted server protocols: expected %v, got %v", name, e, a)
			}
			continue
		}
		if !test.expectError && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
			continue
		}
		if w.statusCode != nil {
			t.Errorf("%s: unexpected non-nil w.statusCode: %d", name, w.statusCode)
		}

		if len(test.expectedProtocol) == 0 {
			if len(w.Header()[HeaderProtocolVersion]) > 0 {
				t.Errorf("%s: unexpected protocol version response header: %s", name, w.Header()[HeaderProtocolVersion])
			}
			continue
		}

		// verify response headers
		if e, a := []string{test.expectedProtocol}, w.Header()[HeaderProtocolVersion]; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: protocol response header: expected %v, got %v", name, e, a)
		}
	}
}

func TestIsUpgradeFailureError(t *testing.T) {
	testCases := map[string]struct {
		err      error
		expected bool
	}{
		"nil error should return false": {
			err:      nil,
			expected: false,
		},
		"Non-upgrade error should return false": {
			err:      fmt.Errorf("this is not an upgrade error"),
			expected: false,
		},
		"UpgradeFailure error should return true": {
			err:      &UpgradeFailureError{},
			expected: true,
		},
		"Wrapped Non-UpgradeFailure error should return false": {
			err:      fmt.Errorf("%s: %w", "first error", errors.New("Non-upgrade error")),
			expected: false,
		},
		"Wrapped UpgradeFailure error should return true": {
			err:      fmt.Errorf("%s: %w", "first error", &UpgradeFailureError{}),
			expected: true,
		},
	}

	for name, test := range testCases {
		t.Run(name, func(t *testing.T) {
			actual := IsUpgradeFailure(test.err)
			if test.expected != actual {
				t.Errorf("expected upgrade failure %t, got %t", test.expected, actual)
			}
		})
	}
}

func TestIsHTTPSProxyError(t *testing.T) {
	testCases := map[string]struct {
		err      error
		expected bool
	}{
		"nil error should return false": {
			err:      nil,
			expected: false,
		},
		"Not HTTPS proxy error should return false": {
			err:      errors.New("this is not an upgrade error"),
			expected: false,
		},
		"HTTPS proxy error should return true": {
			err:      errors.New("proxy: unknown scheme: https"),
			expected: true,
		},
	}

	for name, test := range testCases {
		t.Run(name, func(t *testing.T) {
			actual := IsHTTPSProxyError(test.err)
			if test.expected != actual {
				t.Errorf("expected HTTPS proxy error %t, got %t", test.expected, actual)
			}
		})
	}
}
