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
	"net/http"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
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
		"no client protocols": {
			clientProtocols:  []string{},
			serverProtocols:  []string{"a", "b"},
			expectedProtocol: "",
		},
		"no common protocol": {
			clientProtocols:  []string{"c"},
			serverProtocols:  []string{"a", "b"},
			expectedProtocol: "",
			expectError:      true,
		},
		"common protocol": {
			clientProtocols:  []string{"b"},
			serverProtocols:  []string{"a", "b"},
			expectedProtocol: "b",
		},
	}

	for name, test := range tests {
		req, err := http.NewRequest("GET", "http://www.example.com/", nil)
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
		if e, a := []string{test.expectedProtocol}, w.Header()[HeaderProtocolVersion]; !api.Semantic.DeepEqual(e, a) {
			t.Errorf("%s: protocol response header: expected %v, got %v", name, e, a)
		}
	}
}
