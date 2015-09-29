/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package tcp

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/probe"
)

func containsAny(s string, substrs []string) bool {
	for _, substr := range substrs {
		if strings.Contains(s, substr) {
			return true
		}
	}
	return false
}

func TestTcpHealthChecker(t *testing.T) {
	prober := New()
	tests := []struct {
		expectedStatus probe.Result
		usePort        bool
		expectError    bool
		// Some errors are different depending on your system, make
		// the test pass on all of them
		accOutputs []string
	}{
		// The probe will be filled in below.  This is primarily testing that a connection is made.
		{probe.Success, true, false, []string{""}},
		{probe.Failure, false, false, []string{"unknown port", "Servname not supported for ai_socktype"}},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	host, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	for _, test := range tests {
		p, err := strconv.Atoi(port)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !test.usePort {
			p = -1
		}
		status, output, err := prober.Probe(host, p, 1*time.Second)
		if status != test.expectedStatus {
			t.Errorf("expected: %v, got: %v", test.expectedStatus, status)
		}
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && test.expectError {
			t.Errorf("unexpected non-error.")
		}
		if !containsAny(output, test.accOutputs) {
			t.Errorf("expected one of %#v, got %s", test.accOutputs, output)
		}
	}
}
