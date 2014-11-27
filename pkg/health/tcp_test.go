/*
Copyright 2014 Google Inc. All rights reserved.

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

package health

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestGetTCPAddrParts(t *testing.T) {
	testCases := []struct {
		probe *api.TCPSocketAction
		ok    bool
		host  string
		port  int
	}{
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromInt(-1)}, false, "", -1},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("")}, false, "", -1},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("-1")}, false, "", -1},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("not-found")}, false, "", -1},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("found")}, true, "1.2.3.4", 93},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromInt(76)}, true, "1.2.3.4", 76},
		{&api.TCPSocketAction{Port: util.NewIntOrStringFromString("118")}, true, "1.2.3.4", 118},
	}

	for _, test := range testCases {
		state := api.PodState{PodIP: "1.2.3.4"}
		container := api.Container{
			Ports: []api.Port{{Name: "found", HostPort: 93}},
			LivenessProbe: &api.LivenessProbe{
				TCPSocket: test.probe,
			},
		}
		host, port, err := getTCPAddrParts(state, container)
		if !test.ok && err == nil {
			t.Errorf("Expected error for %+v, got %s:%d", test, host, port)
		}
		if test.ok && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if test.ok {
			if host != test.host || port != test.port {
				t.Errorf("Expected %s:%d, got %s:%d", test.host, test.port, host, port)
			}
		}
	}
}

func TestTcpHealthChecker(t *testing.T) {
	tests := []struct {
		probe          *api.TCPSocketAction
		expectedStatus Status
		expectError    bool
	}{
		// The probe will be filled in below.  This is primarily testing that a connection is made.
		{&api.TCPSocketAction{}, Healthy, false},
		{&api.TCPSocketAction{}, Unhealthy, false},
		{nil, Unknown, true},
	}

	checker := &TCPHealthChecker{}
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
		container := api.Container{
			LivenessProbe: &api.LivenessProbe{
				TCPSocket: test.probe,
			},
		}
		params := container.LivenessProbe.TCPSocket
		if params != nil && test.expectedStatus == Healthy {
			params.Port = util.NewIntOrStringFromString(port)
		}
		status, err := checker.HealthCheck("test", "", api.PodState{PodIP: host}, container)
		if status != test.expectedStatus {
			t.Errorf("expected: %v, got: %v", test.expectedStatus, status)
		}
		if err != nil && !test.expectError {
			t.Errorf("unexpected error: %#v", err)
		}
		if err == nil && test.expectError {
			t.Errorf("unexpected non-error.")
		}
	}
}
