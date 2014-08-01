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
	"strconv"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

const statusServerEarlyShutdown = -1

func TestHealthChecker(t *testing.T) {
	var healthCheckerTests = []struct {
		status int
		health Status
	}{
		{http.StatusOK, Healthy},
		{statusServerEarlyShutdown, Unknown},
		{http.StatusBadRequest, Unhealthy},
		{http.StatusBadGateway, Unhealthy},
		{http.StatusInternalServerError, Unhealthy},
	}
	for _, healthCheckerTest := range healthCheckerTests {
		tt := healthCheckerTest
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tt.status)
		}))
		u, err := url.Parse(ts.URL)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		host, port, err := net.SplitHostPort(u.Host)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if tt.status == statusServerEarlyShutdown {
			ts.Close()
		}
		container := api.Container{
			LivenessProbe: &api.LivenessProbe{
				HTTPGet: &api.HTTPGetProbe{
					Port: port,
					Path: "/foo/bar",
					Host: host,
				},
				Type: "http",
			},
		}
		hc := NewHealthChecker()
		health, err := hc.HealthCheck(api.PodState{}, container)
		if err != nil && tt.health != Unknown {
			t.Errorf("Unexpected error: %v", err)
		}
		if health != tt.health {
			t.Errorf("Expected %v, got %v", tt.health, health)
		}
	}
}

func TestFindPort(t *testing.T) {
	container := api.Container{
		Ports: []api.Port{
			{
				Name:     "foo",
				HostPort: 8080,
			},
			{
				Name:     "bar",
				HostPort: 9000,
			},
		},
	}
	checker := HTTPHealthChecker{}
	want := int64(8080)
	got := checker.findPort(container, "foo")
	if got != want {
		t.Errorf("Expected %v, got %v", want, got)
	}
}

func TestHTTPHealthChecker(t *testing.T) {
	httpHealthCheckerTests := []struct {
		probe  *api.HTTPGetProbe
		status int
		health Status
	}{
		{&api.HTTPGetProbe{Host: "httptest"}, http.StatusOK, Healthy},
		{&api.HTTPGetProbe{}, http.StatusOK, Healthy},
		{nil, -1, Unknown},
		{&api.HTTPGetProbe{Port: "-1"}, -1, Unknown},
	}
	hc := &HTTPHealthChecker{
		client: &http.Client{},
	}
	for _, httpHealthCheckerTest := range httpHealthCheckerTests {
		tt := httpHealthCheckerTest
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tt.status)
		}))
		u, err := url.Parse(ts.URL)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		host, port, err := net.SplitHostPort(u.Host)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		container := api.Container{
			LivenessProbe: &api.LivenessProbe{
				HTTPGet: tt.probe,
				Type:    "http",
			},
		}
		params := container.LivenessProbe.HTTPGet
		if params != nil {
			if params.Port == "" {
				params.Port = port
			}
			if params.Host == "httptest" {
				params.Host = host
			}
		}
		health, err := hc.HealthCheck(api.PodState{PodIP: host}, container)
		if tt.health == Unknown && err == nil {
			t.Errorf("Expected error")
		}
		if tt.health != Unknown && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if health != tt.health {
			t.Errorf("Expected %v, got %v", tt.health, health)
		}
	}
}

func TestTcpHealthChecker(t *testing.T) {
	type tcpHealthTest struct {
		probe          *api.LivenessProbe
		expectedStatus Status
		expectError    bool
	}

	checker := &TCPHealthChecker{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	host, port, err := net.SplitHostPort(u.Host)
	portNum, _ := strconv.Atoi(port)

	tests := []tcpHealthTest{
		{&api.LivenessProbe{TCPSocket: &api.TCPSocketProbe{Port: portNum}}, Healthy, false},
		{&api.LivenessProbe{TCPSocket: &api.TCPSocketProbe{Port: 100000}}, Unhealthy, false},
		{&api.LivenessProbe{}, Unknown, true},
	}
	for _, test := range tests {
		probe := test.probe
		container := api.Container{
			LivenessProbe: probe,
		}
		status, err := checker.HealthCheck(api.PodState{PodIP: host}, container)
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

func TestMuxHealthChecker(t *testing.T) {
	muxHealthCheckerTests := []struct {
		health    Status
		probeType string
	}{
		{Healthy, "http"},
		{Unknown, "ftp"},
	}
	mc := &MuxHealthChecker{
		checkers: make(map[string]HealthChecker),
	}
	hc := &HTTPHealthChecker{
		client: &http.Client{},
	}
	mc.checkers["http"] = hc
	for _, muxHealthCheckerTest := range muxHealthCheckerTests {
		tt := muxHealthCheckerTest
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		u, err := url.Parse(ts.URL)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		host, port, err := net.SplitHostPort(u.Host)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		container := api.Container{
			LivenessProbe: &api.LivenessProbe{
				HTTPGet: &api.HTTPGetProbe{},
			},
		}
		container.LivenessProbe.Type = tt.probeType
		container.LivenessProbe.HTTPGet.Port = port
		container.LivenessProbe.HTTPGet.Host = host
		health, err := mc.HealthCheck(api.PodState{}, container)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if health != tt.health {
			t.Errorf("Expected %v, got %v", tt.health, health)
		}
	}
}
