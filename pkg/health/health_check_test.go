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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
					Port: util.MakeIntOrStringFromString(port),
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
	want := 8080
	got := findPortByName(container, "foo")
	if got != want {
		t.Errorf("Expected %v, got %v", want, got)
	}
}

func TestGetURLParts(t *testing.T) {
	testCases := []struct {
		probe *api.HTTPGetProbe
		ok    bool
		host  string
		port  int
		path  string
	}{
		{&api.HTTPGetProbe{Host: "", Port: util.MakeIntOrStringFromInt(-1), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.MakeIntOrStringFromString(""), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.MakeIntOrStringFromString("-1"), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.MakeIntOrStringFromString("not-found"), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.MakeIntOrStringFromString("found"), Path: ""}, true, "127.0.0.1", 93, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.MakeIntOrStringFromInt(76), Path: ""}, true, "127.0.0.1", 76, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.MakeIntOrStringFromString("118"), Path: ""}, true, "127.0.0.1", 118, ""},
		{&api.HTTPGetProbe{Host: "hostname", Port: util.MakeIntOrStringFromInt(76), Path: "path"}, true, "hostname", 76, "path"},
	}

	for _, test := range testCases {
		state := api.PodState{PodIP: "127.0.0.1"}
		container := api.Container{
			Ports: []api.Port{{Name: "found", HostPort: 93}},
			LivenessProbe: &api.LivenessProbe{
				HTTPGet: test.probe,
				Type:    "http",
			},
		}
		host, port, path, err := getURLParts(state, container)
		if !test.ok && err == nil {
			t.Errorf("Expected error for %+v, got %s:%d/%s", test, host, port, path)
		}
		if test.ok && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if test.ok {
			if host != test.host || port != test.port || path != test.path {
				t.Errorf("Expected %s:%d/%s, got %s:%d/%s",
					test.host, test.port, test.path, host, port, path)
			}
		}
	}
}

func TestFormatURL(t *testing.T) {
	testCases := []struct {
		host   string
		port   int
		path   string
		result string
	}{
		{"localhost", 93, "", "http://localhost:93"},
		{"localhost", 93, "/path", "http://localhost:93/path"},
	}
	for _, test := range testCases {
		url := formatURL(test.host, test.port, test.path)
		if url != test.result {
			t.Errorf("Expected %s, got %s", test.result, url)
		}
	}
}

func TestHTTPHealthChecker(t *testing.T) {
	testCases := []struct {
		probe  *api.HTTPGetProbe
		status int
		health Status
	}{
		// The probe will be filled in below.  This is primarily testing that an HTTP GET happens.
		{&api.HTTPGetProbe{}, http.StatusOK, Healthy},
		{&api.HTTPGetProbe{}, -1, Unhealthy},
		{nil, -1, Unknown},
	}
	hc := &HTTPHealthChecker{
		client: &http.Client{},
	}
	for _, test := range testCases {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(test.status)
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
				HTTPGet: test.probe,
				Type:    "http",
			},
		}
		params := container.LivenessProbe.HTTPGet
		if params != nil {
			params.Port = util.MakeIntOrStringFromString(port)
			params.Host = host
		}
		health, err := hc.HealthCheck(api.PodState{PodIP: host}, container)
		if test.health == Unknown && err == nil {
			t.Errorf("Expected error")
		}
		if test.health != Unknown && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if health != test.health {
			t.Errorf("Expected %v, got %v", test.health, health)
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
		container.LivenessProbe.HTTPGet.Port = util.MakeIntOrStringFromString(port)
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
