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

const statusServerEarlyShutdown = -1

func TestHealthChecker(t *testing.T) {
	AddHealthChecker("http", &HTTPHealthChecker{client: &http.Client{}})
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
					Port: util.NewIntOrStringFromString(port),
					Path: "/foo/bar",
					Host: host,
				},
				Type: "http",
			},
		}
		hc := NewHealthChecker()
		health, err := hc.HealthCheck("test", api.PodState{}, container)
		if err != nil && tt.health != Unknown {
			t.Errorf("Unexpected error: %v", err)
		}
		if health != tt.health {
			t.Errorf("Expected %v, got %v", tt.health, health)
		}
	}
}

func TestFindPortByName(t *testing.T) {
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

func TestMuxHealthChecker(t *testing.T) {
	muxHealthCheckerTests := []struct {
		health    Status
		probeType string
	}{
		{Healthy, "http"},
		{Unknown, "ftp"},
	}
	mc := &muxHealthChecker{
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
		container.LivenessProbe.HTTPGet.Port = util.NewIntOrStringFromString(port)
		container.LivenessProbe.HTTPGet.Host = host
		health, err := mc.HealthCheck("test", api.PodState{}, container)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if health != tt.health {
			t.Errorf("Expected %v, got %v", tt.health, health)
		}
	}
}
