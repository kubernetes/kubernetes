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

func TestGetURLParts(t *testing.T) {
	testCases := []struct {
		probe *api.HTTPGetProbe
		ok    bool
		host  string
		port  int
		path  string
	}{
		{&api.HTTPGetProbe{Host: "", Port: util.NewIntOrStringFromInt(-1), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.NewIntOrStringFromString(""), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.NewIntOrStringFromString("-1"), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.NewIntOrStringFromString("not-found"), Path: ""}, false, "", -1, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.NewIntOrStringFromString("found"), Path: ""}, true, "127.0.0.1", 93, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.NewIntOrStringFromInt(76), Path: ""}, true, "127.0.0.1", 76, ""},
		{&api.HTTPGetProbe{Host: "", Port: util.NewIntOrStringFromString("118"), Path: ""}, true, "127.0.0.1", 118, ""},
		{&api.HTTPGetProbe{Host: "hostname", Port: util.NewIntOrStringFromInt(76), Path: "path"}, true, "hostname", 76, "path"},
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
			params.Port = util.NewIntOrStringFromString(port)
			params.Host = host
		}
		health, err := hc.HealthCheck("test", api.PodState{PodIP: host}, container)
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
