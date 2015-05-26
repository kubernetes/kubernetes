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

package http

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
)

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

func TestHTTPProbeChecker(t *testing.T) {
	handleReq := func(s int, body string) func(w http.ResponseWriter) {
		return func(w http.ResponseWriter) {
			w.WriteHeader(s)
			w.Write([]byte(body))
		}
	}

	prober := New()
	testCases := []struct {
		handler func(w http.ResponseWriter)
		health  probe.Result
		body    string
	}{
		// The probe will be filled in below.  This is primarily testing that an HTTP GET happens.
		{handleReq(http.StatusOK, "ok body"), probe.Success, "ok body"},
		{handleReq(-1, "fail body"), probe.Failure, "fail body"},
		{func(w http.ResponseWriter) { time.Sleep(3 * time.Second) }, probe.Failure, "use of closed network connection"},
	}
	for _, test := range testCases {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			test.handler(w)
		}))
		u, err := url.Parse(server.URL)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		host, port, err := net.SplitHostPort(u.Host)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		p, err := strconv.Atoi(port)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		health, output, err := prober.Probe(host, p, "", 1*time.Second)
		if test.health == probe.Unknown && err == nil {
			t.Errorf("Expected error")
		}
		if test.health != probe.Unknown && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if health != test.health {
			t.Errorf("Expected %v, got %v", test.health, health)
		}
		if !strings.Contains(output, test.body) {
			t.Errorf("Expected %v, got %v", test.body, output)
		}
	}
}
