// +build integration,!no-etcd

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

package integration

// This file tests authentication and (soon) authorization of HTTP requests to a master object.
// It does not use the client in pkg/client/... because authentication and authorization needs
// to work for any client of the HTTP interface.

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
)

func init() {
	requireEtcd()
}

// TestWhoAmI passes a known Bearer Token to the master's /_whoami endpoint and checks that
// the master authenticates the user.
func TestWhoAmI(t *testing.T) {
	deleteAllEtcdKeys()

	// Write a token file.
	json := `
abc123,alice,1
xyz987,bob,2
`
	f, err := ioutil.TempFile("", "auth_integration_test")
	f.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(f.Name())
	if err := ioutil.WriteFile(f.Name(), []byte(json), 0700); err != nil {
		t.Fatalf("unexpected error writing tokenfile: %v", err)
	}

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	m := master.New(&master.Config{
		EtcdHelper:        helper,
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		TokenAuthFile:     f.Name(),
		AuthorizationMode: "AlwaysAllow",
	})

	s := httptest.NewServer(m.Handler)
	defer s.Close()

	// TODO: also test TLS, using e.g NewUnsafeTLSTransport() and NewClientCertTLSTransport() (see pkg/client/helper.go)
	transport := http.DefaultTransport

	testCases := []struct {
		name     string
		token    string
		expected string
		succeeds bool
	}{
		{"Valid token", "abc123", "AUTHENTICATED AS alice", true},
		{"Unknown token", "456jkl", "", false},
		{"No token", "", "", false},
	}
	for _, tc := range testCases {
		req, err := http.NewRequest("GET", s.URL+"/_whoami", nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", tc.token))

		resp, err := transport.RoundTrip(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if tc.succeeds {
			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			actual := string(body)
			if tc.expected != actual {
				t.Errorf("case: %s expected: %v got: %v", tc.name, tc.expected, actual)
			}
		} else {
			if resp.StatusCode != http.StatusUnauthorized {
				t.Errorf("case: %s expected Unauthorized, got: %v", tc.name, resp.StatusCode)
			}

		}
	}
}

// Bodies for requests used in subsequent tests.
var aPod string = `
{
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "id": "a",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "a",
      "containers": [{ "name": "foo", "image": "bar/foo", }]
    }
  },
}
`
var aRC string = `
{
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "id": "a",
  "desiredState": {
    "replicas": 2,
    "replicaSelector": {"name": "a"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "a",
           "containers": [{
             "name": "foo",
             "image": "bar/foo",
           }]
         }
       },
       "labels": {"name": "a"}
      }},
  "labels": {"name": "a"}
}
`
var aService string = `
{
  "kind": "Service",
  "apiVersion": "v1beta1",
  "id": "a",
  "port": 8000,
  "labels": { "name": "a" },
  "selector": { "name": "a" }
}
`
var aMinion string = `
{
  "kind": "Minion",
  "apiVersion": "v1beta1",
  "id": "a",
  "hostIP": "10.10.10.10",
}
`

var aEvent string = `
{
  "kind": "Binding",
  "apiVersion": "v1beta1",
  "id": "a",
  "involvedObject": {
    {
      "kind": "Minion",
      "name": "a"
      "apiVersion": "v1beta1",
    }
}
`

var aBinding string = `
{
  "kind": "Binding",
  "apiVersion": "v1beta1",
  "id": "a",
  "host": "10.10.10.10",
  "podID": "a"
}
`

var aEndpoints string = `
{
  "kind": "Endpoints",
  "apiVersion": "v1beta1",
  "id": "a",
  "endpoints": ["10.10.1.1:1909"],
}
`

// Requests to try.  Each one should be forbidden or not forbidden
// depending on the authentication and authorization setup of the master.

func getTestRequests() []struct {
	verb string
	URL  string
	body string
} {
	requests := []struct {
		verb string
		URL  string
		body string
	}{
		// Normal methods on pods
		{"GET", "/api/v1beta1/pods", ""},
		{"GET", "/api/v1beta1/pods/a", ""},
		{"POST", "/api/v1beta1/pods", aPod},
		{"PUT", "/api/v1beta1/pods", aPod},
		{"GET", "/api/v1beta1/pods", ""},
		{"GET", "/api/v1beta1/pods/a", ""},
		{"DELETE", "/api/v1beta1/pods", ""},

		// Non-standard methods (not expected to work,
		// but expected to pass/fail authorization prior to
		// failing validation.
		{"PATCH", "/api/v1beta1/pods/a", ""},
		{"OPTIONS", "/api/v1beta1/pods", ""},
		{"OPTIONS", "/api/v1beta1/pods/a", ""},
		{"HEAD", "/api/v1beta1/pods", ""},
		{"HEAD", "/api/v1beta1/pods/a", ""},
		{"TRACE", "/api/v1beta1/pods", ""},
		{"TRACE", "/api/v1beta1/pods/a", ""},
		{"NOSUCHVERB", "/api/v1beta1/pods", ""},

		// Normal methods on services
		{"GET", "/api/v1beta1/services", ""},
		{"GET", "/api/v1beta1/services/a", ""},
		{"POST", "/api/v1beta1/services", aService},
		{"PUT", "/api/v1beta1/services", aService},
		{"GET", "/api/v1beta1/services", ""},
		{"GET", "/api/v1beta1/services/a", ""},
		{"DELETE", "/api/v1beta1/services", ""},

		// Normal methods on replicationControllers
		{"GET", "/api/v1beta1/replicationControllers", ""},
		{"GET", "/api/v1beta1/replicationControllers/a", ""},
		{"POST", "/api/v1beta1/replicationControllers", aRC},
		{"PUT", "/api/v1beta1/replicationControllers", aRC},
		{"GET", "/api/v1beta1/replicationControllers", ""},
		{"GET", "/api/v1beta1/replicationControllers/a", ""},
		{"DELETE", "/api/v1beta1/replicationControllers", ""},

		// Normal methods on endpoints
		{"GET", "/api/v1beta1/endpoints", ""},
		{"GET", "/api/v1beta1/endpoints/a", ""},
		{"POST", "/api/v1beta1/endpoints", aEndpoints},
		{"PUT", "/api/v1beta1/endpoints", aEndpoints},
		{"GET", "/api/v1beta1/endpoints", ""},
		{"GET", "/api/v1beta1/endpoints/a", ""},
		{"DELETE", "/api/v1beta1/endpoints", ""},

		// Normal methods on minions
		{"GET", "/api/v1beta1/minions", ""},
		{"GET", "/api/v1beta1/minions/a", ""},
		{"POST", "/api/v1beta1/minions", aMinion},
		{"PUT", "/api/v1beta1/minions", aMinion},
		{"GET", "/api/v1beta1/minions", ""},
		{"GET", "/api/v1beta1/minions/a", ""},
		{"DELETE", "/api/v1beta1/minions", ""},

		// Normal methods on events
		{"GET", "/api/v1beta1/events", ""},
		{"GET", "/api/v1beta1/events/a", ""},
		{"POST", "/api/v1beta1/events", aEvent},
		{"PUT", "/api/v1beta1/events", aEvent},
		{"GET", "/api/v1beta1/events", ""},
		{"GET", "/api/v1beta1/events/a", ""},
		{"DELETE", "/api/v1beta1/events", ""},

		// Normal methods on bindings
		{"GET", "/api/v1beta1/events", ""},
		{"GET", "/api/v1beta1/events/a", ""},
		{"POST", "/api/v1beta1/events", aBinding},
		{"PUT", "/api/v1beta1/events", aBinding},
		{"GET", "/api/v1beta1/events", ""},
		{"GET", "/api/v1beta1/events/a", ""},
		{"DELETE", "/api/v1beta1/events", ""},

		// Non-existent object type.
		{"GET", "/api/v1beta1/foo", ""},
		{"GET", "/api/v1beta1/foo/a", ""},
		{"POST", "/api/v1beta1/foo", `{"foo": "foo"}`},
		{"PUT", "/api/v1beta1/foo", `{"foo": "foo"}`},
		{"GET", "/api/v1beta1/foo", ""},
		{"GET", "/api/v1beta1/foo/a", ""},
		{"DELETE", "/api/v1beta1/foo", ""},

		// Operations
		{"GET", "/api/v1beta1/operations", ""},
		{"GET", "/api/v1beta1/operations/1234567890", ""},

		// Special verbs on pods
		{"GET", "/api/v1beta1/proxy/pods/a", ""},
		{"GET", "/api/v1beta1/redirect/pods/a", ""},
		// TODO: test .../watch/..., which doesn't end before the test timeout.

		// Non-object endpoints
		{"GET", "/", ""},
		{"GET", "/healthz", ""},
		{"GET", "/versions", ""},
	}
	return requests
}

// The TestAuthMode* tests tests a large number of URLs and checks that they
// are FORBIDDEN or not, depending on the mode.  They do not attempt to do
// detailed verification of behaviour beyond authorization.  They are not
// fuzz tests.
//
// TODO(etune): write a fuzz test of the REST API.
func TestAuthModeAlwaysAllow(t *testing.T) {
	deleteAllEtcdKeys()

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	m := master.New(&master.Config{
		EtcdHelper:        helper,
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		AuthorizationMode: "AlwaysAllow",
	})

	s := httptest.NewServer(m.Handler)
	defer s.Close()
	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		resp, err := transport.RoundTrip(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.StatusCode == http.StatusForbidden {
			t.Errorf("Expected status other than Forbidden")
		}
	}
}

func TestAuthModeAlwaysDeny(t *testing.T) {
	deleteAllEtcdKeys()

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	m := master.New(&master.Config{
		EtcdHelper:        helper,
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		AuthorizationMode: "AlwaysDeny",
	})

	s := httptest.NewServer(m.Handler)
	defer s.Close()
	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		resp, err := transport.RoundTrip(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.StatusCode != http.StatusForbidden {
			t.Errorf("Expected status Forbidden but got status %v", resp.Status)
		}
	}
}
