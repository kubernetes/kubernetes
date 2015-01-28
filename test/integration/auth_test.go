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
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator/bearertoken"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer/abac"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/admit"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/token/tokentest"
)

func init() {
	requireEtcd()
}

const (
	AliceToken   string = "abc123" // username: alice.  Present in token file.
	BobToken     string = "xyz987" // username: bob.  Present in token file.
	UnknownToken string = "qwerty" // Not present in token file.
)

func getTestTokenAuth() authenticator.Request {
	tokenAuthenticator := tokentest.New()
	tokenAuthenticator.Tokens[AliceToken] = &user.DefaultInfo{Name: "alice", UID: "1"}
	tokenAuthenticator.Tokens[BobToken] = &user.DefaultInfo{Name: "bob", UID: "2"}
	return bearertoken.New(tokenAuthenticator)
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
  "kind": "Event",
  "apiVersion": "v1beta1",
  "namespace": "default",
  "id": "a",
  "involvedObject": {
    "kind": "Minion",
    "name": "a",
    "namespace": "default",
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

// To ensure that a POST completes before a dependent GET, set a timeout.
var timeoutFlag = "?timeout=60s"

// Requests to try.  Each one should be forbidden or not forbidden
// depending on the authentication and authorization setup of the master.
var code200 = map[int]bool{200: true}
var code400 = map[int]bool{400: true}
var code403 = map[int]bool{403: true}
var code404 = map[int]bool{404: true}
var code405 = map[int]bool{405: true}
var code409 = map[int]bool{409: true}
var code422 = map[int]bool{422: true}
var code500 = map[int]bool{500: true}

func getTestRequests() []struct {
	verb        string
	URL         string
	body        string
	statusCodes map[int]bool // allowed status codes.
} {
	requests := []struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // Set of expected resp.StatusCode if all goes well.
	}{
		// Normal methods on pods
		{"GET", "/api/v1beta1/pods", "", code200},
		{"POST", "/api/v1beta1/pods" + timeoutFlag, aPod, code200},
		{"PUT", "/api/v1beta1/pods/a" + timeoutFlag, aPod, code500}, // See #2114 about why 500
		{"GET", "/api/v1beta1/pods", "", code200},
		{"GET", "/api/v1beta1/pods/a", "", code200},
		{"DELETE", "/api/v1beta1/pods/a" + timeoutFlag, "", code200},

		// Non-standard methods (not expected to work,
		// but expected to pass/fail authorization prior to
		// failing validation.
		{"PATCH", "/api/v1beta1/pods/a", "", code405},
		{"OPTIONS", "/api/v1beta1/pods", "", code405},
		{"OPTIONS", "/api/v1beta1/pods/a", "", code405},
		{"HEAD", "/api/v1beta1/pods", "", code405},
		{"HEAD", "/api/v1beta1/pods/a", "", code405},
		{"TRACE", "/api/v1beta1/pods", "", code405},
		{"TRACE", "/api/v1beta1/pods/a", "", code405},
		{"NOSUCHVERB", "/api/v1beta1/pods", "", code405},

		// Normal methods on services
		{"GET", "/api/v1beta1/services", "", code200},
		{"POST", "/api/v1beta1/services" + timeoutFlag, aService, code200},
		{"PUT", "/api/v1beta1/services/a" + timeoutFlag, aService, code409}, // TODO: GET and put back server-provided fields to avoid a 422
		{"GET", "/api/v1beta1/services", "", code200},
		{"GET", "/api/v1beta1/services/a", "", code200},
		{"DELETE", "/api/v1beta1/services/a" + timeoutFlag, "", code200},

		// Normal methods on replicationControllers
		{"GET", "/api/v1beta1/replicationControllers", "", code200},
		{"POST", "/api/v1beta1/replicationControllers" + timeoutFlag, aRC, code200},
		{"PUT", "/api/v1beta1/replicationControllers/a" + timeoutFlag, aRC, code409}, // See #2115 about why 409
		{"GET", "/api/v1beta1/replicationControllers", "", code200},
		{"GET", "/api/v1beta1/replicationControllers/a", "", code200},
		{"DELETE", "/api/v1beta1/replicationControllers/a" + timeoutFlag, "", code200},

		// Normal methods on endpoints
		{"GET", "/api/v1beta1/endpoints", "", code200},
		{"POST", "/api/v1beta1/endpoints" + timeoutFlag, aEndpoints, code200},
		{"PUT", "/api/v1beta1/endpoints/a" + timeoutFlag, aEndpoints, code200},
		{"GET", "/api/v1beta1/endpoints", "", code200},
		{"GET", "/api/v1beta1/endpoints/a", "", code200},
		{"DELETE", "/api/v1beta1/endpoints/a" + timeoutFlag, "", code405},

		// Normal methods on minions
		{"GET", "/api/v1beta1/minions", "", code200},
		{"POST", "/api/v1beta1/minions" + timeoutFlag, aMinion, code200},
		{"PUT", "/api/v1beta1/minions/a" + timeoutFlag, aMinion, code422}, // TODO: GET and put back server-provided fields to avoid a 422
		{"GET", "/api/v1beta1/minions", "", code200},
		{"GET", "/api/v1beta1/minions/a", "", code200},
		{"DELETE", "/api/v1beta1/minions/a" + timeoutFlag, "", code200},

		// Normal methods on events
		{"GET", "/api/v1beta1/events", "", code200},
		{"POST", "/api/v1beta1/events" + timeoutFlag, aEvent, code200},
		{"PUT", "/api/v1beta1/events/a" + timeoutFlag, aEvent, code405},
		{"GET", "/api/v1beta1/events", "", code200},
		{"GET", "/api/v1beta1/events", "", code200},
		{"GET", "/api/v1beta1/events/a", "", code200},
		{"DELETE", "/api/v1beta1/events/a" + timeoutFlag, "", code200},

		// Normal methods on bindings
		{"GET", "/api/v1beta1/bindings", "", code405},              // Bindings are write-only
		{"POST", "/api/v1beta1/pods" + timeoutFlag, aPod, code200}, // Need a pod to bind or you get a 404
		{"POST", "/api/v1beta1/bindings" + timeoutFlag, aBinding, code200},
		{"PUT", "/api/v1beta1/bindings/a" + timeoutFlag, aBinding, code405},
		{"GET", "/api/v1beta1/bindings", "", code405},
		{"GET", "/api/v1beta1/bindings/a", "", code405}, // No bindings instances
		{"DELETE", "/api/v1beta1/bindings/a" + timeoutFlag, "", code405},

		// Non-existent object type.
		{"GET", "/api/v1beta1/foo", "", code404},
		{"POST", "/api/v1beta1/foo", `{"foo": "foo"}`, code404},
		{"PUT", "/api/v1beta1/foo/a", `{"foo": "foo"}`, code404},
		{"GET", "/api/v1beta1/foo", "", code404},
		{"GET", "/api/v1beta1/foo/a", "", code404},
		{"DELETE", "/api/v1beta1/foo" + timeoutFlag, "", code404},

		// Operations
		{"GET", "/api/v1beta1/operations", "", code200},
		{"GET", "/api/v1beta1/operations/1234567890", "", code404},

		// Special verbs on nodes
		// TODO: Will become 405 once these are converted to go-restful
		{"GET", "/api/v1beta1/proxy/minions/a", "", code404},
		{"GET", "/api/v1beta1/redirect/minions/a", "", code404},
		// TODO: test .../watch/..., which doesn't end before the test timeout.
		// TODO: figure out how to create a minion so that it can successfully proxy/redirect.

		// Non-object endpoints
		{"GET", "/", "", code200},
		{"GET", "/api", "", code200},
		{"GET", "/healthz", "", code200},
		{"GET", "/version", "", code200},
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

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		Client:            client.NewOrDie(&client.Config{Host: s.URL}),
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		Authorizer:        apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				b, _ := ioutil.ReadAll(resp.Body)
				t.Errorf("Body: %v", string(b))
			}
		}()
	}
}

func TestAuthModeAlwaysDeny(t *testing.T) {
	deleteAllEtcdKeys()

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		Client:            client.NewOrDie(&client.Config{Host: s.URL}),
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		Authorizer:        apiserver.NewAlwaysDenyAuthorizer(),
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp.StatusCode != http.StatusForbidden {
				t.Errorf("Expected status Forbidden but got status %v", resp.Status)
			}
		}()
	}
}

// Inject into master an authorizer that uses user info.
// TODO(etune): remove this test once a more comprehensive built-in authorizer is implemented.
type allowAliceAuthorizer struct{}

func (allowAliceAuthorizer) Authorize(a authorizer.Attributes) error {
	if a.GetUserName() == "alice" {
		return nil
	}
	return errors.New("I can't allow that.  Go ask alice.")
}

// TestAliceNotForbiddenOrUnauthorized tests a user who is known to
// the authentication system and authorized to do any actions.
func TestAliceNotForbiddenOrUnauthorized(t *testing.T) {

	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		Client:            client.NewOrDie(&client.Config{Host: s.URL}),
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		Authenticator:     getTestTokenAuth(),
		Authorizer:        allowAliceAuthorizer{},
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		token := AliceToken
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				b, _ := ioutil.ReadAll(resp.Body)
				t.Errorf("Body: %v", string(b))
			}
		}()
	}
}

// TestBobIsForbidden tests that a user who is known to
// the authentication system but not authorized to do any actions
// should receive "Forbidden".
func TestBobIsForbidden(t *testing.T) {
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		Client:            client.NewOrDie(&client.Config{Host: s.URL}),
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		Authenticator:     getTestTokenAuth(),
		Authorizer:        allowAliceAuthorizer{},
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		token := BobToken
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// Expect all of bob's actions to return Forbidden
			if resp.StatusCode != http.StatusForbidden {
				t.Errorf("Expected not status Forbidden, but got %s", resp.Status)
			}
		}()
	}
}

// TestUnknownUserIsUnauthorized tests that a user who is unknown
// to the authentication system get status code "Unauthorized".
// An authorization module is installed in this scenario for integration
// test purposes, but requests aren't expected to reach it.
func TestUnknownUserIsUnauthorized(t *testing.T) {
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		Client:            client.NewOrDie(&client.Config{Host: s.URL}),
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		Authenticator:     getTestTokenAuth(),
		Authorizer:        allowAliceAuthorizer{},
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		token := UnknownToken
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// Expect all of unauthenticated user's request to be "Unauthorized"
			if resp.StatusCode != http.StatusUnauthorized {
				t.Errorf("Expected status %v, but got %v", http.StatusUnauthorized, resp.StatusCode)
				b, _ := ioutil.ReadAll(resp.Body)
				t.Errorf("Body: %v", string(b))
			}
		}()
	}
}

func newAuthorizerWithContents(t *testing.T, contents string) authorizer.Authorizer {
	f, err := ioutil.TempFile("", "auth_test")
	if err != nil {
		t.Fatalf("unexpected error creating policyfile: %v", err)
	}
	f.Close()
	defer os.Remove(f.Name())

	if err := ioutil.WriteFile(f.Name(), []byte(contents), 0700); err != nil {
		t.Fatalf("unexpected error writing policyfile: %v", err)
	}

	pl, err := abac.NewFromFile(f.Name())
	if err != nil {
		t.Fatalf("unexpected error creating authorizer from policyfile: %v", err)
	}
	return pl
}

// TestNamespaceAuthorization tests that authorization can be controlled
// by namespace.
func TestNamespaceAuthorization(t *testing.T) {
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	a := newAuthorizerWithContents(t, `{"namespace": "foo"}
`)

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		Client:            client.NewOrDie(&client.Config{Host: s.URL}),
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		Authenticator:     getTestTokenAuth(),
		Authorizer:        a,
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{

		{"POST", "/api/v1beta1/pods" + timeoutFlag + "&namespace=foo", aPod, code200},
		{"GET", "/api/v1beta1/pods?namespace=foo", "", code200},
		{"GET", "/api/v1beta1/pods/a?namespace=foo", "", code200},
		{"DELETE", "/api/v1beta1/pods/a" + timeoutFlag + "&namespace=foo", "", code200},

		{"POST", "/api/v1beta1/pods" + timeoutFlag + "&namespace=bar", aPod, code403},
		{"GET", "/api/v1beta1/pods?namespace=bar", "", code403},
		{"GET", "/api/v1beta1/pods/a?namespace=bar", "", code403},
		{"DELETE", "/api/v1beta1/pods/a" + timeoutFlag + "&namespace=bar", "", code403},

		{"POST", "/api/v1beta1/pods" + timeoutFlag, aPod, code403},
		{"GET", "/api/v1beta1/pods", "", code403},
		{"GET", "/api/v1beta1/pods/a", "", code403},
		{"DELETE", "/api/v1beta1/pods/a" + timeoutFlag, "", code403},
	}

	for _, r := range requests {
		token := BobToken
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				b, _ := ioutil.ReadAll(resp.Body)
				t.Errorf("Body: %v", string(b))
			}
		}()
	}
}

// TestKindAuthorization tests that authorization can be controlled
// by namespace.
func TestKindAuthorization(t *testing.T) {
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	a := newAuthorizerWithContents(t, `{"kind": "services"}
`)

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		Client:            client.NewOrDie(&client.Config{Host: s.URL}),
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		Authenticator:     getTestTokenAuth(),
		Authorizer:        a,
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{
		{"POST", "/api/v1beta1/services" + timeoutFlag, aService, code200},
		{"GET", "/api/v1beta1/services", "", code200},
		{"GET", "/api/v1beta1/services/a", "", code200},
		{"DELETE", "/api/v1beta1/services/a" + timeoutFlag, "", code200},

		{"POST", "/api/v1beta1/pods" + timeoutFlag, aPod, code403},
		{"GET", "/api/v1beta1/pods", "", code403},
		{"GET", "/api/v1beta1/pods/a", "", code403},
		{"DELETE", "/api/v1beta1/pods/a" + timeoutFlag, "", code403},
	}

	for _, r := range requests {
		token := BobToken
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		{
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				b, _ := ioutil.ReadAll(resp.Body)
				t.Errorf("Body: %v", string(b))
			}
		}
	}
}

// TestReadOnlyAuthorization tests that authorization can be controlled
// by namespace.
func TestReadOnlyAuthorization(t *testing.T) {
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	a := newAuthorizerWithContents(t, `{"readonly": true}
`)

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		Client:            client.NewOrDie(&client.Config{Host: s.URL}),
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		APIPrefix:         "/api",
		Authenticator:     getTestTokenAuth(),
		Authorizer:        a,
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{
		{"POST", "/api/v1beta1/pods", aPod, code403},
		{"GET", "/api/v1beta1/pods", "", code200},
		{"GET", "/api/v1beta1/pods/a", "", code404},
	}

	for _, r := range requests {
		token := BobToken
		t.Logf("case %v", r)
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				b, _ := ioutil.ReadAll(resp.Body)
				t.Errorf("Body: %v", string(b))
			}
		}()
	}
}
