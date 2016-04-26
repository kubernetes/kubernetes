// +build integration,!no-etcd

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authenticator/bearertoken"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/tokentest"
	"k8s.io/kubernetes/test/integration/framework"
)

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

func path(resource, namespace, name string) string {
	return testapi.Default.ResourcePath(resource, namespace, name)
}

func pathWithPrefix(prefix, resource, namespace, name string) string {
	return testapi.Default.ResourcePathWithPrefix(prefix, resource, namespace, name)
}

func timeoutPath(resource, namespace, name string) string {
	return addTimeoutFlag(testapi.Default.ResourcePath(resource, namespace, name))
}

// Bodies for requests used in subsequent tests.
var aPod string = `
{
  "kind": "Pod",
  "apiVersion": "` + testapi.Default.GroupVersion().String() + `",
  "metadata": {
    "name": "a",
    "creationTimestamp": null%s
  },
  "spec": {
    "containers": [
      {
        "name": "foo",
        "image": "bar/foo"
      }
    ]
  }
}
`
var aRC string = `
{
  "kind": "ReplicationController",
  "apiVersion": "` + testapi.Default.GroupVersion().String() + `",
  "metadata": {
    "name": "a",
    "labels": {
      "name": "a"
    }%s
  },
  "spec": {
    "replicas": 2,
    "selector": {
      "name": "a"
    },
    "template": {
      "metadata": {
        "labels": {
          "name": "a"
        }
      },
      "spec": {
        "containers": [
          {
            "name": "foo",
            "image": "bar/foo"
          }
        ]
      }
    }
  }
}
`
var aService string = `
{
  "kind": "Service",
  "apiVersion": "` + testapi.Default.GroupVersion().String() + `",
  "metadata": {
    "name": "a",
    "labels": {
      "name": "a"
    }%s
  },
  "spec": {
    "ports": [
      {
        "protocol": "TCP",
        "port": 8000
      }
    ],
    "selector": {
      "name": "a"
    },
    "clusterIP": "10.0.0.100"
  }
}
`
var aNode string = `
{
  "kind": "Node",
  "apiVersion": "` + testapi.Default.GroupVersion().String() + `",
  "metadata": {
    "name": "a"%s
  },
  "spec": {
    "externalID": "external"
  }
}
`
var aEvent string = `
{
  "kind": "Event",
  "apiVersion": "` + testapi.Default.GroupVersion().String() + `",
  "metadata": {
    "name": "a"%s
  },
  "involvedObject": {
    "kind": "Pod",
    "namespace": "default",
    "name": "a",
    "apiVersion": "v1"
  }
}
`

var aBinding string = `
{
  "kind": "Binding",
  "apiVersion": "` + testapi.Default.GroupVersion().String() + `",
  "metadata": {
    "name": "a"%s
  },
  "target": {
    "name": "10.10.10.10"
  }
}
`

var emptyEndpoints string = `
{
  "kind": "Endpoints",
  "apiVersion": "v1",
  "metadata": {
    "name": "a"%s
  }
}
`

var aEndpoints string = `
{
  "kind": "Endpoints",
  "apiVersion": "` + testapi.Default.GroupVersion().String() + `",
  "metadata": {
    "name": "a"%s
  },
  "subsets": [
    {
      "addresses": [
        {
          "ip": "10.10.1.1"
        }
      ],
      "ports": [
        {
          "port": 1909,
          "protocol": "TCP"
        }
      ]
    }
  ]
}
`

var deleteNow string = `
{
  "kind": "DeleteOptions",
  "apiVersion": "` + testapi.Default.GroupVersion().String() + `",
  "gracePeriodSeconds": 0%s
}
`

// Requests to try.  Each one should be forbidden or not forbidden
// depending on the authentication and authorization setup of the master.
var code200 = map[int]bool{200: true}
var code201 = map[int]bool{201: true}
var code400 = map[int]bool{400: true}
var code403 = map[int]bool{403: true}
var code404 = map[int]bool{404: true}
var code405 = map[int]bool{405: true}
var code409 = map[int]bool{409: true}
var code422 = map[int]bool{422: true}
var code500 = map[int]bool{500: true}
var code503 = map[int]bool{503: true}

// To ensure that a POST completes before a dependent GET, set a timeout.
func addTimeoutFlag(URLString string) string {
	u, _ := url.Parse(URLString)
	values := u.Query()
	values.Set("timeout", "60s")
	u.RawQuery = values.Encode()
	return u.String()
}

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
		{"GET", path("pods", "", ""), "", code200},
		{"GET", path("pods", api.NamespaceDefault, ""), "", code200},
		{"POST", timeoutPath("pods", api.NamespaceDefault, ""), aPod, code201},
		{"PUT", timeoutPath("pods", api.NamespaceDefault, "a"), aPod, code200},
		{"GET", path("pods", api.NamespaceDefault, "a"), "", code200},
		// GET and POST for /exec should return Bad Request (400) since the pod has not been assigned a node yet.
		{"GET", path("pods", api.NamespaceDefault, "a") + "/exec", "", code400},
		{"POST", path("pods", api.NamespaceDefault, "a") + "/exec", "", code400},
		// PUT for /exec should return Method Not Allowed (405).
		{"PUT", path("pods", api.NamespaceDefault, "a") + "/exec", "", code405},
		// GET and POST for /portforward should return Bad Request (400) since the pod has not been assigned a node yet.
		{"GET", path("pods", api.NamespaceDefault, "a") + "/portforward", "", code400},
		{"POST", path("pods", api.NamespaceDefault, "a") + "/portforward", "", code400},
		// PUT for /portforward should return Method Not Allowed (405).
		{"PUT", path("pods", api.NamespaceDefault, "a") + "/portforward", "", code405},
		{"PATCH", path("pods", api.NamespaceDefault, "a"), "{%v}", code200},
		{"DELETE", timeoutPath("pods", api.NamespaceDefault, "a"), deleteNow, code200},

		// Non-standard methods (not expected to work,
		// but expected to pass/fail authorization prior to
		// failing validation.
		{"OPTIONS", path("pods", api.NamespaceDefault, ""), "", code405},
		{"OPTIONS", path("pods", api.NamespaceDefault, "a"), "", code405},
		{"HEAD", path("pods", api.NamespaceDefault, ""), "", code405},
		{"HEAD", path("pods", api.NamespaceDefault, "a"), "", code405},
		{"TRACE", path("pods", api.NamespaceDefault, ""), "", code405},
		{"TRACE", path("pods", api.NamespaceDefault, "a"), "", code405},
		{"NOSUCHVERB", path("pods", api.NamespaceDefault, ""), "", code405},

		// Normal methods on services
		{"GET", path("services", "", ""), "", code200},
		{"GET", path("services", api.NamespaceDefault, ""), "", code200},
		{"POST", timeoutPath("services", api.NamespaceDefault, ""), aService, code201},
		// Create an endpoint for the service (this is done automatically by endpoint controller
		// whenever a service is created, but this test does not run that controller)
		{"POST", timeoutPath("endpoints", api.NamespaceDefault, ""), emptyEndpoints, code201},
		// Should return service unavailable when endpoint.subset is empty.
		{"GET", pathWithPrefix("proxy", "services", api.NamespaceDefault, "a") + "/", "", code503},
		{"PUT", timeoutPath("services", api.NamespaceDefault, "a"), aService, code200},
		{"GET", path("services", api.NamespaceDefault, "a"), "", code200},
		{"DELETE", timeoutPath("endpoints", api.NamespaceDefault, "a"), "", code200},
		{"DELETE", timeoutPath("services", api.NamespaceDefault, "a"), "", code200},

		// Normal methods on replicationControllers
		{"GET", path("replicationControllers", "", ""), "", code200},
		{"GET", path("replicationControllers", api.NamespaceDefault, ""), "", code200},
		{"POST", timeoutPath("replicationControllers", api.NamespaceDefault, ""), aRC, code201},
		{"PUT", timeoutPath("replicationControllers", api.NamespaceDefault, "a"), aRC, code200},
		{"GET", path("replicationControllers", api.NamespaceDefault, "a"), "", code200},
		{"DELETE", timeoutPath("replicationControllers", api.NamespaceDefault, "a"), "", code200},

		// Normal methods on endpoints
		{"GET", path("endpoints", "", ""), "", code200},
		{"GET", path("endpoints", api.NamespaceDefault, ""), "", code200},
		{"POST", timeoutPath("endpoints", api.NamespaceDefault, ""), aEndpoints, code201},
		{"PUT", timeoutPath("endpoints", api.NamespaceDefault, "a"), aEndpoints, code200},
		{"GET", path("endpoints", api.NamespaceDefault, "a"), "", code200},
		{"DELETE", timeoutPath("endpoints", api.NamespaceDefault, "a"), "", code200},

		// Normal methods on nodes
		{"GET", path("nodes", "", ""), "", code200},
		{"POST", timeoutPath("nodes", "", ""), aNode, code201},
		{"PUT", timeoutPath("nodes", "", "a"), aNode, code200},
		{"GET", path("nodes", "", "a"), "", code200},
		{"DELETE", timeoutPath("nodes", "", "a"), "", code200},

		// Normal methods on events
		{"GET", path("events", "", ""), "", code200},
		{"GET", path("events", api.NamespaceDefault, ""), "", code200},
		{"POST", timeoutPath("events", api.NamespaceDefault, ""), aEvent, code201},
		{"PUT", timeoutPath("events", api.NamespaceDefault, "a"), aEvent, code200},
		{"GET", path("events", api.NamespaceDefault, "a"), "", code200},
		{"DELETE", timeoutPath("events", api.NamespaceDefault, "a"), "", code200},

		// Normal methods on bindings
		{"GET", path("bindings", api.NamespaceDefault, ""), "", code405},
		{"POST", timeoutPath("pods", api.NamespaceDefault, ""), aPod, code201}, // Need a pod to bind or you get a 404
		{"POST", timeoutPath("bindings", api.NamespaceDefault, ""), aBinding, code201},
		{"PUT", timeoutPath("bindings", api.NamespaceDefault, "a"), aBinding, code404},
		{"GET", path("bindings", api.NamespaceDefault, "a"), "", code404}, // No bindings instances
		{"DELETE", timeoutPath("bindings", api.NamespaceDefault, "a"), "", code404},

		// Non-existent object type.
		{"GET", path("foo", "", ""), "", code404},
		{"POST", path("foo", api.NamespaceDefault, ""), `{"foo": "foo"}`, code404},
		{"PUT", path("foo", api.NamespaceDefault, "a"), `{"foo": "foo"}`, code404},
		{"GET", path("foo", api.NamespaceDefault, "a"), "", code404},
		{"DELETE", timeoutPath("foo", api.NamespaceDefault, ""), "", code404},

		// Special verbs on nodes
		{"GET", pathWithPrefix("proxy", "nodes", api.NamespaceDefault, "a"), "", code404},
		{"GET", pathWithPrefix("redirect", "nodes", api.NamespaceDefault, "a"), "", code404},
		// TODO: test .../watch/..., which doesn't end before the test timeout.
		// TODO: figure out how to create a node so that it can successfully proxy/redirect.

		// Non-object endpoints
		{"GET", "/", "", code200},
		{"GET", "/api", "", code200},
		{"GET", "/healthz", "", code200},
		{"GET", "/version", "", code200},
		{"GET", "/invalidURL", "", code404},
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
	framework.DeleteAllEtcdKeys()

	// Set up a master
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	transport := http.DefaultTransport
	previousResourceVersion := make(map[string]float64)

	for _, r := range getTestRequests() {
		var bodyStr string
		if r.body != "" {
			sub := ""
			if r.verb == "PUT" {
				// For update operations, insert previous resource version
				if resVersion := previousResourceVersion[getPreviousResourceVersionKey(r.URL, "")]; resVersion != 0 {
					sub += fmt.Sprintf(",\r\n\"resourceVersion\": \"%v\"", resVersion)
				}
				namespace := "default"
				sub += fmt.Sprintf(",\r\n\"namespace\": %q", namespace)
			}
			bodyStr = fmt.Sprintf(r.body, sub)
		}
		r.body = bodyStr
		bodyBytes := bytes.NewReader([]byte(bodyStr))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Logf("case %v", r)
			t.Fatalf("unexpected error: %v", err)
		}
		if r.verb == "PATCH" {
			req.Header.Set("Content-Type", "application/merge-patch+json")
		}
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			b, _ := ioutil.ReadAll(resp.Body)
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Logf("case %v", r)
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				t.Errorf("Body: %v", string(b))
			} else {
				if r.verb == "POST" {
					// For successful create operations, extract resourceVersion
					id, currentResourceVersion, err := parseResourceVersion(b)
					if err == nil {
						key := getPreviousResourceVersionKey(r.URL, id)
						previousResourceVersion[key] = currentResourceVersion
					} else {
						t.Logf("error in trying to extract resource version: %s", err)
					}
				}
			}
		}()
	}
}

func parseResourceVersion(response []byte) (string, float64, error) {
	var resultBodyMap map[string]interface{}
	err := json.Unmarshal(response, &resultBodyMap)
	if err != nil {
		return "", 0, fmt.Errorf("unexpected error unmarshaling resultBody: %v", err)
	}
	metadata, ok := resultBodyMap["metadata"].(map[string]interface{})
	if !ok {
		return "", 0, fmt.Errorf("unexpected error, metadata not found in JSON response: %v", string(response))
	}
	id, ok := metadata["name"].(string)
	if !ok {
		return "", 0, fmt.Errorf("unexpected error, id not found in JSON response: %v", string(response))
	}
	resourceVersionString, ok := metadata["resourceVersion"].(string)
	if !ok {
		return "", 0, fmt.Errorf("unexpected error, resourceVersion not found in JSON response: %v", string(response))
	}
	resourceVersion, err := strconv.ParseFloat(resourceVersionString, 64)
	if err != nil {
		return "", 0, fmt.Errorf("unexpected error, could not parse resourceVersion as float64, err: %s. JSON response: %v", err, string(response))
	}
	return id, resourceVersion, nil
}

func getPreviousResourceVersionKey(url, id string) string {
	baseUrl := strings.Split(url, "?")[0]
	key := baseUrl
	if id != "" {
		key = fmt.Sprintf("%s/%v", baseUrl, id)
	}
	return key
}

func TestAuthModeAlwaysDeny(t *testing.T) {
	framework.DeleteAllEtcdKeys()

	// Set up a master
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authorizer = apiserver.NewAlwaysDenyAuthorizer()
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Logf("case %v", r)
			t.Fatalf("unexpected error: %v", err)
		}
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			if resp.StatusCode != http.StatusForbidden {
				t.Logf("case %v", r)
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

	framework.DeleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = getTestTokenAuth()
	masterConfig.Authorizer = allowAliceAuthorizer{}
	masterConfig.AdmissionControl = admit.NewAlwaysAdmit()
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	previousResourceVersion := make(map[string]float64)
	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		token := AliceToken
		var bodyStr string
		if r.body != "" {
			sub := ""
			if r.verb == "PUT" {
				// For update operations, insert previous resource version
				if resVersion := previousResourceVersion[getPreviousResourceVersionKey(r.URL, "")]; resVersion != 0 {
					sub += fmt.Sprintf(",\r\n\"resourceVersion\": \"%v\"", resVersion)
				}
				namespace := "default"
				sub += fmt.Sprintf(",\r\n\"namespace\": %q", namespace)
			}
			bodyStr = fmt.Sprintf(r.body, sub)
		}
		r.body = bodyStr
		bodyBytes := bytes.NewReader([]byte(bodyStr))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		if r.verb == "PATCH" {
			req.Header.Set("Content-Type", "application/merge-patch+json")
		}

		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			b, _ := ioutil.ReadAll(resp.Body)
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Logf("case %v", r)
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				t.Errorf("Body: %v", string(b))
			} else {
				if r.verb == "POST" {
					// For successful create operations, extract resourceVersion
					id, currentResourceVersion, err := parseResourceVersion(b)
					if err == nil {
						key := getPreviousResourceVersionKey(r.URL, id)
						previousResourceVersion[key] = currentResourceVersion
					}
				}
			}

		}()
	}
}

// TestBobIsForbidden tests that a user who is known to
// the authentication system but not authorized to do any actions
// should receive "Forbidden".
func TestBobIsForbidden(t *testing.T) {
	framework.DeleteAllEtcdKeys()

	// This file has alice and bob in it.
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = getTestTokenAuth()
	masterConfig.Authorizer = allowAliceAuthorizer{}
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		token := BobToken
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
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			// Expect all of bob's actions to return Forbidden
			if resp.StatusCode != http.StatusForbidden {
				t.Logf("case %v", r)
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
	framework.DeleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = getTestTokenAuth()
	masterConfig.Authorizer = allowAliceAuthorizer{}
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	transport := http.DefaultTransport

	for _, r := range getTestRequests() {
		token := UnknownToken
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
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			// Expect all of unauthenticated user's request to be "Unauthorized"
			if resp.StatusCode != http.StatusUnauthorized {
				t.Logf("case %v", r)
				t.Errorf("Expected status %v, but got %v", http.StatusUnauthorized, resp.StatusCode)
				b, _ := ioutil.ReadAll(resp.Body)
				t.Errorf("Body: %v", string(b))
			}
		}()
	}
}

type impersonateAuthorizer struct{}

// alice can't act as anyone and bob can't do anything but act-as someone
func (impersonateAuthorizer) Authorize(a authorizer.Attributes) error {
	// alice can impersonate service accounts and do other actions
	if a.GetUserName() == "alice" && a.GetVerb() == "impersonate" && a.GetResource() == "serviceaccounts" {
		return nil
	}
	if a.GetUserName() == "alice" && a.GetVerb() != "impersonate" {
		return nil
	}
	// bob can impersonate anyone, but that it
	if a.GetUserName() == "bob" && a.GetVerb() == "impersonate" {
		return nil
	}
	// service accounts can do everything
	if strings.HasPrefix(a.GetUserName(), serviceaccount.ServiceAccountUsernamePrefix) {
		return nil
	}

	return errors.New("I can't allow that.  Go ask alice.")
}

func TestImpersonateIsForbidden(t *testing.T) {
	framework.DeleteAllEtcdKeys()

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = getTestTokenAuth()
	masterConfig.Authorizer = impersonateAuthorizer{}
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	transport := http.DefaultTransport

	// bob can't perform actions himself
	for _, r := range getTestRequests() {
		token := BobToken
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
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			// Expect all of bob's actions to return Forbidden
			if resp.StatusCode != http.StatusForbidden {
				t.Logf("case %v", r)
				t.Errorf("Expected not status Forbidden, but got %s", resp.Status)
			}
		}()
	}

	// bob can impersonate alice to do other things
	for _, r := range getTestRequests() {
		token := BobToken
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		req.Header.Set("Impersonate-User", "alice")
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			// Expect all the requests to be allowed, don't care what they actually do
			if resp.StatusCode == http.StatusForbidden {
				t.Logf("case %v", r)
				t.Errorf("Expected status not %v, but got %v", http.StatusForbidden, resp.StatusCode)
			}
		}()
	}

	// alice can't impersonate bob
	for _, r := range getTestRequests() {
		token := AliceToken
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		req.Header.Set("Impersonate-User", "bob")

		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			// Expect all of bob's actions to return Forbidden
			if resp.StatusCode != http.StatusForbidden {
				t.Logf("case %v", r)
				t.Errorf("Expected not status Forbidden, but got %s", resp.Status)
			}
		}()
	}

	// alice can impersonate a service account
	for _, r := range getTestRequests() {
		token := BobToken
		bodyBytes := bytes.NewReader([]byte(r.body))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		req.Header.Set("Impersonate-User", serviceaccount.MakeUsername("default", "default"))
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			// Expect all the requests to be allowed, don't care what they actually do
			if resp.StatusCode == http.StatusForbidden {
				t.Logf("case %v", r)
				t.Errorf("Expected status not %v, but got %v", http.StatusForbidden, resp.StatusCode)
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

type trackingAuthorizer struct {
	requestAttributes []authorizer.Attributes
}

func (a *trackingAuthorizer) Authorize(attributes authorizer.Attributes) error {
	a.requestAttributes = append(a.requestAttributes, attributes)
	return nil
}

// TestAuthorizationAttributeDetermination tests that authorization attributes are built correctly
func TestAuthorizationAttributeDetermination(t *testing.T) {
	framework.DeleteAllEtcdKeys()

	trackingAuthorizer := &trackingAuthorizer{}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = getTestTokenAuth()
	masterConfig.Authorizer = trackingAuthorizer
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	transport := http.DefaultTransport

	requests := map[string]struct {
		verb               string
		URL                string
		expectedAttributes authorizer.Attributes
	}{
		"prefix/version/resource":        {"GET", "/api/v1/pods", authorizer.AttributesRecord{APIGroup: api.GroupName, Resource: "pods"}},
		"prefix/group/version/resource":  {"GET", "/apis/extensions/v1/pods", authorizer.AttributesRecord{APIGroup: extensions.GroupName, Resource: "pods"}},
		"prefix/group/version/resource2": {"GET", "/apis/autoscaling/v1/horizontalpodautoscalers", authorizer.AttributesRecord{APIGroup: autoscaling.GroupName, Resource: "horizontalpodautoscalers"}},
	}

	currentAuthorizationAttributesIndex := 0

	for testName, r := range requests {
		token := BobToken
		req, err := http.NewRequest(r.verb, s.URL+r.URL, nil)
		if err != nil {
			t.Logf("case %v", testName)
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}

			found := false
			for i := currentAuthorizationAttributesIndex; i < len(trackingAuthorizer.requestAttributes); i++ {
				if trackingAuthorizer.requestAttributes[i].GetAPIGroup() == r.expectedAttributes.GetAPIGroup() &&
					trackingAuthorizer.requestAttributes[i].GetResource() == r.expectedAttributes.GetResource() {
					found = true
					break
				}

				t.Logf("%#v did not match %#v", r.expectedAttributes, trackingAuthorizer.requestAttributes[i].(*authorizer.AttributesRecord))
			}
			if !found {
				t.Errorf("did not find %#v in %#v", r.expectedAttributes, trackingAuthorizer.requestAttributes[currentAuthorizationAttributesIndex:])
			}

			currentAuthorizationAttributesIndex = len(trackingAuthorizer.requestAttributes)
		}()
	}
}

// TestNamespaceAuthorization tests that authorization can be controlled
// by namespace.
func TestNamespaceAuthorization(t *testing.T) {
	framework.DeleteAllEtcdKeys()

	// This file has alice and bob in it.
	a := newAuthorizerWithContents(t, `{"namespace": "foo"}
`)

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = getTestTokenAuth()
	masterConfig.Authorizer = a
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	previousResourceVersion := make(map[string]float64)
	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		namespace   string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{

		{"POST", timeoutPath("pods", "foo", ""), "foo", aPod, code201},
		{"GET", path("pods", "foo", ""), "foo", "", code200},
		{"GET", path("pods", "foo", "a"), "foo", "", code200},
		{"DELETE", timeoutPath("pods", "foo", "a"), "foo", "", code200},

		{"POST", timeoutPath("pods", "bar", ""), "bar", aPod, code403},
		{"GET", path("pods", "bar", ""), "bar", "", code403},
		{"GET", path("pods", "bar", "a"), "bar", "", code403},
		{"DELETE", timeoutPath("pods", "bar", "a"), "bar", "", code403},

		{"POST", timeoutPath("pods", api.NamespaceDefault, ""), "", aPod, code403},
		{"GET", path("pods", "", ""), "", "", code403},
		{"GET", path("pods", api.NamespaceDefault, "a"), "", "", code403},
		{"DELETE", timeoutPath("pods", api.NamespaceDefault, "a"), "", "", code403},
	}

	for _, r := range requests {
		token := BobToken
		var bodyStr string
		if r.body != "" {
			sub := ""
			if r.verb == "PUT" && r.body != "" {
				// For update operations, insert previous resource version
				if resVersion := previousResourceVersion[getPreviousResourceVersionKey(r.URL, "")]; resVersion != 0 {
					sub += fmt.Sprintf(",\r\n\"resourceVersion\": \"%v\"", resVersion)
				}
				namespace := r.namespace
				if len(namespace) == 0 {
					namespace = "default"
				}
				sub += fmt.Sprintf(",\r\n\"namespace\": %q", namespace)
			}
			bodyStr = fmt.Sprintf(r.body, sub)
		}
		r.body = bodyStr
		bodyBytes := bytes.NewReader([]byte(bodyStr))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Logf("case %v", r)
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		func() {
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			b, _ := ioutil.ReadAll(resp.Body)
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Logf("case %v", r)
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				t.Errorf("Body: %v", string(b))
			} else {
				if r.verb == "POST" {
					// For successful create operations, extract resourceVersion
					id, currentResourceVersion, err := parseResourceVersion(b)
					if err == nil {
						key := getPreviousResourceVersionKey(r.URL, id)
						previousResourceVersion[key] = currentResourceVersion
					}
				}
			}

		}()
	}
}

// TestKindAuthorization tests that authorization can be controlled
// by namespace.
func TestKindAuthorization(t *testing.T) {
	framework.DeleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	a := newAuthorizerWithContents(t, `{"resource": "services"}
`)

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = getTestTokenAuth()
	masterConfig.Authorizer = a
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	previousResourceVersion := make(map[string]float64)
	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{
		{"POST", timeoutPath("services", api.NamespaceDefault, ""), aService, code201},
		{"GET", path("services", api.NamespaceDefault, ""), "", code200},
		{"GET", path("services", api.NamespaceDefault, "a"), "", code200},
		{"DELETE", timeoutPath("services", api.NamespaceDefault, "a"), "", code200},

		{"POST", timeoutPath("pods", api.NamespaceDefault, ""), aPod, code403},
		{"GET", path("pods", "", ""), "", code403},
		{"GET", path("pods", api.NamespaceDefault, "a"), "", code403},
		{"DELETE", timeoutPath("pods", api.NamespaceDefault, "a"), "", code403},
	}

	for _, r := range requests {
		token := BobToken
		var bodyStr string
		if r.body != "" {
			bodyStr = fmt.Sprintf(r.body, "")
			if r.verb == "PUT" && r.body != "" {
				// For update operations, insert previous resource version
				if resVersion := previousResourceVersion[getPreviousResourceVersionKey(r.URL, "")]; resVersion != 0 {
					resourceVersionJson := fmt.Sprintf(",\r\n\"resourceVersion\": \"%v\"", resVersion)
					bodyStr = fmt.Sprintf(r.body, resourceVersionJson)
				}
			}
		}
		r.body = bodyStr
		bodyBytes := bytes.NewReader([]byte(bodyStr))
		req, err := http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Logf("case %v", r)
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		{
			resp, err := transport.RoundTrip(req)
			defer resp.Body.Close()
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			b, _ := ioutil.ReadAll(resp.Body)
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Logf("case %v", r)
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				t.Errorf("Body: %v", string(b))
			} else {
				if r.verb == "POST" {
					// For successful create operations, extract resourceVersion
					id, currentResourceVersion, err := parseResourceVersion(b)
					if err == nil {
						key := getPreviousResourceVersionKey(r.URL, id)
						previousResourceVersion[key] = currentResourceVersion
					}
				}
			}

		}
	}
}

// TestReadOnlyAuthorization tests that authorization can be controlled
// by namespace.
func TestReadOnlyAuthorization(t *testing.T) {
	framework.DeleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	a := newAuthorizerWithContents(t, `{"readonly": true}`)

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = getTestTokenAuth()
	masterConfig.Authorizer = a

	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{
		{"POST", path("pods", "", ""), aPod, code403},
		{"GET", path("pods", "", ""), "", code200},
		{"GET", path("pods", api.NamespaceDefault, "a"), "", code404},
	}

	for _, r := range requests {
		token := BobToken
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
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			if _, ok := r.statusCodes[resp.StatusCode]; !ok {
				t.Logf("case %v", r)
				t.Errorf("Expected status one of %v, but got %v", r.statusCodes, resp.StatusCode)
				b, _ := ioutil.ReadAll(resp.Body)
				t.Errorf("Body: %v", string(b))
			}
		}()
	}
}
