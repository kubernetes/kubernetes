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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator/bearertoken"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer/abac"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/admit"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/token/tokentest"
)

var nodeResourceName string

func init() {
	requireEtcd()
	if api.PreV1Beta3(testapi.Version()) {
		nodeResourceName = "minions"
	} else {
		nodeResourceName = "nodes"
	}
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

func path(resource, namespace, name string) string {
	return testapi.ResourcePath(resource, namespace, name)
}

func pathWithNamespaceQuery(resource, namespace, name string) string {
	return testapi.ResourcePathWithNamespaceQuery(resource, namespace, name)
}

func pathWithPrefix(prefix, resource, namespace, name string) string {
	return testapi.ResourcePathWithPrefix(prefix, resource, namespace, name)
}

func timeoutPath(resource, namespace, name string) string {
	return addTimeoutFlag(testapi.ResourcePath(resource, namespace, name))
}

func timeoutPathWithNamespaceQuery(resource, namespace, name string) string {
	return addTimeoutFlag(testapi.ResourcePathWithNamespaceQuery(resource, namespace, name))
}

// Bodies for requests used in subsequent tests.
var aPod string = `
{
  "kind": "Pod",
  "apiVersion": "v1beta3",
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
  "apiVersion": "v1beta3",
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
  "apiVersion": "v1beta3",
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
    "portalIP": "10.0.0.100"
  }
}
`
var aNode string = `
{
  "kind": "Node",
  "apiVersion": "v1beta3",
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
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "a"%s
  },
  "involvedObject": {
    "kind": "Node",
    "namespace": "default",
    "name": "a",
    "apiVersion": "v1beta3"
  }
}
`

var aBinding string = `
{
  "kind": "Binding",
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "a"%s
  },
  "target": {
    "name": "10.10.10.10"
  }
}
`

var aEndpoints string = `
{
  "kind": "Endpoints",
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "a"%s
  },
  "subsets": [
    {
      "addresses": [
        {
          "IP": "10.10.1.1"
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
  "apiVersion": "v1beta3",
  "gracePeriodSeconds": null%s
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
		{"PUT", timeoutPath("services", api.NamespaceDefault, "a"), aService, code200},
		{"GET", path("services", api.NamespaceDefault, "a"), "", code200},
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

		// Normal methods on minions
		{"GET", path(nodeResourceName, "", ""), "", code200},
		{"POST", timeoutPath(nodeResourceName, "", ""), aNode, code201},
		{"PUT", timeoutPath(nodeResourceName, "", "a"), aNode, code200},
		{"GET", path(nodeResourceName, "", "a"), "", code200},
		{"DELETE", timeoutPath(nodeResourceName, "", "a"), "", code200},

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
		{"GET", pathWithPrefix("proxy", nodeResourceName, api.NamespaceDefault, "a"), "", code404},
		{"GET", pathWithPrefix("redirect", nodeResourceName, api.NamespaceDefault, "a"), "", code404},
		// TODO: test .../watch/..., which doesn't end before the test timeout.
		// TODO: figure out how to create a minion so that it can successfully proxy/redirect.

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
	deleteAllEtcdKeys()

	// Set up a master

	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authorizer:            apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

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
	apiVersion, ok := resultBodyMap["apiVersion"].(string)
	if !ok {
		return "", 0, fmt.Errorf("unexpected error, apiVersion not found in JSON response: %v", string(response))
	}
	if api.PreV1Beta3(apiVersion) {
		return parsePreV1Beta3ResourceVersion(resultBodyMap, response)
	}
	return parseV1Beta3ResourceVersion(resultBodyMap, response)
}

func parseV1Beta3ResourceVersion(resultBodyMap map[string]interface{}, response []byte) (string, float64, error) {
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

func parsePreV1Beta3ResourceVersion(resultBodyMap map[string]interface{}, response []byte) (string, float64, error) {
	id, ok := resultBodyMap["id"].(string)
	if !ok {
		return "", 0, fmt.Errorf("unexpected error, id not found in JSON response: %v", string(response))
	}

	resourceVersion, ok := resultBodyMap["resourceVersion"].(float64)
	if !ok {
		return "", 0, fmt.Errorf("unexpected error, resourceVersion not found in JSON response: %v", string(response))
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
	deleteAllEtcdKeys()

	// Set up a master
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authorizer:            apiserver.NewAlwaysDenyAuthorizer(),
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

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

	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authenticator:         getTestTokenAuth(),
		Authorizer:            allowAliceAuthorizer{},
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

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
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authenticator:         getTestTokenAuth(),
		Authorizer:            allowAliceAuthorizer{},
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

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
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authenticator:         getTestTokenAuth(),
		Authorizer:            allowAliceAuthorizer{},
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

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
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
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
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authenticator:         getTestTokenAuth(),
		Authorizer:            a,
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

	previousResourceVersion := make(map[string]float64)
	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		namespace   string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{

		{"POST", timeoutPathWithNamespaceQuery("pods", "foo", ""), "foo", aPod, code201},
		{"GET", pathWithNamespaceQuery("pods", "foo", ""), "foo", "", code200},
		{"GET", pathWithNamespaceQuery("pods", "foo", "a"), "foo", "", code200},
		{"DELETE", timeoutPathWithNamespaceQuery("pods", "foo", "a"), "foo", "", code200},

		{"POST", timeoutPath("pods", "bar", ""), "bar", aPod, code403},
		{"GET", pathWithNamespaceQuery("pods", "bar", ""), "bar", "", code403},
		{"GET", pathWithNamespaceQuery("pods", "bar", "a"), "bar", "", code403},
		{"DELETE", timeoutPathWithNamespaceQuery("pods", "bar", "a"), "bar", "", code403},

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
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	a := newAuthorizerWithContents(t, `{"resource": "services"}
`)

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authenticator:         getTestTokenAuth(),
		Authorizer:            a,
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

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
	deleteAllEtcdKeys()

	// This file has alice and bob in it.

	// Set up a master
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	a := newAuthorizerWithContents(t, `{"readonly": true}`)

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authenticator:         getTestTokenAuth(),
		Authorizer:            a,
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

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
