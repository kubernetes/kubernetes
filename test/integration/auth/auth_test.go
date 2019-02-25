/*
Copyright 2014 The Kubernetes Authors.

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

package auth

// This file tests authentication and (soon) authorization of HTTP requests to a master object.
// It does not use the client in pkg/client/... because authentication and authorization needs
// to work for any client of the HTTP interface.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/group"
	"k8s.io/apiserver/pkg/authentication/request/bearertoken"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/token/cache"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/apiserver/plugin/pkg/authenticator/token/tokentest"
	"k8s.io/apiserver/plugin/pkg/authenticator/token/webhook"
	"k8s.io/client-go/tools/clientcmd/api/v1"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/test/integration"
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
	return group.NewGroupAdder(bearertoken.New(tokenAuthenticator), []string{user.AllAuthenticated})
}

func getTestWebhookTokenAuth(serverURL string) (authenticator.Request, error) {
	kubecfgFile, err := ioutil.TempFile("", "webhook-kubecfg")
	if err != nil {
		return nil, err
	}
	defer os.Remove(kubecfgFile.Name())
	config := v1.Config{
		Clusters: []v1.NamedCluster{
			{
				Cluster: v1.Cluster{Server: serverURL},
			},
		},
	}
	if err := json.NewEncoder(kubecfgFile).Encode(config); err != nil {
		return nil, err
	}
	webhookTokenAuth, err := webhook.New(kubecfgFile.Name(), nil)
	if err != nil {
		return nil, err
	}
	return bearertoken.New(cache.New(webhookTokenAuth, false, 2*time.Minute, 2*time.Minute)), nil
}

func path(resource, namespace, name string) string {
	return testapi.Default.ResourcePath(resource, namespace, name)
}

func pathWithPrefix(prefix, resource, namespace, name string) string {
	return testapi.Default.ResourcePathWithPrefix(prefix, resource, namespace, name)
}

func pathWithSubResource(resource, namespace, name, subresource string) string {
	return testapi.Default.SubResourcePath(resource, namespace, name, subresource)
}

func timeoutPath(resource, namespace, name string) string {
	return addTimeoutFlag(testapi.Default.ResourcePath(resource, namespace, name))
}

// Bodies for requests used in subsequent tests.
var aPod string = `
{
  "kind": "Pod",
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
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
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
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
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
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
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
  "metadata": {
    "name": "a"%s
  },
  "spec": {
    "externalID": "external"
  }
}
`

func aEvent(namespace string) string {
	return `
{
  "kind": "Event",
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
  "metadata": {
    "name": "a"%s
  },
  "involvedObject": {
    "kind": "Pod",
    "namespace": "` + namespace + `",
    "name": "a",
    "apiVersion": "v1"
  }
}
`
}

var aBinding string = `
{
  "kind": "Binding",
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
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
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
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
  "apiVersion": "` + testapi.Groups[api.GroupName].GroupVersion().String() + `",
  "gracePeriodSeconds": 0%s
}
`

// To ensure that a POST completes before a dependent GET, set a timeout.
func addTimeoutFlag(URLString string) string {
	u, _ := url.Parse(URLString)
	values := u.Query()
	values.Set("timeout", "60s")
	u.RawQuery = values.Encode()
	return u.String()
}

func getTestRequests(namespace string) []struct {
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
		{"GET", path("pods", "", ""), "", integration.Code200},
		{"GET", path("pods", namespace, ""), "", integration.Code200},
		{"POST", timeoutPath("pods", namespace, ""), aPod, integration.Code201},
		{"PUT", timeoutPath("pods", namespace, "a"), aPod, integration.Code200},
		{"GET", path("pods", namespace, "a"), "", integration.Code200},
		// GET and POST for /exec should return Bad Request (400) since the pod has not been assigned a node yet.
		{"GET", path("pods", namespace, "a") + "/exec", "", integration.Code400},
		{"POST", path("pods", namespace, "a") + "/exec", "", integration.Code400},
		// PUT for /exec should return Method Not Allowed (405).
		{"PUT", path("pods", namespace, "a") + "/exec", "", integration.Code405},
		// GET and POST for /portforward should return Bad Request (400) since the pod has not been assigned a node yet.
		{"GET", path("pods", namespace, "a") + "/portforward", "", integration.Code400},
		{"POST", path("pods", namespace, "a") + "/portforward", "", integration.Code400},
		// PUT for /portforward should return Method Not Allowed (405).
		{"PUT", path("pods", namespace, "a") + "/portforward", "", integration.Code405},
		{"PATCH", path("pods", namespace, "a"), "{%v}", integration.Code200},
		{"DELETE", timeoutPath("pods", namespace, "a"), deleteNow, integration.Code200},

		// Non-standard methods (not expected to work,
		// but expected to pass/fail authorization prior to
		// failing validation.
		{"OPTIONS", path("pods", namespace, ""), "", integration.Code405},
		{"OPTIONS", path("pods", namespace, "a"), "", integration.Code405},
		{"HEAD", path("pods", namespace, ""), "", integration.Code405},
		{"HEAD", path("pods", namespace, "a"), "", integration.Code405},
		{"TRACE", path("pods", namespace, ""), "", integration.Code405},
		{"TRACE", path("pods", namespace, "a"), "", integration.Code405},
		{"NOSUCHVERB", path("pods", namespace, ""), "", integration.Code405},

		// Normal methods on services
		{"GET", path("services", "", ""), "", integration.Code200},
		{"GET", path("services", namespace, ""), "", integration.Code200},
		{"POST", timeoutPath("services", namespace, ""), aService, integration.Code201},
		// Create an endpoint for the service (this is done automatically by endpoint controller
		// whenever a service is created, but this test does not run that controller)
		{"POST", timeoutPath("endpoints", namespace, ""), emptyEndpoints, integration.Code201},
		// Should return service unavailable when endpoint.subset is empty.
		{"GET", pathWithSubResource("services", namespace, "a", "proxy") + "/", "", integration.Code503},
		{"PUT", timeoutPath("services", namespace, "a"), aService, integration.Code200},
		{"GET", path("services", namespace, "a"), "", integration.Code200},
		{"DELETE", timeoutPath("endpoints", namespace, "a"), "", integration.Code200},
		{"DELETE", timeoutPath("services", namespace, "a"), "", integration.Code200},

		// Normal methods on replicationControllers
		{"GET", path("replicationControllers", "", ""), "", integration.Code200},
		{"GET", path("replicationControllers", namespace, ""), "", integration.Code200},
		{"POST", timeoutPath("replicationControllers", namespace, ""), aRC, integration.Code201},
		{"PUT", timeoutPath("replicationControllers", namespace, "a"), aRC, integration.Code200},
		{"GET", path("replicationControllers", namespace, "a"), "", integration.Code200},
		{"DELETE", timeoutPath("replicationControllers", namespace, "a"), "", integration.Code200},

		// Normal methods on endpoints
		{"GET", path("endpoints", "", ""), "", integration.Code200},
		{"GET", path("endpoints", namespace, ""), "", integration.Code200},
		{"POST", timeoutPath("endpoints", namespace, ""), aEndpoints, integration.Code201},
		{"PUT", timeoutPath("endpoints", namespace, "a"), aEndpoints, integration.Code200},
		{"GET", path("endpoints", namespace, "a"), "", integration.Code200},
		{"DELETE", timeoutPath("endpoints", namespace, "a"), "", integration.Code200},

		// Normal methods on nodes
		{"GET", path("nodes", "", ""), "", integration.Code200},
		{"POST", timeoutPath("nodes", "", ""), aNode, integration.Code201},
		{"PUT", timeoutPath("nodes", "", "a"), aNode, integration.Code200},
		{"GET", path("nodes", "", "a"), "", integration.Code200},
		{"DELETE", timeoutPath("nodes", "", "a"), "", integration.Code200},

		// Normal methods on events
		{"GET", path("events", "", ""), "", integration.Code200},
		{"GET", path("events", namespace, ""), "", integration.Code200},
		{"POST", timeoutPath("events", namespace, ""), aEvent(namespace), integration.Code201},
		{"PUT", timeoutPath("events", namespace, "a"), aEvent(namespace), integration.Code200},
		{"GET", path("events", namespace, "a"), "", integration.Code200},
		{"DELETE", timeoutPath("events", namespace, "a"), "", integration.Code200},

		// Normal methods on bindings
		{"GET", path("bindings", namespace, ""), "", integration.Code405},
		{"POST", timeoutPath("pods", namespace, ""), aPod, integration.Code201}, // Need a pod to bind or you get a 404
		{"POST", timeoutPath("bindings", namespace, ""), aBinding, integration.Code201},
		{"PUT", timeoutPath("bindings", namespace, "a"), aBinding, integration.Code404},
		{"GET", path("bindings", namespace, "a"), "", integration.Code404}, // No bindings instances
		{"DELETE", timeoutPath("bindings", namespace, "a"), "", integration.Code404},

		// Non-existent object type.
		{"GET", path("foo", "", ""), "", integration.Code404},
		{"POST", path("foo", namespace, ""), `{"foo": "foo"}`, integration.Code404},
		{"PUT", path("foo", namespace, "a"), `{"foo": "foo"}`, integration.Code404},
		{"GET", path("foo", namespace, "a"), "", integration.Code404},
		{"DELETE", timeoutPath("foo", namespace, ""), "", integration.Code404},

		// Special verbs on nodes
		{"GET", pathWithSubResource("nodes", namespace, "a", "proxy"), "", integration.Code404},
		{"GET", pathWithPrefix("redirect", "nodes", namespace, "a"), "", integration.Code404},
		// TODO: test .../watch/..., which doesn't end before the test timeout.
		// TODO: figure out how to create a node so that it can successfully proxy/redirect.

		// Non-object endpoints
		{"GET", "/", "", integration.Code200},
		{"GET", "/api", "", integration.Code200},
		{"GET", "/healthz", "", integration.Code200},
		{"GET", "/version", "", integration.Code200},
		{"GET", "/invalidURL", "", integration.Code404},
	}
	return requests
}

// The TestAuthMode* tests a large number of URLs and checks that they
// are FORBIDDEN or not, depending on the mode.  They do not attempt to do
// detailed verification of behaviour beyond authorization.  They are not
// fuzz tests.
//
// TODO(etune): write a fuzz test of the REST API.
func TestAuthModeAlwaysAllow(t *testing.T) {
	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-always-allow", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	transport := http.DefaultTransport
	previousResourceVersion := make(map[string]float64)

	for _, r := range getTestRequests(ns.Name) {
		var bodyStr string
		if r.body != "" {
			sub := ""
			if r.verb == "PUT" {
				// For update operations, insert previous resource version
				if resVersion := previousResourceVersion[getPreviousResourceVersionKey(r.URL, "")]; resVersion != 0 {
					sub += fmt.Sprintf(",\r\n\"resourceVersion\": \"%v\"", resVersion)
				}
				sub += fmt.Sprintf(",\r\n\"namespace\": %q", ns.Name)
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
	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authorization.Authorizer = authorizerfactory.NewAlwaysDenyAuthorizer()
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-always-deny", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	transport := http.DefaultTransport

	for _, r := range getTestRequests(ns.Name) {
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

func (allowAliceAuthorizer) Authorize(a authorizer.Attributes) (authorizer.Decision, string, error) {
	if a.GetUser() != nil && a.GetUser().GetName() == "alice" {
		return authorizer.DecisionAllow, "", nil
	}
	return authorizer.DecisionNoOpinion, "I can't allow that.  Go ask alice.", nil
}

// TestAliceNotForbiddenOrUnauthorized tests a user who is known to
// the authentication system and authorized to do any actions.
func TestAliceNotForbiddenOrUnauthorized(t *testing.T) {
	// This file has alice and bob in it.

	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = getTestTokenAuth()
	masterConfig.GenericConfig.Authorization.Authorizer = allowAliceAuthorizer{}
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-alice-not-forbidden", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	previousResourceVersion := make(map[string]float64)
	transport := http.DefaultTransport

	for _, r := range getTestRequests(ns.Name) {
		token := AliceToken
		var bodyStr string
		if r.body != "" {
			sub := ""
			if r.verb == "PUT" {
				// For update operations, insert previous resource version
				if resVersion := previousResourceVersion[getPreviousResourceVersionKey(r.URL, "")]; resVersion != 0 {
					sub += fmt.Sprintf(",\r\n\"resourceVersion\": \"%v\"", resVersion)
				}
				sub += fmt.Sprintf(",\r\n\"namespace\": %q", ns.Name)
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
	// This file has alice and bob in it.
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = getTestTokenAuth()
	masterConfig.GenericConfig.Authorization.Authorizer = allowAliceAuthorizer{}
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-bob-forbidden", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	transport := http.DefaultTransport

	for _, r := range getTestRequests(ns.Name) {
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
	// This file has alice and bob in it.

	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = getTestTokenAuth()
	masterConfig.GenericConfig.Authorization.Authorizer = allowAliceAuthorizer{}
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-unknown-unauthorized", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	transport := http.DefaultTransport

	for _, r := range getTestRequests(ns.Name) {
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
func (impersonateAuthorizer) Authorize(a authorizer.Attributes) (authorizer.Decision, string, error) {
	// alice can impersonate service accounts and do other actions
	if a.GetUser() != nil && a.GetUser().GetName() == "alice" && a.GetVerb() == "impersonate" && a.GetResource() == "serviceaccounts" {
		return authorizer.DecisionAllow, "", nil
	}
	if a.GetUser() != nil && a.GetUser().GetName() == "alice" && a.GetVerb() != "impersonate" {
		return authorizer.DecisionAllow, "", nil
	}
	// bob can impersonate anyone, but that it
	if a.GetUser() != nil && a.GetUser().GetName() == "bob" && a.GetVerb() == "impersonate" {
		return authorizer.DecisionAllow, "", nil
	}
	// service accounts can do everything
	if a.GetUser() != nil && strings.HasPrefix(a.GetUser().GetName(), serviceaccount.ServiceAccountUsernamePrefix) {
		return authorizer.DecisionAllow, "", nil
	}

	return authorizer.DecisionNoOpinion, "I can't allow that.  Go ask alice.", nil
}

func TestImpersonateIsForbidden(t *testing.T) {
	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = getTestTokenAuth()
	masterConfig.GenericConfig.Authorization.Authorizer = impersonateAuthorizer{}
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-impersonate-forbidden", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	transport := http.DefaultTransport

	// bob can't perform actions himself
	for _, r := range getTestRequests(ns.Name) {
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
	for _, r := range getTestRequests(ns.Name) {
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
	for _, r := range getTestRequests(ns.Name) {
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
	for _, r := range getTestRequests(ns.Name) {
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

func (a *trackingAuthorizer) Authorize(attributes authorizer.Attributes) (authorizer.Decision, string, error) {
	a.requestAttributes = append(a.requestAttributes, attributes)
	return authorizer.DecisionAllow, "", nil
}

// TestAuthorizationAttributeDetermination tests that authorization attributes are built correctly
func TestAuthorizationAttributeDetermination(t *testing.T) {
	trackingAuthorizer := &trackingAuthorizer{}

	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = getTestTokenAuth()
	masterConfig.GenericConfig.Authorization.Authorizer = trackingAuthorizer
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-attribute-determination", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

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
	// This file has alice and bob in it.
	a := newAuthorizerWithContents(t, `{"namespace": "auth-namespace"}
`)

	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = getTestTokenAuth()
	masterConfig.GenericConfig.Authorization.Authorizer = a
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-namespace", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	previousResourceVersion := make(map[string]float64)
	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		namespace   string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{

		{"POST", timeoutPath("pods", ns.Name, ""), "foo", aPod, integration.Code201},
		{"GET", path("pods", ns.Name, ""), "foo", "", integration.Code200},
		{"GET", path("pods", ns.Name, "a"), "foo", "", integration.Code200},
		{"DELETE", timeoutPath("pods", ns.Name, "a"), "foo", "", integration.Code200},

		{"POST", timeoutPath("pods", "foo", ""), "bar", aPod, integration.Code403},
		{"GET", path("pods", "foo", ""), "bar", "", integration.Code403},
		{"GET", path("pods", "foo", "a"), "bar", "", integration.Code403},
		{"DELETE", timeoutPath("pods", "foo", "a"), "bar", "", integration.Code403},

		{"POST", timeoutPath("pods", metav1.NamespaceDefault, ""), "", aPod, integration.Code403},
		{"GET", path("pods", "", ""), "", "", integration.Code403},
		{"GET", path("pods", metav1.NamespaceDefault, "a"), "", "", integration.Code403},
		{"DELETE", timeoutPath("pods", metav1.NamespaceDefault, "a"), "", "", integration.Code403},
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
				// FIXME: Is that correct?
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
	// This file has alice and bob in it.
	a := newAuthorizerWithContents(t, `{"resource": "services"}
`)

	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = getTestTokenAuth()
	masterConfig.GenericConfig.Authorization.Authorizer = a
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-kind", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	previousResourceVersion := make(map[string]float64)
	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{
		{"POST", timeoutPath("services", ns.Name, ""), aService, integration.Code201},
		{"GET", path("services", ns.Name, ""), "", integration.Code200},
		{"GET", path("services", ns.Name, "a"), "", integration.Code200},
		{"DELETE", timeoutPath("services", ns.Name, "a"), "", integration.Code200},

		{"POST", timeoutPath("pods", ns.Name, ""), aPod, integration.Code403},
		{"GET", path("pods", "", ""), "", integration.Code403},
		{"GET", path("pods", ns.Name, "a"), "", integration.Code403},
		{"DELETE", timeoutPath("pods", ns.Name, "a"), "", integration.Code403},
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
	// This file has alice and bob in it.
	a := newAuthorizerWithContents(t, `{"readonly": true}`)

	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = getTestTokenAuth()
	masterConfig.GenericConfig.Authorization.Authorizer = a
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-read-only", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	transport := http.DefaultTransport

	requests := []struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // allowed status codes.
	}{
		{"POST", path("pods", ns.Name, ""), aPod, integration.Code403},
		{"GET", path("pods", ns.Name, ""), "", integration.Code200},
		{"GET", path("pods", metav1.NamespaceDefault, "a"), "", integration.Code404},
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

// TestWebhookTokenAuthenticator tests that a master can use the webhook token
// authenticator to call out to a remote web server for authentication
// decisions.
func TestWebhookTokenAuthenticator(t *testing.T) {
	authServer := newTestWebhookTokenAuthServer()
	defer authServer.Close()
	authenticator, err := getTestWebhookTokenAuth(authServer.URL)
	if err != nil {
		t.Fatalf("error starting webhook token authenticator server: %v", err)
	}

	// Set up a master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = authenticator
	masterConfig.GenericConfig.Authorization.Authorizer = allowAliceAuthorizer{}
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	ns := framework.CreateTestingNamespace("auth-webhook-token", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	transport := http.DefaultTransport

	for _, r := range getTestRequests(ns.Name) {
		// Expect Bob's requests to all fail.
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
			// Expect all of Bob's actions to return Forbidden
			if resp.StatusCode != http.StatusForbidden {
				t.Logf("case %v", r)
				t.Errorf("Expected http.Forbidden, but got %s", resp.Status)
			}
		}()
		// Expect Alice's requests to succeed.
		token = AliceToken
		bodyBytes = bytes.NewReader([]byte(r.body))
		req, err = http.NewRequest(r.verb, s.URL+r.URL, bodyBytes)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

		func() {
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Logf("case %v", r)
				t.Fatalf("unexpected error: %v", err)
			}
			defer resp.Body.Close()
			// Expect all of Alice's actions to at least get past authn/authz.
			if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
				t.Logf("case %v", r)
				t.Errorf("Expected something other than Unauthorized/Forbidden, but got %s", resp.Status)
			}
		}()
	}
}

// newTestWebhookTokenAuthServer creates an http token authentication server
// that knows about both Alice and Bob.
func newTestWebhookTokenAuthServer() *httptest.Server {
	serveHTTP := func(w http.ResponseWriter, r *http.Request) {
		var review authenticationv1beta1.TokenReview
		if err := json.NewDecoder(r.Body).Decode(&review); err != nil {
			http.Error(w, fmt.Sprintf("failed to decode body: %v", err), http.StatusBadRequest)
			return
		}
		type userInfo struct {
			Username string   `json:"username"`
			UID      string   `json:"uid"`
			Groups   []string `json:"groups"`
		}
		type status struct {
			Authenticated bool     `json:"authenticated"`
			User          userInfo `json:"user"`
		}
		var username, uid string
		authenticated := false
		if review.Spec.Token == AliceToken {
			authenticated, username, uid = true, "alice", "1"
		} else if review.Spec.Token == BobToken {
			authenticated, username, uid = true, "bob", "2"
		}

		resp := struct {
			APIVersion string `json:"apiVersion"`
			Status     status `json:"status"`
		}{
			APIVersion: authenticationv1beta1.SchemeGroupVersion.String(),
			Status: status{
				authenticated,
				userInfo{
					Username: username,
					UID:      uid,
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}

	server := httptest.NewUnstartedServer(http.HandlerFunc(serveHTTP))
	server.Start()
	return server
}
