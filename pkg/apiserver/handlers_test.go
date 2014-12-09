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

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type fakeRL bool

func (fakeRL) Stop()             {}
func (f fakeRL) CanAccept() bool { return bool(f) }

func TestRateLimit(t *testing.T) {
	for _, allow := range []bool{true, false} {
		rl := fakeRL(allow)
		server := httptest.NewServer(RateLimit(rl, http.HandlerFunc(
			func(w http.ResponseWriter, req *http.Request) {
				if !allow {
					t.Errorf("Unexpected call")
				}
			},
		)))
		defer server.Close()
		http.Get(server.URL)
	}
}

func TestReadOnly(t *testing.T) {
	server := httptest.NewServer(ReadOnly(http.HandlerFunc(
		func(w http.ResponseWriter, req *http.Request) {
			if req.Method != "GET" {
				t.Errorf("Unexpected call: %v", req.Method)
			}
		},
	)))
	defer server.Close()
	for _, verb := range []string{"GET", "POST", "PUT", "DELETE", "CREATE"} {
		req, err := http.NewRequest(verb, server.URL, nil)
		if err != nil {
			t.Fatalf("Couldn't make request: %v", err)
		}
		http.DefaultClient.Do(req)
	}
}

func TestKindAndNamespace(t *testing.T) {
	successCases := []struct {
		url               string
		expectedNamespace string
		expectedKind      string
		expectedParts     []string
	}{
		// resource paths
		{"/ns/other/pods", "other", "pods", []string{"pods"}},
		{"/ns/other/pods/foo", "other", "pods", []string{"pods", "foo"}},
		{"/pods", api.NamespaceAll, "pods", []string{"pods"}},
		{"/pods/foo", api.NamespaceDefault, "pods", []string{"pods", "foo"}},
		{"/pods/foo?namespace=other", "other", "pods", []string{"pods", "foo"}},
		{"/pods?namespace=other", "other", "pods", []string{"pods"}},

		// special verbs
		{"/proxy/ns/other/pods/foo", "other", "pods", []string{"pods", "foo"}},
		{"/proxy/pods/foo", api.NamespaceDefault, "pods", []string{"pods", "foo"}},
		{"/redirect/ns/other/pods/foo", "other", "pods", []string{"pods", "foo"}},
		{"/redirect/pods/foo", api.NamespaceDefault, "pods", []string{"pods", "foo"}},
		{"/watch/pods", api.NamespaceAll, "pods", []string{"pods"}},
		{"/watch/ns/other/pods", "other", "pods", []string{"pods"}},

		// full-qualified paths
		{"/api/v1beta1/ns/other/pods", "other", "pods", []string{"pods"}},
		{"/api/v1beta1/ns/other/pods/foo", "other", "pods", []string{"pods", "foo"}},
		{"/api/v1beta1/pods", api.NamespaceAll, "pods", []string{"pods"}},
		{"/api/v1beta1/pods/foo", api.NamespaceDefault, "pods", []string{"pods", "foo"}},
		{"/api/v1beta1/pods/foo?namespace=other", "other", "pods", []string{"pods", "foo"}},
		{"/api/v1beta1/pods?namespace=other", "other", "pods", []string{"pods"}},
		{"/api/v1beta1/proxy/pods/foo", api.NamespaceDefault, "pods", []string{"pods", "foo"}},
		{"/api/v1beta1/redirect/pods/foo", api.NamespaceDefault, "pods", []string{"pods", "foo"}},
		{"/api/v1beta1/watch/pods", api.NamespaceAll, "pods", []string{"pods"}},
		{"/api/v1beta1/watch/ns/other/pods", "other", "pods", []string{"pods"}},
	}

	for _, successCase := range successCases {
		req, _ := http.NewRequest("GET", successCase.url, nil)
		namespace, kind, parts, err := KindAndNamespace(req)
		if err != nil {
			t.Errorf("Unexpected error for url: %s", successCase.url)
		}
		if successCase.expectedNamespace != namespace {
			t.Errorf("Unexpected namespace for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedNamespace, namespace)
		}
		if successCase.expectedKind != kind {
			t.Errorf("Unexpected resourceType for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedKind, kind)
		}
		if !reflect.DeepEqual(successCase.expectedParts, parts) {
			t.Errorf("Unexpected parts for url: %s, expected: %v, actual: %v", successCase.url, successCase.expectedParts, parts)
		}
	}

	errorCases := map[string]string{
		"no resource path":               "/",
		"missing resource type":          "/ns/other",
		"just apiversion":                "/api/v1beta1/",
		"apiversion with no resource":    "/api/v1beta1/",
		"apiversion with just namespace": "/api/v1beta1/ns/other",
	}
	for k, v := range errorCases {
		req, err := http.NewRequest("GET", v, nil)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
		_, _, _, err = KindAndNamespace(req)
		if err == nil {
			t.Errorf("Expected error for key: %s", k)
		}
	}
}
