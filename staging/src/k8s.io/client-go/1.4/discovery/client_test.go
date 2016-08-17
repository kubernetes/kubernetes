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

package discovery

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/emicklei/go-restful/swagger"

	"k8s.io/client-go/1.4/pkg/api/unversioned"
	"k8s.io/client-go/1.4/pkg/api/v1"
	"k8s.io/client-go/1.4/pkg/version"
	"k8s.io/client-go/1.4/rest"
)

func TestGetServerVersion(t *testing.T) {
	expect := version.Info{
		Major:     "foo",
		Minor:     "bar",
		GitCommit: "baz",
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(expect)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})

	got, err := client.ServerVersion()
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	if e, a := expect, *got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGetServerGroupsWithV1Server(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var obj interface{}
		switch req.URL.Path {
		case "/api":
			obj = &unversioned.APIVersions{
				Versions: []string{
					"v1",
				},
			}
		default:
			w.WriteHeader(http.StatusNotFound)
			return
		}
		output, err := json.Marshal(obj)
		if err != nil {
			t.Fatalf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
	// ServerGroups should not return an error even if server returns error at /api and /apis
	apiGroupList, err := client.ServerGroups()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	groupVersions := unversioned.ExtractGroupVersions(apiGroupList)
	if !reflect.DeepEqual(groupVersions, []string{"v1"}) {
		t.Errorf("expected: %q, got: %q", []string{"v1"}, groupVersions)
	}
}

func TestGetServerGroupsWithBrokenServer(t *testing.T) {
	for _, statusCode := range []int{http.StatusNotFound, http.StatusForbidden} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.WriteHeader(statusCode)
		}))
		defer server.Close()
		client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
		// ServerGroups should not return an error even if server returns Not Found or Forbidden error at all end points
		apiGroupList, err := client.ServerGroups()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		groupVersions := unversioned.ExtractGroupVersions(apiGroupList)
		if len(groupVersions) != 0 {
			t.Errorf("expected empty list, got: %q", groupVersions)
		}
	}
}

func TestGetServerResourcesWithV1Server(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var obj interface{}
		switch req.URL.Path {
		case "/api":
			obj = &unversioned.APIVersions{
				Versions: []string{
					"v1",
				},
			}
		default:
			w.WriteHeader(http.StatusNotFound)
			return
		}
		output, err := json.Marshal(obj)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
	// ServerResources should not return an error even if server returns error at /api/v1.
	resourceMap, err := client.ServerResources()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, found := resourceMap["v1"]; !found {
		t.Errorf("missing v1 in resource map")
	}

}

func TestGetServerResources(t *testing.T) {
	stable := unversioned.APIResourceList{
		GroupVersion: "v1",
		APIResources: []unversioned.APIResource{
			{"pods", true, "Pod"},
			{"services", true, "Service"},
			{"namespaces", false, "Namespace"},
		},
	}
	beta := unversioned.APIResourceList{
		GroupVersion: "extensions/v1",
		APIResources: []unversioned.APIResource{
			{"deployments", true, "Deployment"},
			{"ingresses", true, "Ingress"},
			{"jobs", true, "Job"},
		},
	}
	tests := []struct {
		resourcesList *unversioned.APIResourceList
		path          string
		request       string
		expectErr     bool
	}{
		{
			resourcesList: &stable,
			path:          "/api/v1",
			request:       "v1",
			expectErr:     false,
		},
		{
			resourcesList: &beta,
			path:          "/apis/extensions/v1beta1",
			request:       "extensions/v1beta1",
			expectErr:     false,
		},
		{
			resourcesList: &stable,
			path:          "/api/v1",
			request:       "foobar",
			expectErr:     true,
		},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var list interface{}
		switch req.URL.Path {
		case "/api/v1":
			list = &stable
		case "/apis/extensions/v1beta1":
			list = &beta
		case "/api":
			list = &unversioned.APIVersions{
				Versions: []string{
					"v1",
				},
			}
		case "/apis":
			list = &unversioned.APIGroupList{
				Groups: []unversioned.APIGroup{
					{
						Versions: []unversioned.GroupVersionForDiscovery{
							{GroupVersion: "extensions/v1beta1"},
						},
					},
				},
			}
		default:
			t.Logf("unexpected request: %s", req.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		output, err := json.Marshal(list)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
	for _, test := range tests {
		got, err := client.ServerResourcesForGroupVersion(test.request)
		if test.expectErr {
			if err == nil {
				t.Error("unexpected non-error")
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !reflect.DeepEqual(got, test.resourcesList) {
			t.Errorf("expected:\n%v\ngot:\n%v\n", test.resourcesList, got)
		}
	}

	resourceMap, err := client.ServerResources()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	for _, api := range []string{"v1", "extensions/v1beta1"} {
		if _, found := resourceMap[api]; !found {
			t.Errorf("missing expected api: %s", api)
		}
	}
}

func swaggerSchemaFakeServer() (*httptest.Server, error) {
	request := 1
	var sErr error

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var resp interface{}
		if request == 1 {
			resp = unversioned.APIVersions{Versions: []string{"v1", "v2", "v3"}}
			request++
		} else {
			resp = swagger.ApiDeclaration{}
		}
		output, err := json.Marshal(resp)
		if err != nil {
			sErr = err
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	return server, sErr
}

func TestGetSwaggerSchema(t *testing.T) {
	expect := swagger.ApiDeclaration{}

	server, err := swaggerSchemaFakeServer()
	if err != nil {
		t.Errorf("unexpected encoding error: %v", err)
	}
	defer server.Close()

	client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
	got, err := client.SwaggerSchema(v1.SchemeGroupVersion)
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	if e, a := expect, *got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGetSwaggerSchemaFail(t *testing.T) {
	expErr := "API version: api.group/v4 is not supported by the server. Use one of: [v1 v2 v3]"

	server, err := swaggerSchemaFakeServer()
	if err != nil {
		t.Errorf("unexpected encoding error: %v", err)
	}
	defer server.Close()

	client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
	got, err := client.SwaggerSchema(unversioned.GroupVersion{Group: "api.group", Version: "v4"})
	if got != nil {
		t.Fatalf("unexpected response: %v", got)
	}
	if err.Error() != expErr {
		t.Errorf("expected an error, got %v", err)
	}
}
