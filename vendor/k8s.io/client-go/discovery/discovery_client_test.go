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

	"k8s.io/client-go/pkg/api/unversioned"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/pkg/version"
	"k8s.io/client-go/rest"
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
			{Name: "pods", Namespaced: true, Kind: "Pod"},
			{Name: "services", Namespaced: true, Kind: "Service"},
			{Name: "namespaces", Namespaced: false, Kind: "Namespace"},
		},
	}
	beta := unversioned.APIResourceList{
		GroupVersion: "extensions/v1",
		APIResources: []unversioned.APIResource{
			{Name: "deployments", Namespaced: true, Kind: "Deployment"},
			{Name: "ingresses", Namespaced: true, Kind: "Ingress"},
			{Name: "jobs", Namespaced: true, Kind: "Job"},
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

func TestServerPreferredResources(t *testing.T) {
	stable := unversioned.APIResourceList{
		GroupVersion: "v1",
		APIResources: []unversioned.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
			{Name: "services", Namespaced: true, Kind: "Service"},
			{Name: "namespaces", Namespaced: false, Kind: "Namespace"},
		},
	}
	tests := []struct {
		resourcesList *unversioned.APIResourceList
		response      func(w http.ResponseWriter, req *http.Request)
		expectErr     func(err error) bool
	}{
		{
			resourcesList: &stable,
			expectErr:     IsGroupDiscoveryFailedError,
			response: func(w http.ResponseWriter, req *http.Request) {
				var list interface{}
				switch req.URL.Path {
				case "/apis/extensions/v1beta1":
					w.WriteHeader(http.StatusInternalServerError)
					return
				case "/api/v1":
					list = &stable
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
			},
		},
		{
			resourcesList: nil,
			expectErr:     IsGroupDiscoveryFailedError,
			response: func(w http.ResponseWriter, req *http.Request) {
				var list interface{}
				switch req.URL.Path {
				case "/apis/extensions/v1beta1":
					w.WriteHeader(http.StatusInternalServerError)
					return
				case "/api/v1":
					w.WriteHeader(http.StatusInternalServerError)
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
			},
		},
	}
	for _, test := range tests {
		server := httptest.NewServer(http.HandlerFunc(test.response))
		defer server.Close()

		client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
		got, err := client.ServerPreferredResources()
		if test.expectErr != nil {
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
		server.Close()
	}
}

func TestServerPreferredResourcesRetries(t *testing.T) {
	stable := unversioned.APIResourceList{
		GroupVersion: "v1",
		APIResources: []unversioned.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
		},
	}
	beta := unversioned.APIResourceList{
		GroupVersion: "extensions/v1",
		APIResources: []unversioned.APIResource{
			{Name: "deployments", Namespaced: true, Kind: "Deployment"},
		},
	}

	response := func(numErrors int) http.HandlerFunc {
		var i = 0
		return func(w http.ResponseWriter, req *http.Request) {
			var list interface{}
			switch req.URL.Path {
			case "/apis/extensions/v1beta1":
				if i < numErrors {
					i++
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				list = &beta
			case "/api/v1":
				list = &stable
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
							Name: "extensions",
							Versions: []unversioned.GroupVersionForDiscovery{
								{GroupVersion: "extensions/v1beta1"},
							},
							PreferredVersion: unversioned.GroupVersionForDiscovery{
								GroupVersion: "extensions/v1beta1",
								Version:      "v1beta1",
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
		}
	}
	tests := []struct {
		responseErrors  int
		expectResources int
		expectedError   func(err error) bool
	}{
		{
			responseErrors:  1,
			expectResources: 2,
			expectedError: func(err error) bool {
				return err == nil
			},
		},
		{
			responseErrors:  2,
			expectResources: 1,
			expectedError:   IsGroupDiscoveryFailedError,
		},
	}

	for i, tc := range tests {
		server := httptest.NewServer(http.HandlerFunc(response(tc.responseErrors)))
		defer server.Close()

		client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
		got, err := client.ServerPreferredResources()
		if !tc.expectedError(err) {
			t.Errorf("case %d: unexpected error: %v", i, err)
		}
		if len(got) != tc.expectResources {
			t.Errorf("case %d: expect %d resources, got %#v", i, tc.expectResources, got)
		}
		server.Close()
	}
}

func TestServerPreferredNamespacedResources(t *testing.T) {
	stable := unversioned.APIResourceList{
		GroupVersion: "v1",
		APIResources: []unversioned.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
			{Name: "services", Namespaced: true, Kind: "Service"},
			{Name: "namespaces", Namespaced: false, Kind: "Namespace"},
		},
	}
	batchv1 := unversioned.APIResourceList{
		GroupVersion: "batch/v1",
		APIResources: []unversioned.APIResource{
			{Name: "jobs", Namespaced: true, Kind: "Job"},
		},
	}
	batchv2alpha1 := unversioned.APIResourceList{
		GroupVersion: "batch/v2alpha1",
		APIResources: []unversioned.APIResource{
			{Name: "jobs", Namespaced: true, Kind: "Job"},
			{Name: "cronjobs", Namespaced: true, Kind: "CronJob"},
		},
	}
	batchv3alpha1 := unversioned.APIResourceList{
		GroupVersion: "batch/v3alpha1",
		APIResources: []unversioned.APIResource{
			{Name: "jobs", Namespaced: true, Kind: "Job"},
			{Name: "cronjobs", Namespaced: true, Kind: "CronJob"},
		},
	}
	tests := []struct {
		response func(w http.ResponseWriter, req *http.Request)
		expected []unversioned.GroupVersionResource
	}{
		{
			response: func(w http.ResponseWriter, req *http.Request) {
				var list interface{}
				switch req.URL.Path {
				case "/api/v1":
					list = &stable
				case "/api":
					list = &unversioned.APIVersions{
						Versions: []string{
							"v1",
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
			},
			expected: []unversioned.GroupVersionResource{
				{Group: "", Version: "v1", Resource: "pods"},
				{Group: "", Version: "v1", Resource: "services"},
			},
		},
		{
			response: func(w http.ResponseWriter, req *http.Request) {
				var list interface{}
				switch req.URL.Path {
				case "/apis":
					list = &unversioned.APIGroupList{
						Groups: []unversioned.APIGroup{
							{
								Name: "batch",
								Versions: []unversioned.GroupVersionForDiscovery{
									{GroupVersion: "batch/v1", Version: "v1"},
									{GroupVersion: "batch/v2alpha1", Version: "v2alpha1"},
									{GroupVersion: "batch/v3alpha1", Version: "v3alpha1"},
								},
								PreferredVersion: unversioned.GroupVersionForDiscovery{GroupVersion: "batch/v1", Version: "v1"},
							},
						},
					}
				case "/apis/batch/v1":
					list = &batchv1
				case "/apis/batch/v2alpha1":
					list = &batchv2alpha1
				case "/apis/batch/v3alpha1":
					list = &batchv3alpha1
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
			},
			expected: []unversioned.GroupVersionResource{
				{Group: "batch", Version: "v1", Resource: "jobs"},
				{Group: "batch", Version: "v2alpha1", Resource: "cronjobs"},
			},
		},
		{
			response: func(w http.ResponseWriter, req *http.Request) {
				var list interface{}
				switch req.URL.Path {
				case "/apis":
					list = &unversioned.APIGroupList{
						Groups: []unversioned.APIGroup{
							{
								Name: "batch",
								Versions: []unversioned.GroupVersionForDiscovery{
									{GroupVersion: "batch/v1", Version: "v1"},
									{GroupVersion: "batch/v2alpha1", Version: "v2alpha1"},
									{GroupVersion: "batch/v3alpha1", Version: "v3alpha1"},
								},
								PreferredVersion: unversioned.GroupVersionForDiscovery{GroupVersion: "batch/v2alpha", Version: "v2alpha1"},
							},
						},
					}
				case "/apis/batch/v1":
					list = &batchv1
				case "/apis/batch/v2alpha1":
					list = &batchv2alpha1
				case "/apis/batch/v3alpha1":
					list = &batchv3alpha1
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
			},
			expected: []unversioned.GroupVersionResource{
				{Group: "batch", Version: "v2alpha1", Resource: "jobs"},
				{Group: "batch", Version: "v2alpha1", Resource: "cronjobs"},
			},
		},
	}
	for _, test := range tests {
		server := httptest.NewServer(http.HandlerFunc(test.response))
		defer server.Close()

		client := NewDiscoveryClientForConfigOrDie(&rest.Config{Host: server.URL})
		got, err := client.ServerPreferredNamespacedResources()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		// we need deterministic order and since during processing in ServerPreferredNamespacedResources
		// a map comes into play the result needs sorting
		if !reflect.DeepEqual(got, test.expected) {
			t.Errorf("expected:\n%v\ngot:\n%v\n", test.expected, got)
		}
		server.Close()
	}
}
