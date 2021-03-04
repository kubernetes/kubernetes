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
	"fmt"
	"mime"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/version"
	restclient "k8s.io/client-go/rest"
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
	client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})

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
			obj = &metav1.APIVersions{
				Versions: []string{
					"v1",
				},
			}
		case "/apis":
			obj = &metav1.APIGroupList{
				Groups: []metav1.APIGroup{
					{
						Name: "extensions",
						Versions: []metav1.GroupVersionForDiscovery{
							{GroupVersion: "extensions/v1beta1"},
						},
					},
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
	client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
	// ServerGroups should not return an error even if server returns error at /api and /apis
	apiGroupList, err := client.ServerGroups()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	groupVersions := metav1.ExtractGroupVersions(apiGroupList)
	if !reflect.DeepEqual(groupVersions, []string{"v1", "extensions/v1beta1"}) {
		t.Errorf("expected: %q, got: %q", []string{"v1", "extensions/v1beta1"}, groupVersions)
	}
}

func TestGetServerGroupsWithBrokenServer(t *testing.T) {
	for _, statusCode := range []int{http.StatusNotFound, http.StatusForbidden} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.WriteHeader(statusCode)
		}))
		defer server.Close()
		client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
		// ServerGroups should not return an error even if server returns Not Found or Forbidden error at all end points
		apiGroupList, err := client.ServerGroups()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		groupVersions := metav1.ExtractGroupVersions(apiGroupList)
		if len(groupVersions) != 0 {
			t.Errorf("expected empty list, got: %q", groupVersions)
		}
	}
}

func TestTimeoutIsSet(t *testing.T) {
	cfg := &restclient.Config{}
	setDiscoveryDefaults(cfg)
	assert.Equal(t, defaultTimeout, cfg.Timeout)
}

func TestGetServerResourcesWithV1Server(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var obj interface{}
		switch req.URL.Path {
		case "/api":
			obj = &metav1.APIVersions{
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
	client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
	// ServerResources should not return an error even if server returns error at /api/v1.
	serverResources, err := client.ServerResources()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	gvs := groupVersions(serverResources)
	if !sets.NewString(gvs...).Has("v1") {
		t.Errorf("missing v1 in resource list: %v", serverResources)
	}
}

func TestGetServerResources(t *testing.T) {
	stable := metav1.APIResourceList{
		GroupVersion: "v1",
		APIResources: []metav1.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
			{Name: "services", Namespaced: true, Kind: "Service"},
			{Name: "namespaces", Namespaced: false, Kind: "Namespace"},
		},
	}
	beta := metav1.APIResourceList{
		GroupVersion: "extensions/v1beta1",
		APIResources: []metav1.APIResource{
			{Name: "deployments", Namespaced: true, Kind: "Deployment"},
			{Name: "ingresses", Namespaced: true, Kind: "Ingress"},
			{Name: "jobs", Namespaced: true, Kind: "Job"},
		},
	}
	beta2 := metav1.APIResourceList{
		GroupVersion: "extensions/v1beta2",
		APIResources: []metav1.APIResource{
			{Name: "deployments", Namespaced: true, Kind: "Deployment"},
			{Name: "ingresses", Namespaced: true, Kind: "Ingress"},
			{Name: "jobs", Namespaced: true, Kind: "Job"},
		},
	}
	extensionsbeta3 := metav1.APIResourceList{GroupVersion: "extensions/v1beta3", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	extensionsbeta4 := metav1.APIResourceList{GroupVersion: "extensions/v1beta4", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	extensionsbeta5 := metav1.APIResourceList{GroupVersion: "extensions/v1beta5", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	extensionsbeta6 := metav1.APIResourceList{GroupVersion: "extensions/v1beta6", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	extensionsbeta7 := metav1.APIResourceList{GroupVersion: "extensions/v1beta7", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	extensionsbeta8 := metav1.APIResourceList{GroupVersion: "extensions/v1beta8", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	extensionsbeta9 := metav1.APIResourceList{GroupVersion: "extensions/v1beta9", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	extensionsbeta10 := metav1.APIResourceList{GroupVersion: "extensions/v1beta10", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}

	appsbeta1 := metav1.APIResourceList{GroupVersion: "apps/v1beta1", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta2 := metav1.APIResourceList{GroupVersion: "apps/v1beta2", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta3 := metav1.APIResourceList{GroupVersion: "apps/v1beta3", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta4 := metav1.APIResourceList{GroupVersion: "apps/v1beta4", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta5 := metav1.APIResourceList{GroupVersion: "apps/v1beta5", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta6 := metav1.APIResourceList{GroupVersion: "apps/v1beta6", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta7 := metav1.APIResourceList{GroupVersion: "apps/v1beta7", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta8 := metav1.APIResourceList{GroupVersion: "apps/v1beta8", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta9 := metav1.APIResourceList{GroupVersion: "apps/v1beta9", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}
	appsbeta10 := metav1.APIResourceList{GroupVersion: "apps/v1beta10", APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}}}

	tests := []struct {
		resourcesList *metav1.APIResourceList
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
		case "/apis/extensions/v1beta2":
			list = &beta2
		case "/apis/extensions/v1beta3":
			list = &extensionsbeta3
		case "/apis/extensions/v1beta4":
			list = &extensionsbeta4
		case "/apis/extensions/v1beta5":
			list = &extensionsbeta5
		case "/apis/extensions/v1beta6":
			list = &extensionsbeta6
		case "/apis/extensions/v1beta7":
			list = &extensionsbeta7
		case "/apis/extensions/v1beta8":
			list = &extensionsbeta8
		case "/apis/extensions/v1beta9":
			list = &extensionsbeta9
		case "/apis/extensions/v1beta10":
			list = &extensionsbeta10
		case "/apis/apps/v1beta1":
			list = &appsbeta1
		case "/apis/apps/v1beta2":
			list = &appsbeta2
		case "/apis/apps/v1beta3":
			list = &appsbeta3
		case "/apis/apps/v1beta4":
			list = &appsbeta4
		case "/apis/apps/v1beta5":
			list = &appsbeta5
		case "/apis/apps/v1beta6":
			list = &appsbeta6
		case "/apis/apps/v1beta7":
			list = &appsbeta7
		case "/apis/apps/v1beta8":
			list = &appsbeta8
		case "/apis/apps/v1beta9":
			list = &appsbeta9
		case "/apis/apps/v1beta10":
			list = &appsbeta10
		case "/api":
			list = &metav1.APIVersions{
				Versions: []string{
					"v1",
				},
			}
		case "/apis":
			list = &metav1.APIGroupList{
				Groups: []metav1.APIGroup{
					{
						Name: "apps",
						Versions: []metav1.GroupVersionForDiscovery{
							{GroupVersion: "apps/v1beta1", Version: "v1beta1"},
							{GroupVersion: "apps/v1beta2", Version: "v1beta2"},
							{GroupVersion: "apps/v1beta3", Version: "v1beta3"},
							{GroupVersion: "apps/v1beta4", Version: "v1beta4"},
							{GroupVersion: "apps/v1beta5", Version: "v1beta5"},
							{GroupVersion: "apps/v1beta6", Version: "v1beta6"},
							{GroupVersion: "apps/v1beta7", Version: "v1beta7"},
							{GroupVersion: "apps/v1beta8", Version: "v1beta8"},
							{GroupVersion: "apps/v1beta9", Version: "v1beta9"},
							{GroupVersion: "apps/v1beta10", Version: "v1beta10"},
						},
					},
					{
						Name: "extensions",
						Versions: []metav1.GroupVersionForDiscovery{
							{GroupVersion: "extensions/v1beta1", Version: "v1beta1"},
							{GroupVersion: "extensions/v1beta2", Version: "v1beta2"},
							{GroupVersion: "extensions/v1beta3", Version: "v1beta3"},
							{GroupVersion: "extensions/v1beta4", Version: "v1beta4"},
							{GroupVersion: "extensions/v1beta5", Version: "v1beta5"},
							{GroupVersion: "extensions/v1beta6", Version: "v1beta6"},
							{GroupVersion: "extensions/v1beta7", Version: "v1beta7"},
							{GroupVersion: "extensions/v1beta8", Version: "v1beta8"},
							{GroupVersion: "extensions/v1beta9", Version: "v1beta9"},
							{GroupVersion: "extensions/v1beta10", Version: "v1beta10"},
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
	for _, test := range tests {
		client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
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

	client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
	start := time.Now()
	serverResources, err := client.ServerResources()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	end := time.Now()
	if d := end.Sub(start); d > time.Second {
		t.Errorf("took too long to perform discovery: %s", d)
	}
	serverGroupVersions := groupVersions(serverResources)
	expectedGroupVersions := []string{
		"v1",
		"apps/v1beta1",
		"apps/v1beta2",
		"apps/v1beta3",
		"apps/v1beta4",
		"apps/v1beta5",
		"apps/v1beta6",
		"apps/v1beta7",
		"apps/v1beta8",
		"apps/v1beta9",
		"apps/v1beta10",
		"extensions/v1beta1",
		"extensions/v1beta2",
		"extensions/v1beta3",
		"extensions/v1beta4",
		"extensions/v1beta5",
		"extensions/v1beta6",
		"extensions/v1beta7",
		"extensions/v1beta8",
		"extensions/v1beta9",
		"extensions/v1beta10",
	}
	if !reflect.DeepEqual(expectedGroupVersions, serverGroupVersions) {
		t.Errorf("unexpected group versions: %v", diff.ObjectReflectDiff(expectedGroupVersions, serverGroupVersions))
	}
}

func returnedOpenAPI() *openapi_v2.Document {
	return &openapi_v2.Document{
		Definitions: &openapi_v2.Definitions{
			AdditionalProperties: []*openapi_v2.NamedSchema{
				{
					Name: "fake.type.1",
					Value: &openapi_v2.Schema{
						Properties: &openapi_v2.Properties{
							AdditionalProperties: []*openapi_v2.NamedSchema{
								{
									Name: "count",
									Value: &openapi_v2.Schema{
										Type: &openapi_v2.TypeItem{
											Value: []string{"integer"},
										},
									},
								},
							},
						},
					},
				},
				{
					Name: "fake.type.2",
					Value: &openapi_v2.Schema{
						Properties: &openapi_v2.Properties{
							AdditionalProperties: []*openapi_v2.NamedSchema{
								{
									Name: "count",
									Value: &openapi_v2.Schema{
										Type: &openapi_v2.TypeItem{
											Value: []string{"array"},
										},
										Items: &openapi_v2.ItemsItem{
											Schema: []*openapi_v2.Schema{
												{
													Type: &openapi_v2.TypeItem{
														Value: []string{"string"},
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
}

func openapiSchemaDeprecatedFakeServer(status int, t *testing.T) (*httptest.Server, error) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/openapi/v2" {
			// write the error status for the new endpoint request
			w.WriteHeader(status)
			return
		}
		if req.URL.Path != "/swagger-2.0.0.pb-v1" {
			errMsg := fmt.Sprintf("Unexpected url %v", req.URL)
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte(errMsg))
			t.Errorf("testing should fail as %s", errMsg)
			return
		}
		if req.Method != "GET" {
			errMsg := fmt.Sprintf("Unexpected method %v", req.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
			w.Write([]byte(errMsg))
			t.Errorf("testing should fail as %s", errMsg)
			return
		}

		mime.AddExtensionType(".pb-v1", "application/com.github.googleapis.gnostic.OpenAPIv2@68f4ded+protobuf")

		output, err := proto.Marshal(returnedOpenAPI())
		if err != nil {
			errMsg := fmt.Sprintf("Unexpected marshal error: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(errMsg))
			t.Errorf("testing should fail as %s", errMsg)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))

	return server, nil
}

func openapiSchemaFakeServer(t *testing.T) (*httptest.Server, error) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/openapi/v2" {
			errMsg := fmt.Sprintf("Unexpected url %v", req.URL)
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte(errMsg))
			t.Errorf("testing should fail as %s", errMsg)
			return
		}
		if req.Method != "GET" {
			errMsg := fmt.Sprintf("Unexpected method %v", req.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
			w.Write([]byte(errMsg))
			t.Errorf("testing should fail as %s", errMsg)
			return
		}
		decipherableFormat := req.Header.Get("Accept")
		if decipherableFormat != "application/com.github.proto-openapi.spec.v2@v1.0+protobuf" {
			errMsg := fmt.Sprintf("Unexpected accept mime type %v", decipherableFormat)
			w.WriteHeader(http.StatusUnsupportedMediaType)
			w.Write([]byte(errMsg))
			t.Errorf("testing should fail as %s", errMsg)
			return
		}

		mime.AddExtensionType(".pb-v1", "application/com.github.googleapis.gnostic.OpenAPIv2@68f4ded+protobuf")

		output, err := proto.Marshal(returnedOpenAPI())
		if err != nil {
			errMsg := fmt.Sprintf("Unexpected marshal error: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(errMsg))
			t.Errorf("testing should fail as %s", errMsg)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))

	return server, nil
}

func TestGetOpenAPISchema(t *testing.T) {
	server, err := openapiSchemaFakeServer(t)
	if err != nil {
		t.Errorf("unexpected error starting fake server: %v", err)
	}
	defer server.Close()

	client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
	got, err := client.OpenAPISchema()
	if err != nil {
		t.Fatalf("unexpected error getting openapi: %v", err)
	}
	if e, a := returnedOpenAPI(), got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGetOpenAPISchemaForbiddenFallback(t *testing.T) {
	server, err := openapiSchemaDeprecatedFakeServer(http.StatusForbidden, t)
	if err != nil {
		t.Errorf("unexpected error starting fake server: %v", err)
	}
	defer server.Close()

	client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
	got, err := client.OpenAPISchema()
	if err != nil {
		t.Fatalf("unexpected error getting openapi: %v", err)
	}
	if e, a := returnedOpenAPI(), got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGetOpenAPISchemaNotFoundFallback(t *testing.T) {
	server, err := openapiSchemaDeprecatedFakeServer(http.StatusNotFound, t)
	if err != nil {
		t.Errorf("unexpected error starting fake server: %v", err)
	}
	defer server.Close()

	client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
	got, err := client.OpenAPISchema()
	if err != nil {
		t.Fatalf("unexpected error getting openapi: %v", err)
	}
	if e, a := returnedOpenAPI(), got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGetOpenAPISchemaNotAcceptableFallback(t *testing.T) {
	server, err := openapiSchemaDeprecatedFakeServer(http.StatusNotAcceptable, t)
	if err != nil {
		t.Errorf("unexpected error starting fake server: %v", err)
	}
	defer server.Close()

	client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
	got, err := client.OpenAPISchema()
	if err != nil {
		t.Fatalf("unexpected error getting openapi: %v", err)
	}
	if e, a := returnedOpenAPI(), got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestServerPreferredResources(t *testing.T) {
	stable := metav1.APIResourceList{
		GroupVersion: "v1",
		APIResources: []metav1.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
			{Name: "services", Namespaced: true, Kind: "Service"},
			{Name: "namespaces", Namespaced: false, Kind: "Namespace"},
		},
	}
	tests := []struct {
		resourcesList []*metav1.APIResourceList
		response      func(w http.ResponseWriter, req *http.Request)
		expectErr     func(err error) bool
	}{
		{
			resourcesList: []*metav1.APIResourceList{&stable},
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
					list = &metav1.APIVersions{
						Versions: []string{
							"v1",
						},
					}
				case "/apis":
					list = &metav1.APIGroupList{
						Groups: []metav1.APIGroup{
							{
								Versions: []metav1.GroupVersionForDiscovery{
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
					list = &metav1.APIVersions{
						Versions: []string{
							"v1",
						},
					}
				case "/apis":
					list = &metav1.APIGroupList{
						Groups: []metav1.APIGroup{
							{
								Versions: []metav1.GroupVersionForDiscovery{
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

		client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
		resources, err := client.ServerPreferredResources()
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
		got, err := GroupVersionResources(resources)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		expected, _ := GroupVersionResources(test.resourcesList)
		if !reflect.DeepEqual(got, expected) {
			t.Errorf("expected:\n%v\ngot:\n%v\n", test.resourcesList, got)
		}
		server.Close()
	}
}

func TestServerPreferredResourcesRetries(t *testing.T) {
	stable := metav1.APIResourceList{
		GroupVersion: "v1",
		APIResources: []metav1.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
		},
	}
	beta := metav1.APIResourceList{
		GroupVersion: "extensions/v1",
		APIResources: []metav1.APIResource{
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
				list = &metav1.APIVersions{
					Versions: []string{
						"v1",
					},
				}
			case "/apis":
				list = &metav1.APIGroupList{
					Groups: []metav1.APIGroup{
						{
							Name: "extensions",
							Versions: []metav1.GroupVersionForDiscovery{
								{GroupVersion: "extensions/v1beta1", Version: "v1beta1"},
							},
							PreferredVersion: metav1.GroupVersionForDiscovery{
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

		client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
		resources, err := client.ServerPreferredResources()
		if !tc.expectedError(err) {
			t.Errorf("case %d: unexpected error: %v", i, err)
		}
		got, err := GroupVersionResources(resources)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
		}
		if len(got) != tc.expectResources {
			t.Errorf("case %d: expect %d resources, got %#v", i, tc.expectResources, got)
		}
		server.Close()
	}
}

func TestServerPreferredNamespacedResources(t *testing.T) {
	stable := metav1.APIResourceList{
		GroupVersion: "v1",
		APIResources: []metav1.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
			{Name: "services", Namespaced: true, Kind: "Service"},
			{Name: "namespaces", Namespaced: false, Kind: "Namespace"},
		},
	}
	batchv1 := metav1.APIResourceList{
		GroupVersion: "batch/v1",
		APIResources: []metav1.APIResource{
			{Name: "jobs", Namespaced: true, Kind: "Job"},
		},
	}
	batchv2alpha1 := metav1.APIResourceList{
		GroupVersion: "batch/v2alpha1",
		APIResources: []metav1.APIResource{
			{Name: "jobs", Namespaced: true, Kind: "Job"},
			{Name: "cronjobs", Namespaced: true, Kind: "CronJob"},
		},
	}
	batchv3alpha1 := metav1.APIResourceList{
		GroupVersion: "batch/v3alpha1",
		APIResources: []metav1.APIResource{
			{Name: "jobs", Namespaced: true, Kind: "Job"},
			{Name: "cronjobs", Namespaced: true, Kind: "CronJob"},
		},
	}
	tests := []struct {
		response func(w http.ResponseWriter, req *http.Request)
		expected map[schema.GroupVersionResource]struct{}
	}{
		{
			response: func(w http.ResponseWriter, req *http.Request) {
				var list interface{}
				switch req.URL.Path {
				case "/api/v1":
					list = &stable
				case "/api":
					list = &metav1.APIVersions{
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
			expected: map[schema.GroupVersionResource]struct{}{
				{Group: "", Version: "v1", Resource: "pods"}:     {},
				{Group: "", Version: "v1", Resource: "services"}: {},
			},
		},
		{
			response: func(w http.ResponseWriter, req *http.Request) {
				var list interface{}
				switch req.URL.Path {
				case "/apis":
					list = &metav1.APIGroupList{
						Groups: []metav1.APIGroup{
							{
								Name: "batch",
								Versions: []metav1.GroupVersionForDiscovery{
									{GroupVersion: "batch/v1", Version: "v1"},
									{GroupVersion: "batch/v2alpha1", Version: "v2alpha1"},
									{GroupVersion: "batch/v3alpha1", Version: "v3alpha1"},
								},
								PreferredVersion: metav1.GroupVersionForDiscovery{GroupVersion: "batch/v1", Version: "v1"},
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
			expected: map[schema.GroupVersionResource]struct{}{
				{Group: "batch", Version: "v1", Resource: "jobs"}:           {},
				{Group: "batch", Version: "v2alpha1", Resource: "cronjobs"}: {},
			},
		},
		{
			response: func(w http.ResponseWriter, req *http.Request) {
				var list interface{}
				switch req.URL.Path {
				case "/apis":
					list = &metav1.APIGroupList{
						Groups: []metav1.APIGroup{
							{
								Name: "batch",
								Versions: []metav1.GroupVersionForDiscovery{
									{GroupVersion: "batch/v1", Version: "v1"},
									{GroupVersion: "batch/v2alpha1", Version: "v2alpha1"},
									{GroupVersion: "batch/v3alpha1", Version: "v3alpha1"},
								},
								PreferredVersion: metav1.GroupVersionForDiscovery{GroupVersion: "batch/v2alpha", Version: "v2alpha1"},
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
			expected: map[schema.GroupVersionResource]struct{}{
				{Group: "batch", Version: "v2alpha1", Resource: "jobs"}:     {},
				{Group: "batch", Version: "v2alpha1", Resource: "cronjobs"}: {},
			},
		},
	}
	for i, test := range tests {
		server := httptest.NewServer(http.HandlerFunc(test.response))
		defer server.Close()

		client := NewDiscoveryClientForConfigOrDie(&restclient.Config{Host: server.URL})
		resources, err := client.ServerPreferredNamespacedResources()
		if err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
			continue
		}
		got, err := GroupVersionResources(resources)
		if err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
			continue
		}

		if !reflect.DeepEqual(got, test.expected) {
			t.Errorf("[%d] expected:\n%v\ngot:\n%v\n", i, test.expected, got)
		}
		server.Close()
	}
}

func groupVersions(resources []*metav1.APIResourceList) []string {
	result := []string{}
	for _, resourceList := range resources {
		result = append(result, resourceList.GroupVersion)
	}
	return result
}
