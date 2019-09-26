/*
Copyright 2017 The Kubernetes Authors.

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
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"sigs.k8s.io/yaml"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/conversion"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/tools/cache"
)

func TestConvertFieldLabel(t *testing.T) {
	tests := []struct {
		name          string
		clusterScoped bool
		label         string
		expectError   bool
	}{
		{
			name:          "cluster scoped - name is ok",
			clusterScoped: true,
			label:         "metadata.name",
		},
		{
			name:          "cluster scoped - namespace is not ok",
			clusterScoped: true,
			label:         "metadata.namespace",
			expectError:   true,
		},
		{
			name:          "cluster scoped - other field is not ok",
			clusterScoped: true,
			label:         "some.other.field",
			expectError:   true,
		},
		{
			name:  "namespace scoped - name is ok",
			label: "metadata.name",
		},
		{
			name:  "namespace scoped - namespace is ok",
			label: "metadata.namespace",
		},
		{
			name:        "namespace scoped - other field is not ok",
			label:       "some.other.field",
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			crd := apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: "None",
					},
				},
			}

			if test.clusterScoped {
				crd.Spec.Scope = apiextensions.ClusterScoped
			} else {
				crd.Spec.Scope = apiextensions.NamespaceScoped
			}
			f, err := conversion.NewCRConverterFactory(nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			_, c, err := f.NewConverter(&crd)
			if err != nil {
				t.Fatalf("Failed to create CR converter. error: %v", err)
			}

			label, value, err := c.ConvertFieldLabel(schema.GroupVersionKind{}, test.label, "value")
			if e, a := test.expectError, err != nil; e != a {
				t.Fatalf("err: expected %t, got %t", e, a)
			}
			if test.expectError {
				if e, a := "field label not supported: "+test.label, err.Error(); e != a {
					t.Errorf("err: expected %s, got %s", e, a)
				}
				return
			}

			if e, a := test.label, label; e != a {
				t.Errorf("label: expected %s, got %s", e, a)
			}
			if e, a := "value", value; e != a {
				t.Errorf("value: expected %s, got %s", e, a)
			}
		})
	}
}

func TestRouting(t *testing.T) {
	hasSynced := false

	crdIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	crdLister := listers.NewCustomResourceDefinitionLister(crdIndexer)

	delegateCalled := false
	delegate := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		delegateCalled = true
		http.Error(w, "", 418)
	})
	customV1 := schema.GroupVersion{Group: "custom", Version: "v1"}
	handler := &crdHandler{
		crdLister: crdLister,
		hasSynced: func() bool { return hasSynced },
		delegate:  delegate,
		versionDiscoveryHandler: &versionDiscoveryHandler{
			discovery: map[schema.GroupVersion]*discovery.APIVersionHandler{
				customV1: discovery.NewAPIVersionHandler(Codecs, customV1, discovery.APIResourceListerFunc(func() []metav1.APIResource {
					return nil
				})),
			},
			delegate: delegate,
		},
		groupDiscoveryHandler: &groupDiscoveryHandler{
			discovery: map[string]*discovery.APIGroupHandler{
				"custom": discovery.NewAPIGroupHandler(Codecs, metav1.APIGroup{
					Name:             customV1.Group,
					Versions:         []metav1.GroupVersionForDiscovery{{GroupVersion: customV1.String(), Version: customV1.Version}},
					PreferredVersion: metav1.GroupVersionForDiscovery{GroupVersion: customV1.String(), Version: customV1.Version},
				}),
			},
			delegate: delegate,
		},
	}

	testcases := []struct {
		Name    string
		Method  string
		Path    string
		Headers map[string]string
		Body    io.Reader

		APIGroup          string
		APIVersion        string
		Verb              string
		Resource          string
		IsResourceRequest bool

		HasSynced bool

		ExpectStatus         int
		ExpectResponse       func(*testing.T, *http.Response, []byte)
		ExpectDelegateCalled bool
	}{
		{
			Name:                 "existing group discovery, presync",
			Method:               "GET",
			Path:                 "/apis/custom",
			APIGroup:             "custom",
			APIVersion:           "",
			HasSynced:            false,
			IsResourceRequest:    false,
			ExpectDelegateCalled: false,
			ExpectStatus:         503,
		},
		{
			Name:                 "existing group discovery",
			Method:               "GET",
			Path:                 "/apis/custom",
			APIGroup:             "custom",
			APIVersion:           "",
			HasSynced:            true,
			IsResourceRequest:    false,
			ExpectDelegateCalled: false,
			ExpectStatus:         200,
		},

		{
			Name:                 "nonexisting group discovery, presync",
			Method:               "GET",
			Path:                 "/apis/other",
			APIGroup:             "other",
			APIVersion:           "",
			HasSynced:            false,
			IsResourceRequest:    false,
			ExpectDelegateCalled: false,
			ExpectStatus:         503,
		},
		{
			Name:                 "nonexisting group discovery",
			Method:               "GET",
			Path:                 "/apis/other",
			APIGroup:             "other",
			APIVersion:           "",
			HasSynced:            true,
			IsResourceRequest:    false,
			ExpectDelegateCalled: true,
			ExpectStatus:         418,
		},

		{
			Name:                 "existing group version discovery, presync",
			Method:               "GET",
			Path:                 "/apis/custom/v1",
			APIGroup:             "custom",
			APIVersion:           "v1",
			HasSynced:            false,
			IsResourceRequest:    false,
			ExpectDelegateCalled: false,
			ExpectStatus:         503,
		},
		{
			Name:                 "existing group version discovery",
			Method:               "GET",
			Path:                 "/apis/custom/v1",
			APIGroup:             "custom",
			APIVersion:           "v1",
			HasSynced:            true,
			IsResourceRequest:    false,
			ExpectDelegateCalled: false,
			ExpectStatus:         200,
		},

		{
			Name:                 "nonexisting group version discovery, presync",
			Method:               "GET",
			Path:                 "/apis/other/v1",
			APIGroup:             "other",
			APIVersion:           "v1",
			HasSynced:            false,
			IsResourceRequest:    false,
			ExpectDelegateCalled: false,
			ExpectStatus:         503,
		},
		{
			Name:                 "nonexisting group version discovery",
			Method:               "GET",
			Path:                 "/apis/other/v1",
			APIGroup:             "other",
			APIVersion:           "v1",
			HasSynced:            true,
			IsResourceRequest:    false,
			ExpectDelegateCalled: true,
			ExpectStatus:         418,
		},

		{
			Name:                 "existing group, nonexisting version discovery, presync",
			Method:               "GET",
			Path:                 "/apis/custom/v2",
			APIGroup:             "custom",
			APIVersion:           "v2",
			HasSynced:            false,
			IsResourceRequest:    false,
			ExpectDelegateCalled: false,
			ExpectStatus:         503,
		},
		{
			Name:                 "existing group, nonexisting version discovery",
			Method:               "GET",
			Path:                 "/apis/custom/v2",
			APIGroup:             "custom",
			APIVersion:           "v2",
			HasSynced:            true,
			IsResourceRequest:    false,
			ExpectDelegateCalled: true,
			ExpectStatus:         418,
		},

		{
			Name:                 "nonexisting group, resource request, presync",
			Method:               "GET",
			Path:                 "/apis/custom/v2/foos",
			APIGroup:             "custom",
			APIVersion:           "v2",
			Verb:                 "list",
			Resource:             "foos",
			HasSynced:            false,
			IsResourceRequest:    true,
			ExpectDelegateCalled: false,
			ExpectStatus:         503,
		},
		{
			Name:                 "nonexisting group, resource request",
			Method:               "GET",
			Path:                 "/apis/custom/v2/foos",
			APIGroup:             "custom",
			APIVersion:           "v2",
			Verb:                 "list",
			Resource:             "foos",
			HasSynced:            true,
			IsResourceRequest:    true,
			ExpectDelegateCalled: true,
			ExpectStatus:         418,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			for _, contentType := range []string{"json", "yaml", "proto", "unknown"} {
				t.Run(contentType, func(t *testing.T) {
					delegateCalled = false
					hasSynced = tc.HasSynced

					recorder := httptest.NewRecorder()

					req := httptest.NewRequest(tc.Method, tc.Path, tc.Body)
					for k, v := range tc.Headers {
						req.Header.Set(k, v)
					}

					expectStatus := tc.ExpectStatus
					switch contentType {
					case "json":
						req.Header.Set("Accept", "application/json")
					case "yaml":
						req.Header.Set("Accept", "application/yaml")
					case "proto":
						req.Header.Set("Accept", "application/vnd.kubernetes.protobuf, application/json")
					case "unknown":
						req.Header.Set("Accept", "application/vnd.kubernetes.unknown")
						// rather than success, we'll get a not supported error
						if expectStatus == 200 {
							expectStatus = 406
						}
					default:
						t.Fatalf("unknown content type %v", contentType)
					}

					req = req.WithContext(apirequest.WithRequestInfo(req.Context(), &apirequest.RequestInfo{
						Verb:              tc.Verb,
						Resource:          tc.Resource,
						APIGroup:          tc.APIGroup,
						APIVersion:        tc.APIVersion,
						IsResourceRequest: tc.IsResourceRequest,
						Path:              tc.Path,
					}))

					handler.ServeHTTP(recorder, req)

					if tc.ExpectDelegateCalled != delegateCalled {
						t.Errorf("expected delegated called %v, got %v", tc.ExpectDelegateCalled, delegateCalled)
					}
					result := recorder.Result()
					content, _ := ioutil.ReadAll(result.Body)
					if e, a := expectStatus, result.StatusCode; e != a {
						t.Log(string(content))
						t.Errorf("expected %v, got %v", e, a)
					}
					if tc.ExpectResponse != nil {
						tc.ExpectResponse(t, result, content)
					}

					// Make sure error responses come back with status objects in all encodings, including unknown encodings
					if !delegateCalled && expectStatus >= 300 {
						status := &metav1.Status{}

						switch contentType {
						// unknown accept headers fall back to json errors
						case "json", "unknown":
							if e, a := "application/json", result.Header.Get("Content-Type"); e != a {
								t.Errorf("expected Content-Type %v, got %v", e, a)
							}
							if err := json.Unmarshal(content, status); err != nil {
								t.Fatal(err)
							}
						case "yaml":
							if e, a := "application/yaml", result.Header.Get("Content-Type"); e != a {
								t.Errorf("expected Content-Type %v, got %v", e, a)
							}
							if err := yaml.Unmarshal(content, status); err != nil {
								t.Fatal(err)
							}
						case "proto":
							if e, a := "application/vnd.kubernetes.protobuf", result.Header.Get("Content-Type"); e != a {
								t.Errorf("expected Content-Type %v, got %v", e, a)
							}
							if _, _, err := protobuf.NewSerializer(Scheme, Scheme).Decode(content, nil, status); err != nil {
								t.Fatal(err)
							}
						default:
							t.Fatalf("unknown content type %v", contentType)
						}

						if e, a := metav1.Unversioned.WithKind("Status"), status.GroupVersionKind(); e != a {
							t.Errorf("expected %#v, got %#v", e, a)
						}
						if int(status.Code) != int(expectStatus) {
							t.Errorf("expected %v, got %v", expectStatus, status.Code)
						}
					}
				})
			}
		})
	}
}
