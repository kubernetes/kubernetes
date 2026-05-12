/*
Copyright 2023 The Kubernetes Authors.

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

package openapi

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
)

func TestGroupVersion(t *testing.T) {
	tests := []struct {
		name                string
		prefix              string
		serverReturnsPrefix bool
	}{
		{
			name:                "no prefix",
			prefix:              "",
			serverReturnsPrefix: false,
		},
		{
			name:                "prefix not in discovery",
			prefix:              "/test-endpoint",
			serverReturnsPrefix: false,
		},
		{
			name:                "prefix in discovery",
			prefix:              "/test-endpoint",
			serverReturnsPrefix: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.URL.Path == test.prefix+"/openapi/v3/apis/apps/v1" && r.URL.RawQuery == "hash=014fbff9a07c":
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					w.Write([]byte(`{"openapi":"3.0.0","info":{"title":"Kubernetes","version":"unversioned"}}`))
				case r.URL.Path == test.prefix+"/openapi/v3":
					// return root content
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					if test.serverReturnsPrefix {
						w.Write([]byte(fmt.Sprintf(`{"paths":{"apis/apps/v1":{"serverRelativeURL":"%s/openapi/v3/apis/apps/v1?hash=014fbff9a07c"}}}`, test.prefix)))
					} else {
						w.Write([]byte(`{"paths":{"apis/apps/v1":{"serverRelativeURL":"/openapi/v3/apis/apps/v1?hash=014fbff9a07c"}}}`))
					}
				default:
					t.Errorf("unexpected request: %s", r.URL.String())
					w.WriteHeader(http.StatusNotFound)
					return
				}
			}))
			defer server.Close()

			c, err := rest.RESTClientFor(&rest.Config{
				Host: server.URL + test.prefix,
				ContentConfig: rest.ContentConfig{
					NegotiatedSerializer: scheme.Codecs,
					GroupVersion:         &appsv1.SchemeGroupVersion,
				},
			})

			if err != nil {
				t.Fatalf("unexpected error occurred: %v", err)
			}

			openapiClient := NewClient(c)
			paths, err := openapiClient.Paths()
			if err != nil {
				t.Fatalf("unexpected error occurred: %v", err)
			}
			schema, err := paths["apis/apps/v1"].Schema(runtime.ContentTypeJSON)
			if err != nil {
				t.Fatalf("unexpected error occurred: %v", err)
			}
			expectedResult := `{"openapi":"3.0.0","info":{"title":"Kubernetes","version":"unversioned"}}`
			if string(schema) != expectedResult {
				t.Fatalf("unexpected result actual: %s expected: %s", string(schema), expectedResult)
			}
		})
	}
}
