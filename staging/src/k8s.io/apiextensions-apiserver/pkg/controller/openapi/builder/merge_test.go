/*
Copyright 2022 The Kubernetes Authors.

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

package builder

import (
	"reflect"
	"testing"

	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestMergeSpecV3(t *testing.T) {
	tests := []struct {
		name     string
		specs    []*spec3.OpenAPI
		expected *spec3.OpenAPI
	}{
		{
			name: "oneCRD",
			specs: []*spec3.OpenAPI{{
				Paths: &spec3.Paths{
					Paths: map[string]*spec3.Path{
						"/apis/stable.example.com/v1/crd1": {},
					},
				},
				Components: &spec3.Components{
					Schemas: map[string]*spec.Schema{
						"io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta": {},
					},
				},
			},
			},
			expected: &spec3.OpenAPI{
				Paths: &spec3.Paths{
					Paths: map[string]*spec3.Path{
						"/apis/stable.example.com/v1/crd1": {},
					},
				},
				Components: &spec3.Components{
					Schemas: map[string]*spec.Schema{
						"io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta": {},
					},
				},
			},
		},
		{
			name: "two CRDs same gv",
			specs: []*spec3.OpenAPI{{
				Paths: &spec3.Paths{
					Paths: map[string]*spec3.Path{
						"/apis/stable.example.com/v1/crd1": {},
					},
				},
				Components: &spec3.Components{
					Schemas: map[string]*spec.Schema{
						"com.example.stable.v1.CRD1": {},

						"io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta": {},
					},
				},
			},
				{
					Paths: &spec3.Paths{
						Paths: map[string]*spec3.Path{
							"/apis/stable.example.com/v1/crd2": {},
						},
					},
					Components: &spec3.Components{
						Schemas: map[string]*spec.Schema{
							"com.example.stable.v1.CRD2":                      {},
							"io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta": {},
						},
					},
				},
			},
			expected: &spec3.OpenAPI{
				Paths: &spec3.Paths{
					Paths: map[string]*spec3.Path{
						"/apis/stable.example.com/v1/crd1": {},
						"/apis/stable.example.com/v1/crd2": {},
					},
				},
				Components: &spec3.Components{
					Schemas: map[string]*spec.Schema{
						"io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta": {},
						"com.example.stable.v1.CRD1":                      {},
						"com.example.stable.v1.CRD2":                      {},
					},
				},
			},
		},
		{
			name: "two CRDs with scale",
			specs: []*spec3.OpenAPI{{
				Paths: &spec3.Paths{
					Paths: map[string]*spec3.Path{
						"/apis/stable.example.com/v1/crd1": {},
					},
				},
				Components: &spec3.Components{
					Schemas: map[string]*spec.Schema{
						"com.example.stable.v1.CRD1": {},

						"io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta": {},
						"io.k8s.api.autoscaling.v1.Scale":                 {},
					},
				},
			},
				{
					Paths: &spec3.Paths{
						Paths: map[string]*spec3.Path{
							"/apis/stable.example.com/v1/crd2": {},
						},
					},
					Components: &spec3.Components{
						Schemas: map[string]*spec.Schema{
							"com.example.stable.v1.CRD2":                      {},
							"io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta": {},
							"io.k8s.api.autoscaling.v1.Scale":                 {},
						},
					},
				},
			},
			expected: &spec3.OpenAPI{
				Paths: &spec3.Paths{
					Paths: map[string]*spec3.Path{
						"/apis/stable.example.com/v1/crd1": {},
						"/apis/stable.example.com/v1/crd2": {},
					},
				},
				Components: &spec3.Components{
					Schemas: map[string]*spec.Schema{
						"io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta": {},
						"com.example.stable.v1.CRD1":                      {},
						"com.example.stable.v1.CRD2":                      {},
						"io.k8s.api.autoscaling.v1.Scale":                 {},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			merged, err := MergeSpecsV3(tt.specs...)
			if err != nil {
				t.Error(err)
			}
			if !reflect.DeepEqual(merged, tt.expected) {
				t.Error("Merged spec is different from expected spec")
			}

		})
	}
}
