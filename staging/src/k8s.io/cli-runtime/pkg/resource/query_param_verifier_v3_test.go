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

package resource

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi/cached"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/client-go/openapi3"
	"k8s.io/kube-openapi/pkg/spec3"
)

func TestV3SupportsQueryParamBatchV1(t *testing.T) {
	tests := map[string]struct {
		crds             []schema.GroupKind      // CRDFinder returns these CRD's
		gvk              schema.GroupVersionKind // GVK whose OpenAPI V3 spec is checked
		queryParam       VerifiableQueryParam    // Usually "fieldValidation"
		expectedSupports bool
	}{
		"Field validation query param is supported for batch/v1/Job": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "batch",
				Version: "v1",
				Kind:    "Job",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: true,
		},
		"Field validation query param supported for core/v1/Namespace": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "Namespace",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: true,
		},
		"Field validation unsupported for unknown GVK": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "bad",
				Version: "v1",
				Kind:    "Uknown",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: false,
		},
		"Unknown query param unsupported (for all GVK's)": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "apps",
				Version: "v1",
				Kind:    "Deployment",
			},
			queryParam:       "UnknownQueryParam",
			expectedSupports: false,
		},
		"Field validation query param supported for found CRD": {
			crds: []schema.GroupKind{
				{
					Group: "example.com",
					Kind:  "ExampleCRD",
				},
			},
			// GVK matches above CRD GroupKind
			gvk: schema.GroupVersionKind{
				Group:   "example.com",
				Version: "v1",
				Kind:    "ExampleCRD",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: true,
		},
		"Field validation query param unsupported for missing CRD": {
			crds: []schema.GroupKind{
				{
					Group: "different.com",
					Kind:  "DifferentCRD",
				},
			},
			// GVK does NOT match above CRD GroupKind
			gvk: schema.GroupVersionKind{
				Group:   "example.com",
				Version: "v1",
				Kind:    "ExampleCRD",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: false,
		},
		"List GVK is specifically unsupported": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "List",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: false,
		},
	}

	root := openapi3.NewRoot(cached.NewClient(openapitest.NewEmbeddedFileClient()))
	for tn, tc := range tests {
		t.Run(tn, func(t *testing.T) {
			verifier := &queryParamVerifierV3{
				finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
					return tc.crds, nil
				}),
				root:       root,
				queryParam: tc.queryParam,
			}
			err := verifier.HasSupport(tc.gvk)
			if tc.expectedSupports && err != nil {
				t.Errorf("Expected supports, but returned err for GVK (%s)", tc.gvk)
			} else if !tc.expectedSupports && err == nil {
				t.Errorf("Expected not supports, but returned no err for GVK (%s)", tc.gvk)
			}
		})
	}
}

func TestInvalidOpenAPIV3Document(t *testing.T) {
	tests := map[string]struct {
		spec *spec3.OpenAPI
		err  string
	}{
		"nil document returns error": {
			spec: nil,
			err:  "Invalid OpenAPI V3 document",
		},
		"empty document returns error": {
			spec: &spec3.OpenAPI{},
			err:  "Invalid OpenAPI V3 document",
		},
		"minimal document returns error": {
			spec: &spec3.OpenAPI{
				Version: "openapi 3.0.0",
				Paths:   nil,
			},
			err: "Invalid OpenAPI V3 document",
		},
		"empty Paths returns error": {
			spec: &spec3.OpenAPI{
				Version: "openapi 3.0.0",
				Paths:   &spec3.Paths{},
			},
			err: "Path not found for GVK",
		},
		"nil Path returns error": {
			spec: &spec3.OpenAPI{
				Version: "openapi 3.0.0",
				Paths:   &spec3.Paths{Paths: map[string]*spec3.Path{"/version": nil}},
			},
			err: "Path not found for GVK",
		},
		"empty Path returns error": {
			spec: &spec3.OpenAPI{
				Version: "openapi 3.0.0",
				Paths:   &spec3.Paths{Paths: map[string]*spec3.Path{"/version": {}}},
			},
			err: "Path not found for GVK",
		},
	}

	gvk := schema.GroupVersionKind{
		Group:   "batch",
		Version: "v1",
		Kind:    "Job",
	}
	for tn, tc := range tests {
		t.Run(tn, func(t *testing.T) {

			verifier := &queryParamVerifierV3{
				finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
					return []schema.GroupKind{}, nil
				}),
				root:       &fakeRoot{tc.spec},
				queryParam: QueryParamFieldValidation,
			}
			err := verifier.HasSupport(gvk)
			if !strings.Contains(err.Error(), tc.err) {
				t.Errorf("Expected error (%s), but received (%s)", tc.err, err.Error())
			}
		})
	}
}

// fakeRoot implements Root interface; manually specifies the returned OpenAPI V3 document.
type fakeRoot struct {
	spec *spec3.OpenAPI
}

func (f *fakeRoot) GroupVersions() ([]schema.GroupVersion, error) {
	// Unused
	return nil, nil
}

// GVSpec returns hard-coded OpenAPI V3 document.
func (f *fakeRoot) GVSpec(gv schema.GroupVersion) (*spec3.OpenAPI, error) {
	return f.spec, nil
}

func (f *fakeRoot) GVSpecAsMap(gv schema.GroupVersion) (map[string]interface{}, error) {
	// Unused
	return nil, nil
}
