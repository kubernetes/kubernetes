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
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/openapi/cached"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/client-go/openapi3"
)

func TestFallbackQueryParamVerifier_PrimaryNoFallback(t *testing.T) {
	tests := map[string]struct {
		crds             []schema.GroupKind      // CRDFinder returns these CRD's
		gvk              schema.GroupVersionKind // GVK whose OpenAPI spec is checked
		queryParam       VerifiableQueryParam    // Usually "fieldValidation"
		expectedSupports bool
	}{
		"Field validation query param is supported for batch/v1/Job, primary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "batch",
				Version: "v1",
				Kind:    "Job",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: true,
		},
		"Field validation query param supported for core/v1/Namespace, primary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "Namespace",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: true,
		},
		"Field validation unsupported for unknown GVK in primary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "bad",
				Version: "v1",
				Kind:    "Uknown",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: false,
		},
		"Unknown query param unsupported (for all GVK's) in primary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "apps",
				Version: "v1",
				Kind:    "Deployment",
			},
			queryParam:       "UnknownQueryParam",
			expectedSupports: false,
		},
		"Field validation query param supported for found CRD in primary verifier": {
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
		"Field validation query param unsupported for missing CRD in primary verifier": {
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
		"List GVK is specifically unsupported in primary verfier": {
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
			primary := createFakeV3Verifier(tc.crds, root, tc.queryParam)
			// secondary verifier should not be called.
			secondary := &failingVerifier{name: "secondary", t: t}
			verifier := NewFallbackQueryParamVerifier(primary, secondary)
			err := verifier.HasSupport(tc.gvk)
			if tc.expectedSupports && err != nil {
				t.Errorf("Expected supports, but returned err for GVK (%s)", tc.gvk)
			} else if !tc.expectedSupports && err == nil {
				t.Errorf("Expected not supports, but returned no err for GVK (%s)", tc.gvk)
			}
		})
	}
}

func TestFallbackQueryParamVerifier_SecondaryFallback(t *testing.T) {
	tests := map[string]struct {
		crds             []schema.GroupKind      // CRDFinder returns these CRD's
		gvk              schema.GroupVersionKind // GVK whose OpenAPI spec is checked
		queryParam       VerifiableQueryParam    // Usually "fieldValidation"
		primaryError     error
		expectedSupports bool
	}{
		"Field validation query param is supported for batch/v1/Job, secondary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "batch",
				Version: "v1",
				Kind:    "Job",
			},
			queryParam:       QueryParamFieldValidation,
			primaryError:     errors.NewNotFound(schema.GroupResource{}, "OpenAPI V3 endpoint not found"),
			expectedSupports: true,
		},
		"Field validation query param is supported for batch/v1/Job, invalid v3 document error": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "batch",
				Version: "v1",
				Kind:    "Job",
			},
			queryParam:       QueryParamFieldValidation,
			primaryError:     fmt.Errorf("Invalid OpenAPI V3 document"),
			expectedSupports: true,
		},
		"Field validation query param is supported for batch/v1/Job, timeout error": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "batch",
				Version: "v1",
				Kind:    "Job",
			},
			queryParam:       QueryParamFieldValidation,
			primaryError:     fmt.Errorf("timeout"),
			expectedSupports: true,
		},
		"Field validation query param supported for core/v1/Namespace, secondary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "Namespace",
			},
			queryParam:       QueryParamFieldValidation,
			primaryError:     errors.NewNotFound(schema.GroupResource{}, "OpenAPI V3 endpoint not found"),
			expectedSupports: true,
		},
		"Field validation unsupported for unknown GVK, secondary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "bad",
				Version: "v1",
				Kind:    "Uknown",
			},
			queryParam:       QueryParamFieldValidation,
			primaryError:     errors.NewNotFound(schema.GroupResource{}, "OpenAPI V3 endpoint not found"),
			expectedSupports: false,
		},
		"Field validation unsupported for unknown GVK, invalid document causes secondary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "bad",
				Version: "v1",
				Kind:    "Uknown",
			},
			queryParam:       QueryParamFieldValidation,
			primaryError:     fmt.Errorf("Invalid OpenAPI V3 document"),
			expectedSupports: false,
		},
		"Unknown query param unsupported (for all GVK's), secondary verifier": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "apps",
				Version: "v1",
				Kind:    "Deployment",
			},
			queryParam:       "UnknownQueryParam",
			primaryError:     errors.NewNotFound(schema.GroupResource{}, "OpenAPI V3 endpoint not found"),
			expectedSupports: false,
		},
		"Field validation query param supported for found CRD, secondary verifier": {
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
			primaryError:     errors.NewNotFound(schema.GroupResource{}, "OpenAPI V3 endpoint not found"),
			expectedSupports: true,
		},
		"Field validation query param unsupported for missing CRD, secondary verifier": {
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
			primaryError:     errors.NewNotFound(schema.GroupResource{}, "OpenAPI V3 endpoint not found"),
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
			primaryError:     errors.NewNotFound(schema.GroupResource{}, "OpenAPI V3 endpoint not found"),
			expectedSupports: false,
		},
	}

	// Primary OpenAPI client always returns "NotFound" error, so secondary verifier is used.
	fakeOpenAPIClient := openapitest.NewFakeClient()
	root := openapi3.NewRoot(fakeOpenAPIClient)
	for tn, tc := range tests {
		t.Run(tn, func(t *testing.T) {
			fakeOpenAPIClient.ForcedErr = tc.primaryError
			primary := createFakeV3Verifier(tc.crds, root, tc.queryParam)
			secondary := createFakeLegacyVerifier(tc.crds, &fakeSchema, tc.queryParam)
			verifier := NewFallbackQueryParamVerifier(primary, secondary)
			err := verifier.HasSupport(tc.gvk)
			if tc.expectedSupports && err != nil {
				t.Errorf("Expected supports, but returned err for GVK (%s)", tc.gvk)
			} else if !tc.expectedSupports && err == nil {
				t.Errorf("Expected not supports, but returned no err for GVK (%s)", tc.gvk)
			}
		})
	}
}

// createFakeV3Verifier returns a fake OpenAPI V3 queryParamVerifierV3 struct
// filled in with passed values; implements Verifier interface.
func createFakeV3Verifier(crds []schema.GroupKind, root openapi3.Root, queryParam VerifiableQueryParam) Verifier {
	return &queryParamVerifierV3{
		finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
			return crds, nil
		}),
		root:       root,
		queryParam: queryParam,
	}
}

// createFakeLegacyVerifier returns a fake QueryParamVerifier struct for legacy
// OpenAPI V2; implements Verifier interface.
func createFakeLegacyVerifier(crds []schema.GroupKind, fakeSchema discovery.OpenAPISchemaInterface, queryParam VerifiableQueryParam) Verifier {
	return &QueryParamVerifier{
		finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
			return crds, nil
		}),
		openAPIGetter: fakeSchema,
		queryParam:    queryParam,
	}
}

// failingVerifier always crashes when called; implements Verifier
type failingVerifier struct {
	name string
	t    *testing.T
}

func (c *failingVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	c.t.Fatalf("%s verifier should not be called", c.name)
	return nil
}
