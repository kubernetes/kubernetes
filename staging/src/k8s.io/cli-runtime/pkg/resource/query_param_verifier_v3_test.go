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
	"encoding/json"
	"strings"
	"sync/atomic"
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
		"Unknown query param unsupported for found CRD": {
			crds: []schema.GroupKind{
				{
					Group: "example.com",
					Kind:  "ExampleCRD",
				},
			},
			gvk: schema.GroupVersionKind{
				Group:   "example.com",
				Version: "v1",
				Kind:    "ExampleCRD",
			},
			queryParam:       "UnknownQueryParam",
			expectedSupports: false,
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
		"Typed List GVK without PATCH operation returns ParamUnsupportedError": {
			crds: []schema.GroupKind{},
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "ReplicationControllerList",
			},
			queryParam:       QueryParamFieldValidation,
			expectedSupports: false,
		},
	}

	resources := openapi3.NewRootAndResources(cached.NewClient(openapitest.NewEmbeddedFileClient()))
	for tn, tc := range tests {
		t.Run(tn, func(t *testing.T) {
			verifier := &queryParamVerifierV3{
				finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
					return tc.crds, nil
				}),
				resources:  resources,
				queryParam: tc.queryParam,
			}
			err := verifier.HasSupport(tc.gvk)
			if tc.expectedSupports && err != nil {
				t.Errorf("Expected supports, but returned err for GVK (%s): %v", tc.gvk, err)
			} else if !tc.expectedSupports && !IsParamUnsupportedError(err) {
				t.Errorf("Expected ParamUnsupportedError, but returned (%v) for GVK (%s)", err, tc.gvk)
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
			err:  "invalid OpenAPI v3 document",
		},
		"empty document returns error": {
			spec: &spec3.OpenAPI{},
			err:  "invalid OpenAPI v3 document",
		},
		"minimal document returns error": {
			spec: &spec3.OpenAPI{
				Version: "openapi 3.0.0",
				Paths:   nil,
			},
			err: "invalid OpenAPI v3 document",
		},
		"empty Paths returns ParamUnsupportedError": {
			spec: &spec3.OpenAPI{
				Version: "openapi 3.0.0",
				Paths:   &spec3.Paths{},
			},
			err: "batch/v1, Kind=Job doesn't support fieldValidation",
		},
		"nil Path returns ParamUnsupportedError": {
			spec: &spec3.OpenAPI{
				Version: "openapi 3.0.0",
				Paths:   &spec3.Paths{Paths: map[string]*spec3.Path{"/version": nil}},
			},
			err: "batch/v1, Kind=Job doesn't support fieldValidation",
		},
		"empty Path returns ParamUnsupportedError": {
			spec: &spec3.OpenAPI{
				Version: "openapi 3.0.0",
				Paths:   &spec3.Paths{Paths: map[string]*spec3.Path{"/version": {}}},
			},
			err: "batch/v1, Kind=Job doesn't support fieldValidation",
		},
	}

	gvk := schema.GroupVersionKind{
		Group:   "batch",
		Version: "v1",
		Kind:    "Job",
	}
	for tn, tc := range tests {
		t.Run(tn, func(t *testing.T) {
			rawSpec, err := json.Marshal(tc.spec)
			if err != nil {
				t.Fatalf("unexpected marshal error: %v", err)
			}
			fakeClient := openapitest.NewFakeClient()
			fakeClient.PathsMap["apis/batch/v1"] = fakeV3GroupVersion{
				spec:        rawSpec,
				relativeURL: "/openapi/v3/apis/batch/v1?hash=test",
			}
			verifier := &queryParamVerifierV3{
				finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
					return []schema.GroupKind{}, nil
				}),
				resources:  openapi3.NewRootAndResources(fakeClient),
				queryParam: QueryParamFieldValidation,
			}
			err = verifier.HasSupport(gvk)
			if !strings.Contains(err.Error(), tc.err) {
				t.Errorf("Expected error (%s), but received (%s)", tc.err, err.Error())
			}
		})
	}
}

func TestQueryParamVerifierV3_CachesGroupVersionAcrossKinds(t *testing.T) {
	embeddedPaths, err := openapitest.NewEmbeddedFileClient().Paths()
	if err != nil {
		t.Fatalf("unexpected error loading embedded paths: %v", err)
	}
	coreV1Bytes, err := embeddedPaths["api/v1"].Schema("application/json")
	if err != nil {
		t.Fatalf("unexpected error loading embedded api/v1 schema: %v", err)
	}

	var schemaCalls atomic.Int32
	fakeClient := openapitest.NewFakeClient()
	fakeClient.PathsMap["api/v1"] = fakeV3GroupVersion{
		spec:        coreV1Bytes,
		relativeURL: "/openapi/v3/api/v1?hash=corev1",
		schemaCalls: &schemaCalls,
	}

	verifier := NewQueryParamVerifierV3(nil, fakeClient, QueryParamFieldValidation)
	for _, kind := range []string{"ServiceAccount", "Secret", "ConfigMap", "Service", "PersistentVolumeClaim", "Pod", "ReplicationController"} {
		gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: kind}
		if err := verifier.HasSupport(gvk); err != nil {
			t.Fatalf("expected %s to support fieldValidation, got: %v", gvk, err)
		}
	}
	if got := schemaCalls.Load(); got != 1 {
		t.Fatalf("expected api/v1 OpenAPI V3 schema to be fetched and unmarshaled once across 7 core/v1 Kinds, got %d", got)
	}

	// Known GroupVersion with missing Kind or unsupported PATCH operation returns ParamUnsupportedError.
	if err := verifier.HasSupport(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "DoesNotExist"}); !IsParamUnsupportedError(err) {
		t.Fatalf("expected ParamUnsupportedError for missing Kind in known GroupVersion, got: %v", err)
	}
	if err := verifier.HasSupport(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Binding"}); !IsParamUnsupportedError(err) {
		t.Fatalf("expected ParamUnsupportedError for non-patchable v1/Binding, got: %v", err)
	}

	// Verify CRD fallback propagates error if Namespace (api/v1) spec is absent.
	emptyVerifier := &queryParamVerifierV3{
		finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
			return []schema.GroupKind{{Group: "example.com", Kind: "ExampleCRD"}}, nil
		}),
		resources:  openapi3.NewRootAndResources(openapitest.NewFakeClient()),
		queryParam: QueryParamFieldValidation,
	}
	if err := emptyVerifier.HasSupport(schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "ExampleCRD"}); err == nil || IsParamUnsupportedError(err) {
		t.Fatalf("expected Namespace GroupVersionNotFoundError when api/v1 is missing during CRD fallback, got: %v", err)
	}
}

type fakeV3GroupVersion struct {
	spec        []byte
	relativeURL string
	schemaCalls *atomic.Int32
}

func (f fakeV3GroupVersion) Schema(contentType string) ([]byte, error) {
	if f.schemaCalls != nil {
		f.schemaCalls.Add(1)
	}
	return f.spec, nil
}

func (f fakeV3GroupVersion) ServerRelativeURL() string {
	return f.relativeURL
}
