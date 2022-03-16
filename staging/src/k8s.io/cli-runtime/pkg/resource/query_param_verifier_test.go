/*
Copyright 2019 The Kubernetes Authors.

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
	"path/filepath"
	"testing"

	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	"k8s.io/apimachinery/pkg/runtime/schema"
	openapitesting "k8s.io/kube-openapi/pkg/util/proto/testing"
)

var fakeSchema = openapitesting.Fake{Path: filepath.Join("..", "..", "artifacts", "openapi", "swagger.json")}

func TestSupportsQueryParam(t *testing.T) {
	doc, err := fakeSchema.OpenAPISchema()
	if err != nil {
		t.Fatalf("Failed to get OpenAPI Schema: %v", err)
	}

	tests := []struct {
		gvk        schema.GroupVersionKind
		success    bool
		supports   bool
		queryParam VerifiableQueryParam
	}{
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "Pod",
			},
			success:    true,
			supports:   true,
			queryParam: QueryParamDryRun,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "Pod",
			},
			success:    true,
			supports:   true,
			queryParam: QueryParamFieldValidation,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "UnknownKind",
			},
			success:    false,
			supports:   false,
			queryParam: QueryParamDryRun,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "UnknownKind",
			},
			success:    false,
			supports:   false,
			queryParam: QueryParamFieldValidation,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "NodeProxyOptions",
			},
			success:    true,
			supports:   false,
			queryParam: QueryParamDryRun,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "NodeProxyOptions",
			},
			success:    true,
			supports:   false,
			queryParam: QueryParamFieldValidation,
		},
	}

	for _, test := range tests {
		supports, err := supportsQueryParam(doc, test.gvk, test.queryParam)
		if supports != test.supports || ((err == nil) != test.success) {
			errStr := "nil"
			if test.success == false {
				errStr = "err"
			}
			t.Errorf("SupportsQueryParam(doc, %v, %v) = (%v, %v), expected (%v, %v)",
				test.gvk, test.queryParam,
				supports, err,
				test.supports, errStr,
			)
		}
	}
}

func TestDryRunVerifier(t *testing.T) {
	dryRunVerifier := QueryParamVerifier{
		finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
			return []schema.GroupKind{
				{
					Group: "crd.com",
					Kind:  "MyCRD",
				},
				{
					Group: "crd.com",
					Kind:  "MyNewCRD",
				},
			}, nil
		}),
		openAPIGetter: &fakeSchema,
		queryParam:    QueryParamDryRun,
	}

	err := dryRunVerifier.HasSupport(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "NodeProxyOptions"})
	if err == nil {
		t.Fatalf("NodeProxyOptions doesn't support dry-run, yet no error found")
	}

	err = dryRunVerifier.HasSupport(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"})
	if err != nil {
		t.Fatalf("Pod should support dry-run: %v", err)
	}

	err = dryRunVerifier.HasSupport(schema.GroupVersionKind{Group: "crd.com", Version: "v1", Kind: "MyCRD"})
	if err != nil {
		t.Fatalf("MyCRD should support dry-run: %v", err)
	}

	err = dryRunVerifier.HasSupport(schema.GroupVersionKind{Group: "crd.com", Version: "v1", Kind: "Random"})
	if err == nil {
		t.Fatalf("Random doesn't support dry-run, yet no error found")
	}
}

func TestFieldValidationVerifier(t *testing.T) {
	fieldValidationVerifier := QueryParamVerifier{
		finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
			return []schema.GroupKind{
				{
					Group: "crd.com",
					Kind:  "MyCRD",
				},
				{
					Group: "crd.com",
					Kind:  "MyNewCRD",
				},
			}, nil
		}),
		openAPIGetter: &fakeSchema,
		queryParam:    QueryParamFieldValidation,
	}

	err := fieldValidationVerifier.HasSupport(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "NodeProxyOptions"})
	if err == nil {
		t.Fatalf("NodeProxyOptions doesn't support fieldValidation, yet no error found")
	}

	err = fieldValidationVerifier.HasSupport(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"})
	if err != nil {
		t.Fatalf("Pod should support fieldValidation: %v", err)
	}

	err = fieldValidationVerifier.HasSupport(schema.GroupVersionKind{Group: "crd.com", Version: "v1", Kind: "MyCRD"})
	if err != nil {
		t.Fatalf("MyCRD should support fieldValidation: %v", err)
	}

	err = fieldValidationVerifier.HasSupport(schema.GroupVersionKind{Group: "crd.com", Version: "v1", Kind: "Random"})
	if err == nil {
		t.Fatalf("Random doesn't support fieldValidation, yet no error found")
	}
}

type EmptyOpenAPI struct{}

func (EmptyOpenAPI) OpenAPISchema() (*openapi_v2.Document, error) {
	return &openapi_v2.Document{}, nil
}

func TestDryRunVerifierNoOpenAPI(t *testing.T) {
	dryRunVerifier := QueryParamVerifier{
		finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
			return []schema.GroupKind{
				{
					Group: "crd.com",
					Kind:  "MyCRD",
				},
				{
					Group: "crd.com",
					Kind:  "MyNewCRD",
				},
			}, nil
		}),
		openAPIGetter: EmptyOpenAPI{},
		queryParam:    QueryParamDryRun,
	}

	err := dryRunVerifier.HasSupport(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"})
	if err == nil {
		t.Fatalf("Pod doesn't support dry-run, yet no error found")
	}

	err = dryRunVerifier.HasSupport(schema.GroupVersionKind{Group: "crd.com", Version: "v1", Kind: "MyCRD"})
	if err == nil {
		t.Fatalf("MyCRD doesn't support dry-run, yet no error found")
	}
}

func TestFieldValidationVerifierNoOpenAPI(t *testing.T) {
	fieldValidationVerifier := QueryParamVerifier{
		finder: NewCRDFinder(func() ([]schema.GroupKind, error) {
			return []schema.GroupKind{
				{
					Group: "crd.com",
					Kind:  "MyCRD",
				},
				{
					Group: "crd.com",
					Kind:  "MyNewCRD",
				},
			}, nil
		}),
		openAPIGetter: EmptyOpenAPI{},
		queryParam:    QueryParamFieldValidation,
	}

	err := fieldValidationVerifier.HasSupport(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"})
	if err == nil {
		t.Fatalf("Pod doesn't support fieldValidation, yet no error found")
	}

	err = fieldValidationVerifier.HasSupport(schema.GroupVersionKind{Group: "crd.com", Version: "v1", Kind: "MyCRD"})
	if err == nil {
		t.Fatalf("MyCRD doesn't support fieldValidation, yet no error found")
	}
}
