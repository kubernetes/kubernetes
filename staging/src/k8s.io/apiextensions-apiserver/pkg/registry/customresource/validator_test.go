/*
Copyright 2025 The Kubernetes Authors.

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

package customresource

import (
	"context"
	"strings"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsvalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// TestValidateCustomResourceName verifies that:
//   - the DNS1123 subdomain name validation applies to every custom resource regardless
//     of the schema an author declares for metadata.name (it cannot be relaxed),
//   - an author-declared constraint on metadata.name is enforced in addition, and
//   - the maxLength that WithTypeAndObjectMeta sets on metadata.name/generateName for CEL
//     cost estimation is never enforced as a value constraint (a 253-char generateName,
//     which exceeds the 252 cost bound, is still accepted).
func TestValidateCustomResourceName(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Foo"}

	// metadata.name uses a pattern that permits uppercase letters - more permissive than
	// DNS1123 in that dimension - so we can show the DNS1123 baseline still applies.
	props := &apiextensions.JSONSchemaProps{
		Type: "object",
		Properties: map[string]apiextensions.JSONSchemaProps{
			"metadata": {
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"name": {Type: "string", Pattern: "^[A-Za-z]+$"},
				},
			},
		},
	}
	schemaValidator, _, err := apiextensionsvalidation.NewSchemaValidator(props)
	if err != nil {
		t.Fatal(err)
	}
	// Cluster-scoped so the test can focus on name validation without a namespace.
	validator := customResourceValidator{namespaceScoped: false, kind: gvk, schemaValidator: schemaValidator}

	newCR := func(meta map[string]interface{}) *unstructured.Unstructured {
		return &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "Foo",
			"metadata":   meta,
		}}
	}

	cases := []struct {
		name        string
		meta        map[string]interface{}
		errContains string // "" means the resource is expected to validate
	}{
		{
			name: "valid lowercase name accepted",
			meta: map[string]interface{}{"name": "foo"},
		},
		{
			// The author's pattern permits "Foo", but the DNS1123 baseline rejects uppercase.
			name:        "uppercase name rejected by DNS1123 despite a permissive pattern",
			meta:        map[string]interface{}{"name": "Foo"},
			errContains: "RFC 1123 subdomain",
		},
		{
			// DNS1123 permits "foo123", but the author's pattern rejects digits.
			name:        "name rejected by the author's pattern",
			meta:        map[string]interface{}{"name": "foo123"},
			errContains: "should match",
		},
		{
			// generateName up to 253 is valid per DNS1123; the 252 CEL cost bound must not reject it.
			name: "253-char generateName accepted; CEL cost bound is not enforced",
			meta: map[string]interface{}{"name": "foo", "generateName": strings.Repeat("a", 253)},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validator.Validate(context.TODO(), newCR(tc.meta), nil)
			if tc.errContains == "" {
				if len(errs) != 0 {
					t.Fatalf("expected no validation error, got: %v", errs.ToAggregate())
				}
				return
			}
			if len(errs) == 0 {
				t.Fatalf("expected a validation error containing %q, got none", tc.errContains)
			}
			if got := errs.ToAggregate().Error(); !strings.Contains(got, tc.errContains) {
				t.Errorf("error %q does not contain %q", got, tc.errContains)
			}
		})
	}
}
