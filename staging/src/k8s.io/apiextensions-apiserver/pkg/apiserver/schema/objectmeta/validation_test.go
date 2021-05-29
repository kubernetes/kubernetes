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

package objectmeta

import (
	"testing"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateEmbeddedResource(t *testing.T) {
	tests := []struct {
		name   string
		object map[string]interface{}
		errors []validationMatch
	}{
		{name: "empty", object: map[string]interface{}{}, errors: []validationMatch{
			required("apiVersion"),
			required("kind"),
		}},
		{name: "version and kind", object: map[string]interface{}{
			"apiVersion": "foo/v1",
			"kind":       "Foo",
		}},
		{name: "invalid kind", object: map[string]interface{}{
			"apiVersion": "foo/v1",
			"kind":       "foo.bar-com",
		}, errors: []validationMatch{
			invalid("kind"),
		}},
		{name: "no name", object: map[string]interface{}{
			"apiVersion": "foo/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"namespace": "kube-system",
			},
		}},
		{name: "no namespace", object: map[string]interface{}{
			"apiVersion": "foo/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"name": "foo",
			},
		}},
		{name: "invalid", object: map[string]interface{}{
			"apiVersion": "foo/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"name":      "..",
				"namespace": "$$$",
				"labels": map[string]interface{}{
					"#": "#",
				},
				"annotations": map[string]interface{}{
					"#": "#",
				},
			},
		}, errors: []validationMatch{
			invalid("metadata", "name"),
			invalid("metadata", "namespace"),
			invalid("metadata", "labels"),      // key
			invalid("metadata", "labels"),      // value
			invalid("metadata", "annotations"), // key
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema := &structuralschema.Structural{Extensions: structuralschema.Extensions{XEmbeddedResource: true}}
			errs := validateEmbeddedResource(nil, tt.object, schema)
			seenErrs := make([]bool, len(errs))

			for _, expectedError := range tt.errors {
				found := false
				for i, err := range errs {
					if expectedError.matches(err) && !seenErrs[i] {
						found = true
						seenErrs[i] = true
						break
					}
				}

				if !found {
					t.Errorf("expected %v at %v, got %v", expectedError.errorType, expectedError.path.String(), errs)
				}
			}

			for i, seen := range seenErrs {
				if !seen {
					t.Errorf("unexpected error: %v", errs[i])
				}
			}
		})
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name        string
		object      string
		includeRoot bool
		errors      []validationMatch
	}{
		{name: "empty", object: `{}`, errors: []validationMatch{}},
		{name: "include root", object: `{}`, includeRoot: true, errors: []validationMatch{
			required("apiVersion"),
			required("kind"),
		}},
		{name: "embedded", object: `
{
  "embedded": {}
}`, errors: []validationMatch{
			required("embedded", "apiVersion"),
			required("embedded", "kind"),
		}},
		{name: "nested", object: `
{
  "nested": {
    "embedded": {}
  }
}`, errors: []validationMatch{
			required("nested", "apiVersion"),
			required("nested", "kind"),
			required("nested", "embedded", "apiVersion"),
			required("nested", "embedded", "kind"),
		}},
		{name: "items", object: `
{
  "items": [{}]
}`, errors: []validationMatch{
			required("items[0]", "apiVersion"),
			required("items[0]", "kind"),
		}},
		{name: "additionalProperties", object: `
{
  "additionalProperties": {"foo":{}}
}`, errors: []validationMatch{
			required("additionalProperties[foo]", "apiVersion"),
			required("additionalProperties[foo]", "kind"),
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema := &structuralschema.Structural{
				Properties: map[string]structuralschema.Structural{
					"embedded": {Extensions: structuralschema.Extensions{XEmbeddedResource: true}},
					"nested": {
						Extensions: structuralschema.Extensions{XEmbeddedResource: true},
						Properties: map[string]structuralschema.Structural{
							"embedded": {Extensions: structuralschema.Extensions{XEmbeddedResource: true}},
						},
					},
					"items": {
						Items: &structuralschema.Structural{
							Extensions: structuralschema.Extensions{XEmbeddedResource: true},
						},
					},
					"additionalProperties": {
						Generic: structuralschema.Generic{
							AdditionalProperties: &structuralschema.StructuralOrBool{
								Structural: &structuralschema.Structural{
									Extensions: structuralschema.Extensions{XEmbeddedResource: true},
								},
							},
						},
					},
				},
			}

			var obj map[string]interface{}
			if err := json.Unmarshal([]byte(tt.object), &obj); err != nil {
				t.Fatal(err)
			}

			errs := Validate(nil, obj, schema, tt.includeRoot)
			seenErrs := make([]bool, len(errs))

			for _, expectedError := range tt.errors {
				found := false
				for i, err := range errs {
					if expectedError.matches(err) && !seenErrs[i] {
						found = true
						seenErrs[i] = true
						break
					}
				}

				if !found {
					t.Errorf("expected %v at %v, got %v", expectedError.errorType, expectedError.path.String(), errs)
				}
			}

			for i, seen := range seenErrs {
				if !seen {
					t.Errorf("unexpected error: %v", errs[i])
				}
			}
		})
	}
}

type validationMatch struct {
	path      *field.Path
	errorType field.ErrorType
}

func required(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeRequired}
}
func invalid(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeInvalid}
}

func (v validationMatch) matches(err *field.Error) bool {
	return err.Type == v.errorType && err.Field == v.path.String()
}
