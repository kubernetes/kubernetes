/*
Copyright The Kubernetes Authors.

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
	"reflect"
	"testing"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestSchemaPropertyAccessors(t *testing.T) {
	cases := []struct {
		name  string
		props map[string]spec.Schema
	}{
		{name: "nil properties"},
		{name: "empty properties", props: map[string]spec.Schema{}},
		{name: "declared properties", props: map[string]spec.Schema{
			"name": *spec.StringProperty(),
			"if":   *spec.Int64Property(),
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &Schema{Schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, Properties: tc.props}}}

			// Callers rely on HasProperties() matching the old Properties() != nil check.
			if got, want := s.HasProperties(), s.Properties() != nil; got != want {
				t.Errorf("HasProperties() = %v, want %v", got, want)
			}

			for key, want := range s.Properties() {
				got, ok := s.Property(key)
				if !ok {
					t.Errorf("Property(%q) not found, want %v", key, want)
					continue
				}
				if !reflect.DeepEqual(got.(*Schema).Schema, want.(*Schema).Schema) {
					t.Errorf("Property(%q) = %#v, want %#v", key, got.(*Schema).Schema, want.(*Schema).Schema)
				}
			}

			// Property names are unescaped, so the escaped CEL form must not resolve.
			for _, key := range []string{"missing", "__if__", ""} {
				if got, ok := s.Property(key); ok || got != nil {
					t.Errorf("Property(%q) = (%v, %v), want (nil, false)", key, got, ok)
				}
			}
		})
	}
}

func TestSchemaPropertyReturnsCopy(t *testing.T) {
	s := &Schema{Schema: &spec.Schema{SchemaProps: spec.SchemaProps{
		Type:       []string{"object"},
		Properties: map[string]spec.Schema{"name": *spec.StringProperty()},
	}}}

	got, ok := s.Property("name")
	if !ok {
		t.Fatal("Property(\"name\") not found")
	}
	got.(*Schema).Schema.Type = []string{"integer"}

	if typ := s.Schema.Properties["name"].Type; !reflect.DeepEqual(typ, spec.StringOrArray{"string"}) {
		t.Errorf("mutating Property() result changed the underlying schema type to %v", typ)
	}
}
