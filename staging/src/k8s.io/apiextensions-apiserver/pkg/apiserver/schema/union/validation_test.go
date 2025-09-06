/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the LicenseC2.
*/

package union

import (
	"fmt"
	"testing"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidate(t *testing.T) {
	fmt.Println("Enter TestValidate")
	tests := []struct {
		name     string
		schema   *schema.Structural
		obj      interface{}
		expected field.ErrorList
	}{
		{
			name: "valid one-of union with one field set",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Members: []v1.UnionMember{
								{FieldName: "a"},
								{FieldName: "b"},
							},
						},
					},
				},
			},
			obj:      map[string]interface{}{"a": 1},
			expected: nil,
		},
		{
			name: "invalid one-of union with no fields set",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Members: []v1.UnionMember{
								{FieldName: "a"},
								{FieldName: "b"},
							},
						},
					},
				},
			},
			obj: map[string]interface{}{},
			expected: field.ErrorList{
				field.Invalid(field.NewPath(""), map[string]interface{}{}, "exactly one of [a b] is required"),
			},
		},
		{
			name: "invalid one-of union with multiple fields set",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Members: []v1.UnionMember{
								{FieldName: "a"},
								{FieldName: "b"},
							},
						},
					},
				},
			},
			obj: map[string]interface{}{"a": 1, "b": 2},
			expected: field.ErrorList{
				field.Invalid(field.NewPath(""), map[string]interface{}{"a": 1, "b": 2}, "exactly one of [a b] is required"),
			},
		},
		{
			name: "valid zero-or-one-of union with no fields set",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							ZeroOrOneOf: true,
							Members: []v1.UnionMember{
								{FieldName: "a"},
								{FieldName: "b"},
							},
						},
					},
				},
			},
			obj:      map[string]interface{}{},
			expected: nil,
		},
		{
			name: "valid zero-or-one-of union with one field set",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							ZeroOrOneOf: true,
							Members: []v1.UnionMember{
								{FieldName: "a"},
								{FieldName: "b"},
							},
						},
					},
				},
			},
			obj:      map[string]interface{}{"a": 1},
			expected: nil,
		},
		{
			name: "invalid zero-or-one-of union with multiple fields set",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							ZeroOrOneOf: true,
							Members: []v1.UnionMember{
								{FieldName: "a"},
								{FieldName: "b"},
							},
						},
					},
				},
			},
			obj: map[string]interface{}{"a": 1, "b": 2},
			expected: field.ErrorList{
				field.Invalid(field.NewPath(""), map[string]interface{}{"a": 1, "b": 2}, "at most one of [a b] is allowed"),
			},
		},
		{
			name: "valid union with discriminator",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Discriminator: "type",
							Members: []v1.UnionMember{
								{FieldName: "a", DiscriminatorValue: "a"},
								{FieldName: "b", DiscriminatorValue: "b"},
							},
						},
					},
				},
			},
			obj:      map[string]interface{}{"type": "a", "a": 1},
			expected: nil,
		},
		{
			name: "invalid union with discriminator but field not set",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Discriminator: "type",
							Members: []v1.UnionMember{
								{FieldName: "a", DiscriminatorValue: "a"},
								{FieldName: "b", DiscriminatorValue: "b"},
							},
						},
					},
				},
			},
			obj:      map[string]interface{}{"type": "a"},
			expected: field.ErrorList{field.Invalid(field.NewPath(""), map[string]interface{}{"type": "a"}, "discriminator set to a, but field a is not set")},
		},
		{
			name: "invalid union with discriminator and wrong field set",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Discriminator: "type",
							Members: []v1.UnionMember{
								{FieldName: "a", DiscriminatorValue: "a"},
								{FieldName: "b", DiscriminatorValue: "b"},
							},
						},
					},
				},
			},
			obj: map[string]interface{}{"type": "a", "b": 1},
			expected: field.ErrorList{
				field.Invalid(field.NewPath(""), map[string]interface{}{"type": "a", "b": 1}, "discriminator set to a, but field a is not set"),
				field.Invalid(field.NewPath(""), map[string]interface{}{"type": "a", "b": 1}, "field b is set but discriminator is 'a', not 'b'"),
			},
		},
		{
			name: "invalid union with discriminator not a string",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Discriminator: "type",
							Members: []v1.UnionMember{
								{FieldName: "a", DiscriminatorValue: "a"},
								{FieldName: "b", DiscriminatorValue: "b"},
							},
						},
					},
				},
			},
			obj: map[string]interface{}{"type": 1, "a": 1},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("").Child("type"), 1, "discriminator must be a string"),
			},
		},
		{
			name: "invalid union with discriminator value not in union fields",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Discriminator: "type",
							Members: []v1.UnionMember{
								{FieldName: "a", DiscriminatorValue: "a"},
								{FieldName: "b", DiscriminatorValue: "b"},
							},
						},
					},
				},
			},
			obj:      map[string]interface{}{"type": "c", "a": 1},
			expected: field.ErrorList{field.Invalid(field.NewPath("").Child("type"), "c", "discriminator value must be one of [a b]")},
		},
		{
			name: "valid list of objects",
			schema: &schema.Structural{
				Items: &schema.Structural{
					Extensions: schema.Extensions{
						XUnions: []v1.Union{
							{
								Members: []v1.UnionMember{
									{FieldName: "a"},
									{FieldName: "b"},
								},
							},
						},
					},
				},
			},
			obj: []interface{}{
				map[string]interface{}{"a": 1},
				map[string]interface{}{"b": 2},
			},
			expected: nil,
		},
		{
			name: "invalid list of objects",
			schema: &schema.Structural{
				Items: &schema.Structural{
					Extensions: schema.Extensions{
						XUnions: []v1.Union{
							{
								Members: []v1.UnionMember{
									{FieldName: "a"},
									{FieldName: "b"},
								},
							},
						},
					},
				},
			},
			obj: []interface{}{
				map[string]interface{}{"a": 1, "b": 2},
				map[string]interface{}{"a": 1},
			},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("").Index(0), map[string]interface{}{"a": 1, "b": 2}, "more than one field is set out of [a b]"),
			},
		},
		{
			name: "nested object with union",
			schema: &schema.Structural{
				Properties: map[string]schema.Structural{
					"nested": {
						Extensions: schema.Extensions{
							XUnions: []v1.Union{
								{
									Members: []v1.UnionMember{
										{FieldName: "a"},
										{FieldName: "b"},
									},
								},
							},
						},
					},
				},
			},
			obj: map[string]interface{}{
				"nested": map[string]interface{}{"a": 1, "b": 2},
			},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("").Child("nested"), map[string]interface{}{"a": 1, "b": 2}, "more than one field is set out of [a b]"),
			},
		},
		{
			name: "additional properties with union",
			schema: &schema.Structural{
				AdditionalProperties: &schema.StructuralOrBool{
					Structural: &schema.Structural{
						Extensions: schema.Extensions{
							XUnions: []v1.Union{
								{
									Members: []v1.UnionMember{
										{FieldName: "a"},
										{FieldName: "b"},
									},
								},
							},
						},
					},
				},
			},
			obj: map[string]interface{}{
				"prop1": map[string]interface{}{"a": 1, "b": 2},
			},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("").Child("prop1"), map[string]interface{}{"a": 1, "b": 2}, "more than one field is set out of [a b]"),
			},
		},
		{
			name: "object is not a map",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Members: []v1.UnionMember{
								{FieldName: "a"},
								{FieldName: "b"},
							},
						},
					},
				},
			},
			obj:      "not a map",
			expected: nil,
		},
		{
			name: "multiple unions",
			schema: &schema.Structural{
				Extensions: schema.Extensions{
					XUnions: []v1.Union{
						{
							Members: []v1.UnionMember{
								{FieldName: "a"},
								{FieldName: "b"},
							},
						},
						{
							Members: []v1.UnionMember{
								{FieldName: "c"},
								{FieldName: "d"},
							},
						},
					},
				},
			},
			obj:      map[string]interface{}{"a": 1, "c": 2},
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := Validate(field.NewPath(""), tt.schema, tt.obj)
			if len(errs) != len(tt.expected) {
				t.Fatalf("expected %d errors, got %d: %v", len(tt.expected), len(errs), errs)
			}
			for i := range errs {
				if errs[i].Error() != tt.expected[i].Error() {
					t.Errorf("expected error %q, got %q", tt.expected[i].Error(), errs[i].Error())
				}
			}
		})
	}
}
