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

package schema

import (
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	"sigs.k8s.io/randfill"
)

func TestValidateStructuralMetadataInvariants(t *testing.T) {
	fuzzer := randfill.New()
	fuzzer.Funcs(
		func(s *JSON, c randfill.Continue) {
			if c.Bool() {
				s.Object = float64(42.0)
			}
		},
		func(s **StructuralOrBool, c randfill.Continue) {
			if c.Bool() {
				*s = &StructuralOrBool{}
			}
		},
		func(s **Structural, c randfill.Continue) {
			if c.Bool() {
				*s = &Structural{}
			}
		},
		func(s *Structural, c randfill.Continue) {
			if c.Bool() {
				*s = Structural{}
			}
		},
		func(vv **NestedValueValidation, c randfill.Continue) {
			if c.Bool() {
				*vv = &NestedValueValidation{}
			}
		},
		func(vv *NestedValueValidation, c randfill.Continue) {
			if c.Bool() {
				*vv = NestedValueValidation{}
			}
		},
	)
	fuzzer.NilChance(0)

	// check that type must be object
	typeNames := []string{"object", "array", "number", "integer", "boolean", "string"}
	for _, typeName := range typeNames {
		s := Structural{
			Generic: Generic{
				Type: typeName,
			},
		}

		errs := validateStructuralMetadataInvariants(&s, true, rootLevel, nil)
		if len(errs) != 0 {
			t.Logf("errors returned: %v", errs)
		}
		if len(errs) != 0 && typeName == "object" {
			t.Errorf("unexpected forbidden field validation errors for: %#v", s)
		}
		if len(errs) == 0 && typeName != "object" {
			t.Errorf("expected forbidden field validation errors for: %#v", s)
		}
	}

	// check that anything other than name and generateName of ObjectMeta in metadata properties is forbidden
	tt := reflect.TypeOf(metav1.ObjectMeta{})
	for i := 0; i < tt.NumField(); i++ {
		property := tt.Field(i).Name
		s := &Structural{
			Generic: Generic{
				Type: "object",
			},
			Properties: map[string]Structural{
				property: {},
			},
		}

		errs := validateStructuralMetadataInvariants(s, true, rootLevel, nil)
		if len(errs) != 0 {
			t.Logf("errors returned: %v", errs)
		}
		if len(errs) != 0 && (property == "name" || property == "generateName") {
			t.Errorf("unexpected forbidden field validation errors for: %#v", s)
		}
		if len(errs) == 0 && property != "name" && property != "generateName" {
			t.Errorf("expected forbidden field validation errors for: %#v", s)
		}
	}

	// check that anything other than type and properties in metadata is forbidden
	tt = reflect.TypeOf(Structural{})
	for i := 0; i < tt.NumField(); i++ {
		s := Structural{}
		x := reflect.ValueOf(&s).Elem()
		fuzzer.Fill(x.Field(i).Addr().Interface())
		s.Type = "object"
		s.Properties = map[string]Structural{
			"name":         {},
			"generateName": {},
		}
		s.Default.Object = nil // this is checked in API validation, we don't need to test it here

		valid := reflect.DeepEqual(s, Structural{
			Generic: Generic{
				Type: "object",
				Default: JSON{
					Object: nil,
				},
			},
			Properties: map[string]Structural{
				"name":         {},
				"generateName": {},
			},
		})

		errs := validateStructuralMetadataInvariants(s.DeepCopy(), true, rootLevel, nil)
		if len(errs) != 0 {
			t.Logf("errors returned: %v", errs)
		}
		if len(errs) != 0 && valid {
			t.Errorf("unexpected forbidden field validation errors for: %#v", s)
		}
		if len(errs) == 0 && !valid {
			t.Errorf("expected forbidden field validation errors for: %#v", s)
		}
	}
}

func TestValidateStructuralCompleteness(t *testing.T) {

	type tct struct {
		name    string
		schema  Structural
		options ValidationOptions
		error   string
	}

	testCases := []tct{
		{
			name: "allowed properties valuevalidation, additional properties structure",
			schema: Structural{
				AdditionalProperties: &StructuralOrBool{
					Structural: &Structural{
						Generic: Generic{
							Type: "object",
						},
						Properties: map[string]Structural{
							"bar": {
								Generic: Generic{
									Type: "string",
								},
							},
						},
					},
				},
				ValueValidation: &ValueValidation{
					AllOf: []NestedValueValidation{
						{
							Properties: map[string]NestedValueValidation{
								"foo": {
									ValueValidation: ValueValidation{
										MinLength: ptr.To[int64](2),
									},
								},
							},
						},
					},
				},
			},
			options: ValidationOptions{
				AllowValidationPropertiesWithAdditionalProperties: true,
			},
		},
		{
			name: "disallowed properties valuevalidation, additional properties structure",
			schema: Structural{
				AdditionalProperties: &StructuralOrBool{
					Structural: &Structural{
						Generic: Generic{
							Type: "object",
						},
						Properties: map[string]Structural{
							"bar": {
								Generic: Generic{
									Type: "string",
								},
							},
						},
					},
				},
				ValueValidation: &ValueValidation{
					AllOf: []NestedValueValidation{
						{
							Properties: map[string]NestedValueValidation{
								"foo": {
									ValueValidation: ValueValidation{
										MinLength: ptr.To[int64](2),
									},
								},
							},
						},
					},
				},
			},
			error: `properties[foo]: Required value: because it is defined in allOf[0].properties[foo]`,
			options: ValidationOptions{
				AllowValidationPropertiesWithAdditionalProperties: false,
			},
		},
		{
			name: "disallowed additionalproperties valuevalidation, properties structure",
			schema: Structural{
				Properties: map[string]Structural{
					"bar": {
						Generic: Generic{
							Type: "string",
						},
					},
				},
				ValueValidation: &ValueValidation{
					AllOf: []NestedValueValidation{
						{
							AdditionalProperties: &NestedValueValidation{
								ValueValidation: ValueValidation{
									MinLength: ptr.To[int64](2),
								},
							},
						},
					},
				},
			},
			error: `additionalProperties: Required value: because it is defined in allOf[0].additionalProperties`,
			options: ValidationOptions{
				AllowNestedAdditionalProperties: true,
			},
		},
		{
			name: "allowed property in valuevalidation, and in structure",
			schema: Structural{
				Properties: map[string]Structural{
					"foo": {
						Generic: Generic{
							Type: "string",
						},
					},
				},
				ValueValidation: &ValueValidation{
					AllOf: []NestedValueValidation{
						{
							Properties: map[string]NestedValueValidation{
								"foo": {
									ValueValidation: ValueValidation{
										MinLength: ptr.To[int64](2),
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "disallowed property in valuevalidation, and in structure",
			schema: Structural{
				Properties: map[string]Structural{
					"foo": {
						Generic: Generic{
							Type: "string",
						},
					},
				},
				ValueValidation: &ValueValidation{
					AllOf: []NestedValueValidation{
						{
							Properties: map[string]NestedValueValidation{
								"notfoo": {
									ValueValidation: ValueValidation{
										MinLength: ptr.To[int64](2),
									},
								},
							},
						},
					},
				},
			},
			error: `properties[notfoo]: Required value: because it is defined in allOf[0].properties[notfoo]`,
		},
		{
			name: "allowed items in valuevalidation, and in structure",
			schema: Structural{
				Generic: Generic{
					Type: "array",
				},
				Items: &Structural{
					Generic: Generic{
						Type: "string",
					},
				},
				ValueValidation: &ValueValidation{
					AllOf: []NestedValueValidation{
						{
							Items: &NestedValueValidation{
								ValueValidation: ValueValidation{
									MinLength: ptr.To[int64](2),
								},
							},
						},
					},
				},
			},
		},
		{
			name: "disallowed items in valuevalidation, and not in structure",
			schema: Structural{
				Generic: Generic{
					Type: "object",
				},
				Properties: map[string]Structural{
					"foo": {
						Generic: Generic{
							Type: "string",
						},
					},
				},
				ValueValidation: &ValueValidation{
					AllOf: []NestedValueValidation{
						{
							Items: &NestedValueValidation{
								ValueValidation: ValueValidation{
									MinLength: ptr.To[int64](2),
								},
							},
						},
					},
				},
			},
			error: `items: Required value: because it is defined in allOf[0].items`,
		},
	}

	for _, tc := range testCases {
		errs := validateStructuralCompleteness(&tc.schema, nil, tc.options)
		if len(tc.error) == 0 && len(errs) != 0 {
			t.Errorf("unexpected errors: %v", errs)
		}
		if len(tc.error) != 0 {
			contains := false

			for _, err := range errs {
				if strings.Contains(err.Error(), tc.error) {
					contains = true
					break
				}
			}

			if !contains {
				t.Errorf("expected error: %s, got %v", tc.error, errs)
			}
		}
	}

}

func TestValidateNestedValueValidationComplete(t *testing.T) {
	fuzzer := randfill.New()
	fuzzer.Funcs(
		func(s *JSON, c randfill.Continue) {
			if c.Bool() {
				s.Object = float64(42.0)
			}
		},
		func(s **NestedValueValidation, c randfill.Continue) {
			if c.Bool() {
				*s = &NestedValueValidation{}
			}
		},
	)
	fuzzer.NilChance(0)

	// check that we didn't forget to check any forbidden generic field
	tt := reflect.TypeOf(Generic{})
	for i := 0; i < tt.NumField(); i++ {
		vv := &NestedValueValidation{}
		x := reflect.ValueOf(&vv.ForbiddenGenerics).Elem()
		fuzzer.Fill(x.Field(i).Addr().Interface())

		errs := validateNestedValueValidation(vv, false, false, fieldLevel, nil, ValidationOptions{})
		if len(errs) == 0 && !reflect.DeepEqual(vv.ForbiddenGenerics, Generic{}) {
			t.Errorf("expected ForbiddenGenerics validation errors for: %#v", vv)
		}
	}

	// check that we didn't forget to check any forbidden extension field
	tt = reflect.TypeOf(Extensions{})
	for i := 0; i < tt.NumField(); i++ {
		vv := &NestedValueValidation{}
		x := reflect.ValueOf(&vv.ForbiddenExtensions).Elem()
		fuzzer.Fill(x.Field(i).Addr().Interface())

		errs := validateNestedValueValidation(vv, false, false, fieldLevel, nil, ValidationOptions{})
		if len(errs) == 0 && !reflect.DeepEqual(vv.ForbiddenExtensions, Extensions{}) {
			t.Errorf("expected ForbiddenExtensions validation errors for: %#v", vv)
		}
	}

	for _, allowedNestedXValidations := range []bool{false, true} {
		for _, allowedNestedAdditionalProperties := range []bool{false, true} {
			opts := ValidationOptions{
				AllowNestedXValidations:         allowedNestedXValidations,
				AllowNestedAdditionalProperties: allowedNestedAdditionalProperties,
			}

			vv := NestedValueValidation{}
			fuzzer.Fill(&vv.ValidationExtensions.XValidations)
			errs := validateNestedValueValidation(&vv, false, false, fieldLevel, nil, opts)
			if allowedNestedXValidations {
				if len(errs) != 0 {
					t.Errorf("unexpected XValidations validation errors for: %#v", vv)
				}
			} else if len(errs) == 0 && len(vv.ValidationExtensions.XValidations) != 0 {
				t.Errorf("expected XValidations validation errors for: %#v", vv)
			}

			vv = NestedValueValidation{}
			fuzzer.Fill(&vv.AdditionalProperties)
			errs = validateNestedValueValidation(&vv, false, false, fieldLevel, nil, opts)
			if allowedNestedAdditionalProperties {
				if len(errs) != 0 {
					t.Errorf("unexpected AdditionalProperties validation errors for: %#v", vv)
				}
			} else if len(errs) == 0 && vv.AdditionalProperties != nil {
				t.Errorf("expected AdditionalProperties validation errors for: %#v", vv)
			}
		}
	}
}
