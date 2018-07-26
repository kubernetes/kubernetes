/*
Copyright 2018 The Kubernetes Authors.

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

package validation

import (
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/utils/pointer"
)

func TestDeriveSkeleton(t *testing.T) {
	type args struct {
		v *apiextensions.JSONSchemaProps
	}
	tests := []struct {
		name    string
		args    args
		want    *apiextensions.JSONSchemaProps
		wantErr bool
	}{
		{"nil", args{v: nil}, nil, false},
		{"empty", args{v: &apiextensions.JSONSchemaProps{}}, &apiextensions.JSONSchemaProps{}, false},
		{"all dropped", args{v: &fullSchemaWithDroppedFields}, &apiextensions.JSONSchemaProps{}, false},
		{"non-array items",
			args{v: withItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrArray{Schema: &fullSchemaWithDroppedFields})},
			&apiextensions.JSONSchemaProps{Items: &apiextensions.JSONSchemaPropsOrArray{Schema: &apiextensions.JSONSchemaProps{}}},
			false,
		},
		{"array items",
			args{v: withItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrArray{JSONSchemas: []apiextensions.JSONSchemaProps{
				fullSchemaWithDroppedFields, fullSchemaWithDroppedFields,
			}})},
			&apiextensions.JSONSchemaProps{Items: &apiextensions.JSONSchemaPropsOrArray{JSONSchemas: []apiextensions.JSONSchemaProps{{}, {}}}},
			false,
		},
		{
			"additionalItems true",
			args{v: withAdditionalItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: true})},
			&apiextensions.JSONSchemaProps{AdditionalItems: &apiextensions.JSONSchemaPropsOrBool{Schema: &apiextensions.JSONSchemaProps{}}},
			false,
		},
		{
			"additionalItems false",
			args{v: withAdditionalItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: false})},
			&apiextensions.JSONSchemaProps{AdditionalItems: &apiextensions.JSONSchemaPropsOrBool{Schema: &apiextensions.JSONSchemaProps{}}}, // intentionally weakened
			false,
		},
		{
			"additionalItems array",
			args{v: withAdditionalItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Schema: &fullSchemaWithDroppedFields})},
			&apiextensions.JSONSchemaProps{AdditionalItems: &apiextensions.JSONSchemaPropsOrBool{Schema: &apiextensions.JSONSchemaProps{}}},
			false,
		},
		{"default",
			args{v: withDefault(fullSchemaWithDroppedFields, 42)},
			&apiextensions.JSONSchemaProps{Default: jsonPtr(42)},
			false,
		},
		{"type",
			args{v: withType(fullSchemaWithDroppedFields, "foo")},
			&apiextensions.JSONSchemaProps{Type: "foo"},
			false,
		},
		{"properties",
			args{v: withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
				"foo": fullSchemaWithDroppedFields,
				"bar": fullSchemaWithDroppedFields,
			})},
			&apiextensions.JSONSchemaProps{Properties: map[string]apiextensions.JSONSchemaProps{
				"foo": {},
				"bar": {},
			}},
			false,
		},
		{"patternProperties",
			args{v: withPatternProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
				"f*oo": fullSchemaWithDroppedFields,
				"ba*r": fullSchemaWithDroppedFields,
			})},
			&apiextensions.JSONSchemaProps{PatternProperties: map[string]apiextensions.JSONSchemaProps{
				"f*oo": {},
				"ba*r": {},
			}},
			false,
		},
		{
			"additionalProperties true",
			args{v: withAdditionalProperties(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: true})},
			&apiextensions.JSONSchemaProps{},
			false,
		},
		{
			"additionalProperties false",
			args{v: withAdditionalProperties(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: false})},
			&apiextensions.JSONSchemaProps{AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{Schema: &apiextensions.JSONSchemaProps{}}}, // intentionally weakened
			false,
		},
		{
			"additionalProperties array",
			args{v: withAdditionalProperties(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Schema: &fullSchemaWithDroppedFields})},
			&apiextensions.JSONSchemaProps{AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{Schema: &apiextensions.JSONSchemaProps{}}},
			false,
		},
		{
			"nesting",
			args{v: withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
				"a": *withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
					"b": *withAdditionalProperties(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{
						Schema: withPatternProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
							"c": *withItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrArray{
								Schema: withAdditionalItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{
									Schema: &fullSchemaWithDroppedFields,
								}),
							}),
						}),
					}),
				}),
			})},
			&apiextensions.JSONSchemaProps{Properties: map[string]apiextensions.JSONSchemaProps{
				"a": {Properties: map[string]apiextensions.JSONSchemaProps{
					"b": {
						AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
							Schema: &apiextensions.JSONSchemaProps{
								PatternProperties: map[string]apiextensions.JSONSchemaProps{
									"c": {
										Items: &apiextensions.JSONSchemaPropsOrArray{
											Schema: &apiextensions.JSONSchemaProps{
												AdditionalItems: &apiextensions.JSONSchemaPropsOrBool{
													Schema: &apiextensions.JSONSchemaProps{},
												},
											},
										},
									},
								},
							},
						},
					},
				}},
			}},
			false,
		},
		{
			"anyOf with nested properties", // allOf, oneOf are exactly the same, same logic. not is as anyOf with one branch.
			args{v: withAnyOf(
				*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
					"a": *withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"k": {}}),
					"b": *withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"l": {}, "m": {}}),
				}), []apiextensions.JSONSchemaProps{
					*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
						"b": *withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"m": {}}),
						"c": *withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"x": {}, "y": {}}),
					}),
					*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
						"c": *withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"y": {}, "z": {}}),
						"d": *withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"n": {}}),
					}),
				})},
			&apiextensions.JSONSchemaProps{Properties: map[string]apiextensions.JSONSchemaProps{
				"a": {Properties: map[string]apiextensions.JSONSchemaProps{"k": {}}},
				"b": {Properties: map[string]apiextensions.JSONSchemaProps{"l": {}, "m": {}}},
				"c": {Properties: map[string]apiextensions.JSONSchemaProps{"x": {}, "y": {}, "z": {}}},
				"d": {Properties: map[string]apiextensions.JSONSchemaProps{"n": {}}},
			}},
			false,
		},
		{
			"anyOf with patternProperties", // allOf, oneOf are exactly the same, same logic. not is as anyOf with one branch.
			args{v: withAnyOf(
				*withPatternProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
					"a": {},
					"b": {},
				}), []apiextensions.JSONSchemaProps{
					*withPatternProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
						"b": {},
						"c": {},
					}),
					*withPatternProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
						"c": {},
						"d": {},
					}),
				})},
			&apiextensions.JSONSchemaProps{PatternProperties: map[string]apiextensions.JSONSchemaProps{
				"a": {},
				"b": {},
				"c": {},
				"d": {},
			}},
			false,
		},
		{
			"anyOf with additionalProperties", // allOf, oneOf are exactly the same, same logic. not is as anyOf with one branch.
			args{v: withAnyOf(
				*withAdditionalProperties(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: false}),
				[]apiextensions.JSONSchemaProps{
					*withAdditionalProperties(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: true}),
					*withAdditionalProperties(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{
						Schema: withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"a": {}}),
					}),
				},
			)},
			&apiextensions.JSONSchemaProps{AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
				Schema: &apiextensions.JSONSchemaProps{ // intentionally weakened by dropping the false case
					Properties: map[string]apiextensions.JSONSchemaProps{
						"a": {},
					},
				},
			}},
			false,
		},
		{
			"anyOf with items", // allOf, oneOf are exactly the same, same logic. not is as anyOf with one branch.
			args{v: withAnyOf(
				*withItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrArray{
					JSONSchemas: []apiextensions.JSONSchemaProps{
						*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"a": {}}),
						*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"b": {}}),
					},
				}), []apiextensions.JSONSchemaProps{
					*withItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrArray{
						JSONSchemas: []apiextensions.JSONSchemaProps{
							*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"a": {}, "m": {}}),
							*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"b": {}, "o": {}}),
							*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"c": {}}),
						},
					}),
					*withItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrArray{
						JSONSchemas: []apiextensions.JSONSchemaProps{
							*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"a": {}, "m": {}, "n": {}}),
							*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"o": {}}),
						},
					}),
					*withItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrArray{
						Schema: withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"x": {}, "y": {}}),
					}),
				})},
			&apiextensions.JSONSchemaProps{Items: &apiextensions.JSONSchemaPropsOrArray{
				JSONSchemas: []apiextensions.JSONSchemaProps{
					{Properties: map[string]apiextensions.JSONSchemaProps{"a": {}, "m": {}, "n": {}, "x": {}, "y": {}}},
					{Properties: map[string]apiextensions.JSONSchemaProps{"b": {}, "o": {}, "x": {}, "y": {}}},
					{Properties: map[string]apiextensions.JSONSchemaProps{"c": {}, "x": {}, "y": {}}},
				},
			}},
			false,
		},
		{
			"anyOf with additionalItems", // allOf, oneOf are exactly the same, same logic. not is as anyOf with one branch.
			args{v: withAnyOf(
				*withAdditionalItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: false}),
				[]apiextensions.JSONSchemaProps{
					*withAdditionalItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: true}),
					*withAdditionalItems(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{
						Schema: withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"a": {}}),
					}),
				},
			)},
			&apiextensions.JSONSchemaProps{AdditionalItems: &apiextensions.JSONSchemaPropsOrBool{
				Schema: &apiextensions.JSONSchemaProps{ // intentionally weakened by dropping the false case
					Properties: map[string]apiextensions.JSONSchemaProps{
						"a": {},
					},
				},
			}},
			false,
		},
		{
			"anyOf object polymorphism with additionaProperties",
			args{v: withAnyOf(fullSchemaWithDroppedFields, []apiextensions.JSONSchemaProps{
				*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"x": {}}),
				*withAdditionalProperties(fullSchemaWithDroppedFields, apiextensions.JSONSchemaPropsOrBool{Allows: false}),
			})},
			nil,
			true,
		},
		{
			"anyOf object polymorphism with patternProperties",
			args{v: withAnyOf(fullSchemaWithDroppedFields, []apiextensions.JSONSchemaProps{
				*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"x": {}}),
				*withPatternProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{"foo*": {}}),
			})},
			&apiextensions.JSONSchemaProps{
				Properties:        map[string]apiextensions.JSONSchemaProps{"x": {}},
				PatternProperties: map[string]apiextensions.JSONSchemaProps{"foo*": {}},
			},
			false, // TODO: this should be true if we forbid properties in parallel to patternProperties
		},
		{
			"anyOf with types",
			args{v: withAnyOf(
				*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
					"a": *withType(fullSchemaWithDroppedFields, "string"), // matching types
					"b": *withType(fullSchemaWithDroppedFields, "string"), // differing types
					"c": *withType(fullSchemaWithDroppedFields, "string"), // matching types, but undefined on one branch
					"d": fullSchemaWithDroppedFields,                      // undefined on outer, matching types on branches
					"e": fullSchemaWithDroppedFields,                      // undefined on outer, undefined on one inner
					"f": *withType(fullSchemaWithDroppedFields, "string"), // only defined outside
					// "g" - undefined outside, matching inside
				}),
				[]apiextensions.JSONSchemaProps{
					*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
						"a": *withType(fullSchemaWithDroppedFields, "string"),
						"b": *withType(fullSchemaWithDroppedFields, "integer"),
						"c": fullSchemaWithDroppedFields,
						"d": *withType(fullSchemaWithDroppedFields, "string"),
						"e": fullSchemaWithDroppedFields,
						"g": *withType(fullSchemaWithDroppedFields, "string"),
					}),
					*withProperties(fullSchemaWithDroppedFields, map[string]apiextensions.JSONSchemaProps{
						"a": *withType(fullSchemaWithDroppedFields, "string"),
						"b": *withType(fullSchemaWithDroppedFields, "bool"),
						"c": *withType(fullSchemaWithDroppedFields, "string"),
						"d": *withType(fullSchemaWithDroppedFields, "string"),
						"e": *withType(fullSchemaWithDroppedFields, "string"),
						"g": *withType(fullSchemaWithDroppedFields, "string"),
					}),
				},
			)},
			&apiextensions.JSONSchemaProps{
				Properties: map[string]apiextensions.JSONSchemaProps{
					"a": {Type: "string"},
					"b": {},
					"c": {}, // this could be stricter, i.e. {Type: "string"} because the type outside of anyOf trumps any undefined branch
					"d": {}, // this could be stricter, i.e. {Type: "string"}, because the matching type on all branches trumps undefined type outside
					"e": {},
					"f": {Type: "string"},
					"g": {Type: "string"},
				},
			},
			false,
		},
		{
			"x-kubernetes-no-prune all true",
			args{v: withAnyOf(fullSchemaWithDroppedFields, []apiextensions.JSONSchemaProps{
				*withXKubernetesNoPrune(fullSchemaWithDroppedFields, true),
				*withXKubernetesNoPrune(fullSchemaWithDroppedFields, true),
				fullSchemaWithDroppedFields,
			})},
			&apiextensions.JSONSchemaProps{XKubernetesNoPrune: pointer.BoolPtr(true)},
			false,
		},
		{
			"x-kubernetes-no-prune all false",
			args{v: withAnyOf(fullSchemaWithDroppedFields, []apiextensions.JSONSchemaProps{
				*withXKubernetesNoPrune(fullSchemaWithDroppedFields, false),
				*withXKubernetesNoPrune(fullSchemaWithDroppedFields, false),
				fullSchemaWithDroppedFields,
			})},
			&apiextensions.JSONSchemaProps{XKubernetesNoPrune: pointer.BoolPtr(false)},
			false,
		},
		{
			"x-kubernetes-no-prune mixed",
			args{v: withAnyOf(fullSchemaWithDroppedFields, []apiextensions.JSONSchemaProps{
				*withXKubernetesNoPrune(fullSchemaWithDroppedFields, false),
				*withXKubernetesNoPrune(fullSchemaWithDroppedFields, true),
				fullSchemaWithDroppedFields,
			})},
			&apiextensions.JSONSchemaProps{XKubernetesNoPrune: pointer.BoolPtr(true)},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DeriveSkeleton(tt.args.v)
			if (err != nil) != tt.wantErr {
				t.Fatalf("DeriveSkeleton() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("DeriveSkeleton()\ndiff: %s\ngot:\n%s\nwant:\n%s\n", diff.ObjectDiff(got, tt.want), spew.Sdump(got), spew.Sdump(tt.want))
			}
		})
	}
}

func withXKubernetesNoPrune(props apiextensions.JSONSchemaProps, b bool) *apiextensions.JSONSchemaProps {
	props.XKubernetesNoPrune = &b
	return &props
}

func withItems(props apiextensions.JSONSchemaProps, items apiextensions.JSONSchemaPropsOrArray) *apiextensions.JSONSchemaProps {
	props.Items = &items
	return &props
}

func withAdditionalItems(props apiextensions.JSONSchemaProps, p apiextensions.JSONSchemaPropsOrBool) *apiextensions.JSONSchemaProps {
	props.AdditionalItems = &p
	return &props
}

func withType(props apiextensions.JSONSchemaProps, t string) *apiextensions.JSONSchemaProps {
	props.Type = t
	return &props
}

func withDefault(props apiextensions.JSONSchemaProps, d apiextensions.JSON) *apiextensions.JSONSchemaProps {
	props.Default = &d
	return &props
}

func withProperties(props apiextensions.JSONSchemaProps, p map[string]apiextensions.JSONSchemaProps) *apiextensions.JSONSchemaProps {
	props.Properties = p
	return &props
}

func withAdditionalProperties(props apiextensions.JSONSchemaProps, p apiextensions.JSONSchemaPropsOrBool) *apiextensions.JSONSchemaProps {
	props.AdditionalProperties = &p
	return &props
}

func withPatternProperties(props apiextensions.JSONSchemaProps, p map[string]apiextensions.JSONSchemaProps) *apiextensions.JSONSchemaProps {
	props.PatternProperties = p
	return &props
}

func withAnyOf(props apiextensions.JSONSchemaProps, ps []apiextensions.JSONSchemaProps) *apiextensions.JSONSchemaProps {
	props.AnyOf = ps
	return &props
}

func withAllOf(props apiextensions.JSONSchemaProps, ps []apiextensions.JSONSchemaProps) *apiextensions.JSONSchemaProps {
	props.AllOf = ps
	return &props
}

func withOneOf(props apiextensions.JSONSchemaProps, ps []apiextensions.JSONSchemaProps) *apiextensions.JSONSchemaProps {
	props.OneOf = ps
	return &props
}

func withNot(props apiextensions.JSONSchemaProps, p apiextensions.JSONSchemaProps) *apiextensions.JSONSchemaProps {
	props.Not = &p
	return &props
}

var fullSchemaWithDroppedFields = apiextensions.JSONSchemaProps{
	ID:               "a",
	Schema:           "b",
	Ref:              strPtr("c"),
	Description:      "d",
	Format:           "f",
	Title:            "g",
	Maximum:          float64Ptr(42.0),
	ExclusiveMaximum: true,
	Minimum:          float64Ptr(-42.0),
	ExclusiveMinimum: true,
	MaxLength:        int64Ptr(42),
	MinLength:        int64Ptr(7),
	Pattern:          "a*",
	MaxItems:         int64Ptr(42),
	MinItems:         int64Ptr(7),
	UniqueItems:      true,
	MultipleOf:       float64Ptr(42.0),
	Enum:             []apiextensions.JSON{1, 2, 3},
	MaxProperties:    int64Ptr(42),
	MinProperties:    int64Ptr(7),
	Required:         []string{"foo", "bar"},
	ExternalDocs: &apiextensions.ExternalDocumentation{
		Description: "a",
		URL:         "b",
	},
	Example: jsonPtr(42.0),
}

func jsonPtr(j apiextensions.JSON) *apiextensions.JSON {
	return &j
}

func strPtr(s string) *string {
	return &s
}

func float64Ptr(f float64) *float64 {
	return &f
}

func int64Ptr(i int64) *int64 {
	return &i
}
