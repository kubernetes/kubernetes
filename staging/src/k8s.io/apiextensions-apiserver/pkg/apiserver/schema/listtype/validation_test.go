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

package listtype

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

func TestValidateListSetsAndMaps(t *testing.T) {
	tests := []struct {
		name   string
		schema *schema.Structural
		obj    map[string]interface{}
		errors []validationMatch
	}{
		{name: "nil"},
		{name: "no schema", obj: make(map[string]interface{})},
		{name: "no object", schema: &schema.Structural{}},
		{name: "list without schema",
			obj: map[string]interface{}{
				"array": []interface{}{"a", "b", "a"},
			},
		},
		{name: "list without items",
			obj: map[string]interface{}{
				"array": []interface{}{"a", "b", "a"},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Generic: schema.Generic{
							Type: "array",
						},
					},
				},
			},
		},

		{name: "set list with one item",
			obj: map[string]interface{}{
				"array": []interface{}{"a"},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
						Generic: schema.Generic{
							Type: "array",
						},
					},
				},
			},
		},
		{name: "set list with two equal items",
			obj: map[string]interface{}{
				"array": []interface{}{"a", "a"},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
						Generic: schema.Generic{
							Type: "array",
						},
					},
				},
			},
			errors: []validationMatch{
				duplicate("root", "array[1]"),
			},
		},
		{name: "set list with two different items",
			obj: map[string]interface{}{
				"array": []interface{}{"a", "b"},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
						Generic: schema.Generic{
							Type: "array",
						},
					},
				},
			},
		},
		{name: "set list with multiple duplicated items",
			obj: map[string]interface{}{
				"array": []interface{}{"a", "a", "b", "c", "d", "c"},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
						Generic: schema.Generic{
							Type: "array",
						},
					},
				},
			},
			errors: []validationMatch{
				duplicate("root", "array[1]"),
				duplicate("root", "array[5]"),
			},
		},

		{name: "normal list with items",
			obj: map[string]interface{}{
				"array": []interface{}{"a", "b", "a"},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "string",
							},
						},
					},
				},
			},
		},
		{name: "set list with items",
			obj: map[string]interface{}{
				"array": []interface{}{"a", "b", "a"},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Generic: schema.Generic{
							Type: "array",
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "string",
							},
						},
					},
				},
			},
			errors: []validationMatch{
				duplicate("root", "array[2]"),
			},
		},
		{name: "set list with items under additionalProperties",
			obj: map[string]interface{}{
				"array": []interface{}{"a", "b", "a"},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
					AdditionalProperties: &schema.StructuralOrBool{
						Structural: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Extensions: schema.Extensions{
								XListType: strPtr("set"),
							},
							Items: &schema.Structural{
								Generic: schema.Generic{
									Type: "string",
								},
							},
						},
					},
				},
			},
			errors: []validationMatch{
				duplicate("root[array][2]"),
			},
		},
		{name: "set list with items under items",
			obj: map[string]interface{}{
				"array": []interface{}{
					[]interface{}{"a", "b", "a"},
					[]interface{}{"b", "b", "a"},
				},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Extensions: schema.Extensions{
								XListType: strPtr("set"),
							},
							Items: &schema.Structural{
								Generic: schema.Generic{
									Type: "string",
								},
							},
						},
					},
				},
			},
			errors: []validationMatch{
				duplicate("root", "array[0][2]"),
				duplicate("root", "array[1][1]"),
			},
		},

		{name: "nested set lists",
			obj: map[string]interface{}{
				"array": []interface{}{
					"a", "b", "a", []interface{}{"b", "b", "a"},
				},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Generic: schema.Generic{
							Type: "array",
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Extensions: schema.Extensions{
								XListType: strPtr("set"),
							},
						},
					},
				},
			},
			errors: []validationMatch{
				duplicate("root", "array[2]"),
				duplicate("root", "array[3][1]"),
			},
		},

		{name: "set list with compound map items",
			obj: map[string]interface{}{
				"strings":             []interface{}{"a", "b", "a"},
				"integers":            []interface{}{int64(1), int64(2), int64(1)},
				"booleans":            []interface{}{false, true, true},
				"float64":             []interface{}{float64(1.0), float64(2.0), float64(2.0)},
				"nil":                 []interface{}{"a", nil, nil},
				"empty maps":          []interface{}{map[string]interface{}{"a": "b"}, map[string]interface{}{}, map[string]interface{}{}},
				"map values":          []interface{}{map[string]interface{}{"a": "b"}, map[string]interface{}{"a": "c"}, map[string]interface{}{"a": "b"}},
				"nil values":          []interface{}{map[string]interface{}{"a": nil}, map[string]interface{}{"b": "c", "a": nil}},
				"array":               []interface{}{[]interface{}{}, []interface{}{"a"}, []interface{}{"b"}, []interface{}{"a"}},
				"nil array":           []interface{}{[]interface{}{}, []interface{}{nil}, []interface{}{nil, nil}, []interface{}{nil}, []interface{}{"a"}},
				"multiple duplicates": []interface{}{map[string]interface{}{"a": "b"}, map[string]interface{}{"a": "c"}, map[string]interface{}{"a": "b"}, map[string]interface{}{"a": "c"}, map[string]interface{}{"a": "c"}},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"strings": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "string",
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"integers": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "integer",
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"booleans": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "boolean",
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"float64": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "number",
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"nil": {
						Generic: schema.Generic{
							Type: "array",
						}, Items: &schema.Structural{
							Generic: schema.Generic{
								Type:     "string",
								Nullable: true,
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"empty maps": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"map values": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"nil values": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type:     "string",
											Nullable: true,
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"array": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Items: &schema.Structural{
								Generic: schema.Generic{
									Type: "string",
								},
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"nil array": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Items: &schema.Structural{
								Generic: schema.Generic{
									Type:     "string",
									Nullable: true,
								},
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
					"multiple duplicates": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
					},
				},
			},
			errors: []validationMatch{
				duplicate("root", "strings[2]"),
				duplicate("root", "integers[2]"),
				duplicate("root", "booleans[2]"),
				duplicate("root", "float64[2]"),
				duplicate("root", "nil[2]"),
				duplicate("root", "empty maps[2]"),
				duplicate("root", "map values[2]"),
				duplicate("root", "array[3]"),
				duplicate("root", "nil array[3]"),
				duplicate("root", "multiple duplicates[2]"),
				duplicate("root", "multiple duplicates[3]"),
			},
		},
		{name: "set list with compound array items",
			obj: map[string]interface{}{
				"array": []interface{}{[]interface{}{}, []interface{}{"a"}, []interface{}{"a"}},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"array": {
						Generic: schema.Generic{
							Type: "array",
						},
						Extensions: schema.Extensions{
							XListType: strPtr("set"),
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "string",
							},
						},
					},
				},
			},
			errors: []validationMatch{
				duplicate("root", "array[2]"),
			},
		},

		{name: "map list with compound map items",
			obj: map[string]interface{}{
				"strings":                []interface{}{"a"},
				"integers":               []interface{}{int64(1)},
				"booleans":               []interface{}{false},
				"float64":                []interface{}{float64(1.0)},
				"nil":                    []interface{}{nil},
				"array":                  []interface{}{[]interface{}{"a"}},
				"one key":                []interface{}{map[string]interface{}{"a": "0", "c": "2"}, map[string]interface{}{"a": "1", "c": "1"}, map[string]interface{}{"a": "1", "c": "2"}, map[string]interface{}{}},
				"two keys":               []interface{}{map[string]interface{}{"a": "1", "b": "1", "c": "1"}, map[string]interface{}{"a": "1", "b": "2", "c": "2"}, map[string]interface{}{"a": "1", "b": "2", "c": "3"}, map[string]interface{}{}},
				"undefined key":          []interface{}{map[string]interface{}{"a": "1", "b": "1", "c": "1"}, map[string]interface{}{"a": "1", "c": "2"}, map[string]interface{}{"a": "1", "c": "3"}, map[string]interface{}{}},
				"compound key":           []interface{}{map[string]interface{}{"a": []interface{}{}, "c": "1"}, map[string]interface{}{"a": nil, "c": "1"}, map[string]interface{}{"a": []interface{}{"a"}, "c": "1"}, map[string]interface{}{"a": []interface{}{"a", int64(42)}, "c": "2"}, map[string]interface{}{"a": []interface{}{"a", int64(42)}, "c": []interface{}{"3"}}},
				"nil key":                []interface{}{map[string]interface{}{"a": []interface{}{}, "c": "1"}, map[string]interface{}{"a": nil, "c": "1"}, map[string]interface{}{"c": "1"}, map[string]interface{}{"a": nil}},
				"nil item":               []interface{}{nil, map[string]interface{}{"a": "0", "c": "1"}, map[string]interface{}{"a": nil}, map[string]interface{}{"c": "1"}},
				"nil item multiple keys": []interface{}{nil, map[string]interface{}{"b": "0", "c": "1"}, map[string]interface{}{"a": nil}, map[string]interface{}{"c": "1"}},
				"multiple duplicates": []interface{}{
					map[string]interface{}{"a": []interface{}{}, "c": "1"},
					map[string]interface{}{"a": nil, "c": "1"},
					map[string]interface{}{"a": []interface{}{"a"}, "c": "1"},
					map[string]interface{}{"a": []interface{}{"a", int64(42)}, "c": "2"},
					map[string]interface{}{"a": []interface{}{"a", int64(42)}, "c": []interface{}{"3"}},
					map[string]interface{}{"a": []interface{}{"a"}, "c": "1", "d": "1"},
					map[string]interface{}{"a": []interface{}{"a"}, "c": "1", "d": "2"},
				},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"strings": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "string",
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"integers": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "integer",
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"booleans": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "boolean",
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"float64": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "number",
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"nil": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type:     "string",
								Nullable: true,
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"array": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "array",
							},
							Items: &schema.Structural{
								Generic: schema.Generic{
									Type: "string",
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"one key": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"two keys": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a", "b"},
						},
					},
					"undefined key": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a", "b"},
						},
					},
					"compound key": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type:     "string",
											Nullable: true,
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"nil key": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
							},
							Properties: map[string]schema.Structural{
								"a": {
									Generic: schema.Generic{
										Type:     "array",
										Nullable: true,
									},
									Items: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
								"c": {
									Generic: schema.Generic{
										Type: "string",
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"nil item": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
							},
							Properties: map[string]schema.Structural{
								"a": {
									Generic: schema.Generic{
										Type:     "array",
										Nullable: true,
									},
									Items: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
								"c": {
									Generic: schema.Generic{
										Type: "string",
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
					"nil item multiple keys": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
							},
							Properties: map[string]schema.Structural{
								"a": {
									Generic: schema.Generic{
										Type:     "array",
										Nullable: true,
									},
									Items: &schema.Structural{
										Generic: schema.Generic{
											Type: "string",
										},
									},
								},
								"b": {
									Generic: schema.Generic{
										Type: "string",
									},
								},
								"c": {
									Generic: schema.Generic{
										Type: "string",
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a", "b"},
						},
					},
					"multiple duplicates": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "object",
								AdditionalProperties: &schema.StructuralOrBool{
									Structural: &schema.Structural{
										Generic: schema.Generic{
											Type:     "string",
											Nullable: true,
										},
									},
								},
							},
						},
						Extensions: schema.Extensions{
							XListType:    strPtr("map"),
							XListMapKeys: []string{"a"},
						},
					},
				},
			},
			errors: []validationMatch{
				invalid("root", "strings[0]"),
				invalid("root", "integers[0]"),
				invalid("root", "booleans[0]"),
				invalid("root", "float64[0]"),
				invalid("root", "array[0]"),
				duplicate("root", "one key[2]"),
				duplicate("root", "two keys[2]"),
				duplicate("root", "undefined key[2]"),
				duplicate("root", "compound key[4]"),
				duplicate("root", "nil key[3]"),
				duplicate("root", "nil item[3]"),
				duplicate("root", "nil item multiple keys[3]"),
				duplicate("root", "multiple duplicates[4]"),
				duplicate("root", "multiple duplicates[5]"),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := ValidateListSetsAndMaps(field.NewPath("root"), tt.schema, tt.obj)

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

func (v validationMatch) matches(err *field.Error) bool {
	return err.Type == v.errorType && err.Field == v.path.String()
}

func duplicate(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeDuplicate}
}
func invalid(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeInvalid}
}

func strPtr(s string) *string { return &s }
