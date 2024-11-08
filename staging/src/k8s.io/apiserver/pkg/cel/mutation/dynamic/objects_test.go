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
limitations under the License.
*/

package dynamic

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

func TestOptional(t *testing.T) {
	for _, tc := range []struct {
		name     string
		fields   map[string]ref.Val
		expected map[string]any
	}{
		{
			name: "present",
			fields: map[string]ref.Val{
				"zero": types.OptionalOf(types.IntZero),
			},
			expected: map[string]any{
				"zero": int64(0),
			},
		},
		{
			name: "none",
			fields: map[string]ref.Val{
				"absent": types.OptionalNone,
			},
			expected: map[string]any{
				// right now no way to differ from a plain null.
				// we will need to filter out optional.none() before this conversion.
				"absent": nil,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := &ObjectVal{
				objectType: nil, // safe in this test, otherwise put a mock
				fields:     tc.fields,
			}
			converted := v.Value()
			if !reflect.DeepEqual(tc.expected, converted) {
				t.Errorf("wrong result, expected %v but got %v", tc.expected, converted)
			}
		})
	}
}

func TestCheckTypeNamesMatchFieldPathNames(t *testing.T) {
	for _, tc := range []struct {
		name        string
		obj         *ObjectVal
		expectError string
	}{
		{
			name: "valid",
			obj: &ObjectVal{
				objectType: types.NewObjectType("Object"),
				fields: map[string]ref.Val{
					"spec": &ObjectVal{
						objectType: types.NewObjectType("Object.spec"),
						fields: map[string]ref.Val{
							"replicas": types.Int(100),
							"m": types.NewRefValMap(nil, map[ref.Val]ref.Val{
								types.String("k1"): &ObjectVal{
									objectType: types.NewObjectType("Object.spec.m"),
								},
							}),
							"l": types.NewRefValList(nil, []ref.Val{
								&ObjectVal{
									objectType: types.NewObjectType("Object.spec.l"),
								},
							}),
						},
					},
				},
			},
		},
		{
			name: "invalid struct field",
			obj: &ObjectVal{
				objectType: types.NewObjectType("Object"),
				fields: map[string]ref.Val{"invalid": &ObjectVal{
					objectType: types.NewObjectType("Object.spec"),
					fields:     map[string]ref.Val{"replicas": types.Int(100)},
				}},
			},
			expectError: "unexpected type name \"Object.spec\", expected \"Object.invalid\"",
		},
		{
			name: "invalid map field",
			obj: &ObjectVal{
				objectType: types.NewObjectType("Object"),
				fields: map[string]ref.Val{
					"spec": &ObjectVal{
						objectType: types.NewObjectType("Object.spec"),
						fields: map[string]ref.Val{
							"replicas": types.Int(100),
							"m": types.NewRefValMap(nil, map[ref.Val]ref.Val{
								types.String("k1"): &ObjectVal{
									objectType: types.NewObjectType("Object.spec.invalid"),
								},
							}),
						},
					},
				},
			},
			expectError: "unexpected type name \"Object.spec.invalid\", expected \"Object.spec.m\"",
		},
		{
			name: "invalid list field",
			obj: &ObjectVal{
				objectType: types.NewObjectType("Object"),
				fields: map[string]ref.Val{
					"spec": &ObjectVal{
						objectType: types.NewObjectType("Object.spec"),
						fields: map[string]ref.Val{
							"replicas": types.Int(100),
							"l": types.NewRefValList(nil, []ref.Val{
								&ObjectVal{
									objectType: types.NewObjectType("Object.spec.invalid"),
								},
							}),
						},
					},
				},
			},
			expectError: "unexpected type name \"Object.spec.invalid\", expected \"Object.spec.l\"",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.obj.CheckTypeNamesMatchFieldPathNames()
			if tc.expectError == "" {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("expected error")
				}
				if !strings.Contains(err.Error(), tc.expectError) {
					t.Errorf("expected error to contain %v, got %v", tc.expectError, err)
				}
			}
		})
	}
}
