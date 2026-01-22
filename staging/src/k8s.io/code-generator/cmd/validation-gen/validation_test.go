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

package main

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
)

// gengo has `PointerTo()` but not the rest, so keep this here for consistency.
func ptrTo(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "*" + t.Name.String(),
		},
		Kind: types.Pointer,
		Elem: t,
	}
}

func sliceOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "[]" + t.Name.String(),
		},
		Kind: types.Slice,
		Elem: t,
	}
}

func mapOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "map[string]" + t.Name.String(),
		},
		Kind: types.Map,
		Key:  types.String,
		Elem: t,
	}
}

func aliasOf(name string, t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "Alias_" + name,
		},
		Kind:       types.Alias,
		Underlying: t,
	}
}

func TestGetLeafTypeAndPrefixes(t *testing.T) {

	cases := []struct {
		in              *types.Type
		expectedType    *types.Type
		expectedTypePfx string
		expectedExprPfx string
	}{{
		// string
		in:              types.String,
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "&",
	}, {
		// *string
		in:              ptrTo(types.String),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "",
	}, {
		// **string
		in:              ptrTo(ptrTo(types.String)),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "*",
	}, {
		// ***string
		in:              ptrTo(ptrTo(ptrTo(types.String))),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "**",
	}, {
		// []string
		in:              sliceOf(types.String),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]string
		in:              ptrTo(sliceOf(types.String)),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]string
		in:              ptrTo(ptrTo(sliceOf(types.String))),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(types.String)))),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]string
		in:              mapOf(types.String),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]string
		in:              ptrTo(mapOf(types.String)),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]string
		in:              ptrTo(ptrTo(mapOf(types.String))),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]string
		in:              ptrTo(ptrTo(ptrTo(mapOf(types.String)))),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// alias of string
		in:              aliasOf("s", types.String),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "&",
	}, {
		// alias of *string
		in:              aliasOf("ps", ptrTo(types.String)),
		expectedType:    aliasOf("ps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **string
		in:              aliasOf("pps", ptrTo(ptrTo(types.String))),
		expectedType:    aliasOf("pps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***string
		in:              aliasOf("ppps", ptrTo(ptrTo(ptrTo(types.String)))),
		expectedType:    aliasOf("ppps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of []string
		in:              aliasOf("ls", sliceOf(types.String)),
		expectedType:    aliasOf("ls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of *[]string
		in:              aliasOf("pls", ptrTo(sliceOf(types.String))),
		expectedType:    aliasOf("pls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **[]string
		in:              aliasOf("ppls", ptrTo(ptrTo(sliceOf(types.String)))),
		expectedType:    aliasOf("ppls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***[]string
		in:              aliasOf("pppls", ptrTo(ptrTo(ptrTo(sliceOf(types.String))))),
		expectedType:    aliasOf("pppls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of map[string]string
		in:              aliasOf("ms", mapOf(types.String)),
		expectedType:    aliasOf("ms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of *map[string]string
		in:              aliasOf("pms", ptrTo(mapOf(types.String))),
		expectedType:    aliasOf("pms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **map[string]string
		in:              aliasOf("ppms", ptrTo(ptrTo(mapOf(types.String)))),
		expectedType:    aliasOf("ppms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***map[string]string
		in:              aliasOf("pppms", ptrTo(ptrTo(ptrTo(mapOf(types.String))))),
		expectedType:    aliasOf("pppms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *alias-of-string
		in:              ptrTo(aliasOf("s", types.String)),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "",
	}, {
		// **alias-of-string
		in:              ptrTo(ptrTo(aliasOf("s", types.String))),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "*",
	}, {
		// ***alias-of-string
		in:              ptrTo(ptrTo(ptrTo(aliasOf("s", types.String)))),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "**",
	}, {
		// []alias-of-string
		in:              sliceOf(aliasOf("s", types.String)),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]alias-of-string
		in:              ptrTo(sliceOf(aliasOf("s", types.String))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]alias-of-string
		in:              ptrTo(ptrTo(sliceOf(aliasOf("s", types.String)))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]alias-of-string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(aliasOf("s", types.String))))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]alias-of-string
		in:              mapOf(aliasOf("s", types.String)),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]alias-of-string
		in:              ptrTo(mapOf(aliasOf("s", types.String))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]alias-of-string
		in:              ptrTo(ptrTo(mapOf(aliasOf("s", types.String)))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]alias-of-string
		in:              ptrTo(ptrTo(ptrTo(mapOf(aliasOf("s", types.String))))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// *alias-of-*string
		in:              ptrTo(aliasOf("ps", ptrTo(types.String))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **alias-of-*string
		in:              ptrTo(ptrTo(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// []alias-of-*string
		in:              sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]alias-of-*string
		in:              ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]alias-of-*string
		in:              ptrTo(ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String)))))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]alias-of-*string
		in:              mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]alias-of-*string
		in:              ptrTo(mapOf(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]alias-of-*string
		in:              ptrTo(ptrTo(mapOf(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(mapOf(aliasOf("ps", ptrTo(types.String)))))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}}

	for _, tc := range cases {
		leafType, typePfx, exprPfx := getLeafTypeAndPrefixes(tc.in)
		if got, want := leafType.Name.String(), tc.expectedType.Name.String(); got != want {
			t.Errorf("%q: wrong leaf type: expected %q, got %q", tc.in, want, got)
		}
		if got, want := typePfx, tc.expectedTypePfx; got != want {
			t.Errorf("%q: wrong type prefix: expected %q, got %q", tc.in, want, got)
		}
		if got, want := exprPfx, tc.expectedExprPfx; got != want {
			t.Errorf("%q: wrong expr prefix: expected %q, got %q", tc.in, want, got)
		}
	}
}

func TestSortIntoCohorts(t *testing.T) {
	cases := []struct {
		in       []validators.FunctionGen
		expected [][]validators.FunctionGen
	}{{
		// empty
		in:       []validators.FunctionGen{},
		expected: [][]validators.FunctionGen{},
	}, {
		// default cohort
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		}},
	}, {
		// default cohort, not already sorted by name
		in: []validators.FunctionGen{
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
		}},
	}, {
		// default cohort, with a short-circuit
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "b", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		}},
	}, {
		// default cohort, with 2 short-circuits
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "d", Cohort: "", Flags: validators.ShortCircuit},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "b", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "d", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		}},
	}, {
		// default and non-default cohorts
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "foo", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "bar", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "d", Cohort: "foo", Flags: validators.DefaultFlags},
			{TagName: "e", Cohort: "bar", Flags: validators.DefaultFlags},
			{TagName: "f", Cohort: "", Flags: validators.DefaultFlags},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "f", Cohort: "", Flags: validators.DefaultFlags},
		}, {
			{TagName: "a", Cohort: "foo", Flags: validators.DefaultFlags},
			{TagName: "d", Cohort: "foo", Flags: validators.DefaultFlags},
		}, {
			{TagName: "b", Cohort: "bar", Flags: validators.DefaultFlags},
			{TagName: "e", Cohort: "bar", Flags: validators.DefaultFlags},
		}},
	}, {
		// default and non-default cohorts with short-circuit
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "foo", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "bar", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "d", Cohort: "foo", Flags: validators.ShortCircuit},
			{TagName: "e", Cohort: "bar", Flags: validators.ShortCircuit},
			{TagName: "f", Cohort: "", Flags: validators.ShortCircuit},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "f", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		}, {
			{TagName: "d", Cohort: "foo", Flags: validators.ShortCircuit},
			{TagName: "a", Cohort: "foo", Flags: validators.DefaultFlags},
		}, {
			{TagName: "e", Cohort: "bar", Flags: validators.ShortCircuit},
			{TagName: "b", Cohort: "bar", Flags: validators.DefaultFlags},
		}},
	}}

	for _, tc := range cases {
		out := sortIntoCohorts(tc.in)
		if !reflect.DeepEqual(out, tc.expected) {
			t.Errorf("expected %v, got %v", tc.expected, out)
		}
	}
}

func TestDiscoverStruct(t *testing.T) {
	c := &generator.Context{
		Namers:    namer.NameSystems{},
		Universe:  types.Universe{},
		FileTypes: map[string]generator.FileType{},
	}
	validator := validators.InitGlobalValidator(c)

	testCases := []struct {
		name                   string
		typeToTest             *types.Type
		expectErr              error
		expectedStabilityLevel validators.StabilityLevel
	}{
		{
			name: "simple struct with stable validations",
			typeToTest: &types.Type{
				Kind: types.Struct,
				Name: types.Name{Name: "MyStruct"},
				Members: []types.Member{
					{
						Name:         "StringField",
						Type:         types.String,
						CommentLines: []string{"+k8s:required"},
						Tags:         `json:"stringField"`,
					},
					{
						Name:         "IntegerField",
						Type:         types.Int64,
						CommentLines: []string{"+k8s:minimum=1", "+k8s:maximum=10"},
						Tags:         `json:"integerField"`,
					},
					{
						Name: "StringSliceField",
						Type: &types.Type{
							Kind: types.Slice,
							Elem: types.String,
						},
						CommentLines: []string{"+k8s:maxItems=10", "+k8s:listType=atomic"},
						Tags:         `json:"sliceStringField"`,
					},
				},
			},
			expectErr:              nil,
			expectedStabilityLevel: validators.Stable,
		},
		{
			name: "struct with non-stable validation on a field",
			typeToTest: &types.Type{
				Kind: types.Struct,
				Name: types.Name{Name: "MyStruct"},
				Members: []types.Member{
					{
						Name:         "AlphaField",
						Type:         types.String,
						CommentLines: []string{"+k8s:validateFalse"},
						Tags:         `json:"alphaField"`,
					},
				},
			},
			expectErr:              nil,
			expectedStabilityLevel: validators.Alpha,
		},
		{
			name: "struct with declarative native fields with stable validations",
			typeToTest: &types.Type{
				Kind: types.Struct,
				Name: types.Name{Name: "MyStruct"},
				Members: []types.Member{
					{
						Name:         "DeclarativeField",
						Type:         types.String,
						CommentLines: []string{"+k8s:declarativeValidationNative", "+k8s:required", "+k8s:format=k8s-uuid"},
						Tags:         `json:"declarativeField"`,
					}, {
						Name:         "DeclarativeField",
						Type:         types.Int64,
						CommentLines: []string{"+k8s:declarativeValidationNative", "+k8s:minimum=1", "+k8s:maximum=10"},
						Tags:         `json:"declarativeField"`,
					}, {
						Name: "DeclarativeField",
						Type: &types.Type{
							Kind: types.Slice,
							Elem: types.String,
						},
						CommentLines: []string{"+k8s:declarativeValidationNative", "+k8s:maxItems=10", "+k8s:listType=atomic", "+k8s:required"},
						Tags:         `json:"declarativeField"`,
					},
				},
			},
			expectErr:              nil,
			expectedStabilityLevel: validators.Stable,
		},
		{
			name: "struct with declarative native field string with non-stable validations",
			typeToTest: &types.Type{
				Kind: types.Struct,
				Name: types.Name{Name: "MyStruct"},
				Members: []types.Member{
					{
						Name:         "DeclarativeField",
						Type:         types.String,
						CommentLines: []string{"+k8s:declarativeValidationNative", "+k8s:validateFalse", "+k8s:required"},
						Tags:         `json:"declarativeField"`,
					},
				},
			},
			expectErr: fmt.Errorf("field MyStruct.declarativeField: +k8s:declarativeValidationNative can only be used with stable validation tags, but found \"k8s:validateFalse\" which is Alpha"),
		},
		{
			name: "struct with declarative native field integer with non-stable validations",
			typeToTest: &types.Type{
				Kind: types.Struct,
				Name: types.Name{Name: "MyStruct"},
				Members: []types.Member{
					{
						Name:         "DeclarativeField",
						Type:         types.Int64,
						CommentLines: []string{"+k8s:declarativeValidationNative", "+k8s:validateFalse", "+k8s:required"},
						Tags:         `json:"declarativeField"`,
					},
				},
			},
			expectErr: fmt.Errorf("field MyStruct.declarativeField: +k8s:declarativeValidationNative can only be used with stable validation tags, but found \"k8s:validateFalse\" which is Alpha"),
		},
		{
			name: "struct with declarative native field slice with stable validations (item+zeroOrOneOf)",
			typeToTest: &types.Type{
				Kind: types.Struct,
				Name: types.Name{Name: "MyStructSlice"},
				Members: []types.Member{
					{
						Name: "DeclarativeField",
						Type: &types.Type{
							Kind: types.Slice,
							Elem: &types.Type{
								Kind: types.Struct,
								Name: types.Name{Name: "MyInnerStruct"},
								Members: []types.Member{
									{Name: "Name", Type: types.String, Tags: `json:"name"`},
								},
							},
						},
						CommentLines: []string{"+k8s:declarativeValidationNative", "+k8s:listType=map", "+k8s:listMapKey=name", `+k8s:item(name: "failed")=+k8s:zeroOrOneOfMember`},
						Tags:         `json:"declarativeField"`,
					},
				},
			},
			expectErr:              nil,
			expectedStabilityLevel: validators.Stable,
		},
		{
			name: "struct with a field whose type has non-stable validations",
			typeToTest: &types.Type{
				Kind: types.Struct,
				Name: types.Name{Name: "MyStruct"},
				Members: []types.Member{
					{
						Name: "OtherField",
						Type: &types.Type{
							Kind: types.Struct,
							Name: types.Name{Name: "OtherType"},
							Members: []types.Member{
								{
									Name:         "AlphaField",
									Type:         types.String,
									CommentLines: []string{"+k8s:validateFalse"},
									Tags:         `json:"alphaField"`,
								},
							},
						},
						Tags: `json:"otherField"`,
					},
				},
			},
			expectErr:              nil,
			expectedStabilityLevel: validators.Alpha,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			discoverer := NewTypeDiscoverer(validator, map[string]string{})
			if err := discoverer.Init(c); err != nil {
				t.Fatalf("discoverer.Init() failed: %v", err)
			}

			// Manually discover the types to populate the typeNodes map
			if err := discoverer.DiscoverType(tc.typeToTest); err != nil {
				if tc.expectErr != nil {
					if err.Error() != tc.expectErr.Error() {
						t.Fatalf("expected error %q, but got %q", tc.expectErr, err)
					}
					return
				}
				t.Fatalf("discoverer.DiscoverType() failed: %v", err)
			}

			thisNode := discoverer.typeNodes[tc.typeToTest]
			if thisNode == nil {
				t.Fatalf("typeNode for %s not found", tc.typeToTest.Name.Name)
			}

			if tc.expectErr != nil {
				// Error was expected during DiscoverType
				return
			}

			if thisNode.lowestStabilityLevel != tc.expectedStabilityLevel {
				t.Errorf("Expected lowestStabilityLevel to be %v, but got %v", tc.expectedStabilityLevel, thisNode.lowestStabilityLevel)
			}
		})
	}
}
