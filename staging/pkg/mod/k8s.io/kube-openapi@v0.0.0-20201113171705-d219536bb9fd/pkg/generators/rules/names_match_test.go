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

package rules

import (
	"reflect"
	"testing"

	"k8s.io/gengo/types"
)

func TestNamesMatch(t *testing.T) {
	tcs := []struct {
		// name of test case
		name string
		t    *types.Type

		// expected list of violation fields
		expected []string
	}{
		// The comments are in format of {goName, jsonName, match},
		// {"PodSpec", "podSpec", true},
		{
			name: "simple",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `json:"podSpec"`,
					},
				},
			},
			expected: []string{},
		},
		// {"PodSpec", "podSpec", true},
		{
			name: "multiple_json_tags",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `json:"podSpec,omitempty"`,
					},
				},
			},
			expected: []string{},
		},
		// {"PodSpec", "podSpec", true},
		{
			name: "protobuf_tag",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `json:"podSpec,omitempty" protobuf:"bytes,1,opt,name=podSpec"`,
					},
				},
			},
			expected: []string{},
		},
		// {"", "podSpec", false},
		{
			name: "empty",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "",
						Tags: `json:"podSpec"`,
					},
				},
			},
			expected: []string{""},
		},
		// {"PodSpec", "PodSpec", false},
		{
			name: "CamelCase_CamelCase",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `json:"PodSpec"`,
					},
				},
			},
			expected: []string{"PodSpec"},
		},
		// {"podSpec", "podSpec", false},
		{
			name: "camelCase_camelCase",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "podSpec",
						Tags: `json:"podSpec"`,
					},
				},
			},
			expected: []string{"podSpec"},
		},
		// {"PodSpec", "spec", false},
		{
			name: "short_json_name",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `json:"spec"`,
					},
				},
			},
			expected: []string{"PodSpec"},
		},
		// {"Spec", "podSpec", false},
		{
			name: "long_json_name",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "Spec",
						Tags: `json:"podSpec"`,
					},
				},
			},
			expected: []string{"Spec"},
		},
		// {"JSONSpec", "jsonSpec", true},
		{
			name: "acronym",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "JSONSpec",
						Tags: `json:"jsonSpec"`,
					},
				},
			},
			expected: []string{},
		},
		// {"JSONSpec", "jsonspec", false},
		{
			name: "acronym_invalid",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "JSONSpec",
						Tags: `json:"jsonspec"`,
					},
				},
			},
			expected: []string{"JSONSpec"},
		},
		// {"HTTPJSONSpec", "httpJSONSpec", true},
		{
			name: "multiple_acronym",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "HTTPJSONSpec",
						Tags: `json:"httpJSONSpec"`,
					},
				},
			},
			expected: []string{},
		},
		// // NOTE: this validator cannot tell two sequential all-capital words from one word,
		// // therefore the case below is also considered matched.
		// {"HTTPJSONSpec", "httpjsonSpec", true},
		{
			name: "multiple_acronym_as_one",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "HTTPJSONSpec",
						Tags: `json:"httpjsonSpec"`,
					},
				},
			},
			expected: []string{},
		},
		// NOTE: JSON tags in jsonTagBlacklist should skip evaluation
		{
			name: "blacklist_tag_dash",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "podSpec",
						Tags: `json:"-"`,
					},
				},
			},
			expected: []string{},
		},
		// {"PodSpec", "-", false},
		{
			name: "invalid_json_name_dash",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `json:"-,"`,
					},
				},
			},
			expected: []string{"PodSpec"},
		},
		// NOTE: JSON names in jsonNameBlacklist should skip evaluation
		// {"", "", true},
		{
			name: "unspecified",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "",
						Tags: `json:""`,
					},
				},
			},
			expected: []string{},
		},
		// {"podSpec", "", true},
		{
			name: "blacklist_empty",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "podSpec",
						Tags: `json:""`,
					},
				},
			},
			expected: []string{},
		},
		// {"podSpec", "metadata", true},
		{
			name: "blacklist_metadata",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "podSpec",
						Tags: `json:"metadata"`,
					},
				},
			},
			expected: []string{},
		},
		{
			name: "non_struct",
			t: &types.Type{
				Kind: types.Map,
			},
			expected: []string{},
		},
		{
			name: "no_json_tag",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `podSpec`,
					},
				},
			},
			expected: []string{"PodSpec"},
		},
		// NOTE: this is to expand test coverage
		// {"S", "s", true},
		{
			name: "single_character",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "S",
						Tags: `json:"s"`,
					},
				},
			},
			expected: []string{},
		},
		// NOTE: names with disallowed substrings should fail evaluation
		// {"Pod-Spec", "pod-Spec", false},
		{
			name: "disallowed_substring_dash",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "Pod-Spec",
						Tags: `json:"pod-Spec"`,
					},
				},
			},
			expected: []string{"Pod-Spec"},
		},
		// {"Pod_Spec", "pod_Spec", false},
		{
			name: "disallowed_substring_underscore",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "Pod_Spec",
						Tags: `json:"pod_Spec"`,
					},
				},
			},
			expected: []string{"Pod_Spec"},
		},
	}

	n := &NamesMatch{}
	for _, tc := range tcs {
		if violations, _ := n.Validate(tc.t); !reflect.DeepEqual(violations, tc.expected) {
			t.Errorf("unexpected validation result: test name %v, want: %v, got: %v",
				tc.name, tc.expected, violations)
		}
	}
}

// TestRuleName tests the Name of API rule. This is to expand test coverage
func TestRuleName(t *testing.T) {
	ruleName := "names_match"
	n := &NamesMatch{}
	if n.Name() != ruleName {
		t.Errorf("unexpected API rule name: want: %v, got: %v", ruleName, n.Name())
	}
}
