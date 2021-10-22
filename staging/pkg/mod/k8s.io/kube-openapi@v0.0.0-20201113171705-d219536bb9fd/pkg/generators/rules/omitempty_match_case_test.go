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

func TestOmitEmptyMatchCase(t *testing.T) {
	tcs := []struct {
		// name of test case
		name string
		t    *types.Type

		// expected list of violation fields
		expected []string
	}{
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
		{
			name: "unserialized",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `json:"-,inline"`,
					},
				},
			},
			expected: []string{},
		},
		{
			name: "named_omitEmpty",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "OmitEmpty",
						Tags: `json:"omitEmpty,inline"`,
					},
				},
			},
			expected: []string{},
		},
		{
			name: "valid",
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
		{
			name: "invalid",
			t: &types.Type{
				Kind: types.Struct,
				Members: []types.Member{
					types.Member{
						Name: "PodSpec",
						Tags: `json:"podSpec,omitEmpty"`,
					},
				},
			},
			expected: []string{"PodSpec"},
		},
	}

	n := &OmitEmptyMatchCase{}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			if violations, _ := n.Validate(tc.t); !reflect.DeepEqual(violations, tc.expected) {
				t.Errorf("unexpected validation result: want: %v, got: %v", tc.expected, violations)
			}
		})
	}
}
