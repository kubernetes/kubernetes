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

package apidefinitions

import (
	"encoding/json"
	"testing"
)

func TestContainsAll(t *testing.T) {
	tests := []struct {
		name   string
		subset string
		value  string
		out    bool
	}{
		{
			name:   "equal maps",
			subset: `{"a": 1, "b": "x"}`,
			value:  `{"a": 1, "b": "x"}`,
			out:    true,
		},
		{
			name:   "subset map match",
			subset: `{"a": 1}`,
			value:  `{"a": 1, "b": "x"}`,
			out:    true,
		},
		{
			name:   "map extra key",
			subset: `{"a": 1, "b": "x"}`,
			value:  `{"a": 1}`,
			out:    false,
		},
		{
			name:   "not equal scalar",
			subset: `{"a": 1}`,
			value:  `{"a": 2}`,
			out:    false,
		},
		{
			name:   "nested map match",
			subset: `{"spec": {"replicas": 3}}`,
			value:  `{"spec": {"replicas": 3, "selector": {"matchLabels": {"app": "x"}}}}`,
			out:    true,
		},
		{
			name:   "nested map mismatch",
			subset: `{"spec": {"replicas": 3}}`,
			value:  `{"spec": {"replicas": 4}}`,
			out:    false,
		},
		{
			name:   "equal slices",
			subset: `{"a": [1, 2, 3]}`,
			value:  `{"a": [1, 2, 3]}`,
			out:    true,
		},
		{
			name:   "subset slice match",
			subset: `{"a": [1, 2]}`,
			value:  `{"a": [1, 2, 3]}`,
			out:    true,
		},
		{
			name:   "slice of scalars: missing element",
			subset: `{"a": [1, 4]}`,
			value:  `{"a": [1, 2, 3]}`,
			out:    false,
		},
		{
			name:   "slice of maps reordered",
			subset: `{"conditions": [{"type": "A"}, {"type": "B"}]}`,
			value:  `{"conditions": [{"type": "B", "status": "True"}, {"type": "A", "status": "False"}]}`,
			out:    true,
		},
		{
			name:   "slice of maps with missing element",
			subset: `{"conditions": [{"type": "A"}, {"type": "B"}]}`,
			value:  `{"conditions": [{"type": "A"}, {"type": "C"}]}`,
			out:    false,
		},
		{
			name:   "type mismatch map vs scalar",
			subset: `{"a": {"b": 1}}`,
			value:  `{"a": 1}`,
			out:    false,
		},
		{
			name:   "type mismatch slice vs map",
			subset: `{"a": [1]}`,
			value:  `{"a": {"0": 1}}`,
			out:    false,
		},
		{
			name:   "empty want map",
			subset: `{}`,
			value:  `{"a": 1}`,
			out:    true,
		},
		{
			name:   "empty want slice",
			subset: `{"a": []}`,
			value:  `{"a": [1, 2]}`,
			out:    true,
		},
		{
			name:   "deeply nested subset",
			subset: `{"a": {"b": {"c": [{"d": 1}]}}}`,
			value:  `{"a": {"b": {"c": [{"d": 1, "e": 2}], "extra": true}}}`,
			out:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var want, got any
			if err := json.Unmarshal([]byte(tc.subset), &want); err != nil {
				t.Fatalf("invalid JSON: %v", err)
			}
			if err := json.Unmarshal([]byte(tc.value), &got); err != nil {
				t.Fatalf("invalid JSON: %v", err)
			}
			if result := containsAll(want, got); result != tc.out {
				t.Errorf("containsAll() = %v, want %v", result, tc.out)
			}
		})
	}
}
