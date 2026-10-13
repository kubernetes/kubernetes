/*
Copyright 2025 The Kubernetes Authors.

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

package architecture

import (
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestCompareObjects(t *testing.T) {
	for name, tc := range map[string]struct {
		expected, actual string
		expectedDiff     string
	}{
		"equal": {
			expected: `{"kind":"test","hello":"world"}`,
			actual:   `{"kind":"test","hello":"world"}`,
		},

		"missing": {
			expected: `{"kind":"test","hello":"world"}`,
			actual:   `{"kind":"test"}`,
			expectedDiff: `  map[string]any{
- 	"hello": string("world"),
  	"kind":  string("test"),
  }
`,
		},

		"added": {
			expected: `{"kind":"test","hello":"world"}`,
			actual:   `{"kind":"test","hello":"world","foo":"bar"}`,
		},

		"replaced": {
			expected: `{"kind":"test","hello":"world"}`,
			actual:   `{"kind":"test","hello":1}`,
			expectedDiff: `  map[string]any{
- 	"hello": string("world"),
+ 	"hello": int64(1),
  	"kind":  string("test"),
  }
`,
		},

		"recursive": {
			expected: `{"kind":"test","spec":{"hello":"world","removed":42}}`,
			actual:   `{"kind":"test","spec":{"hello":1,"added":42}}`,
			expectedDiff: `  map[string]any{
  	"kind": string("test"),
  	"spec": map[string]any{
  		... // 1 ignored entry
- 		"hello":   string("world"),
+ 		"hello":   int64(1),
- 		"removed": int64(42),
  	},
  }
`,
		},

		"list": {
			expected: `{"kind":"test","items":[{"index":0},{"hello":"world","removed":42}]}`,
			actual:   `{"kind":"test","items":[{"index":0,"added":true},{"hello":1,"added":42},{"new-entry": true},"new-non-object-entry"]}`,
			expectedDiff: `  map[string]any{
  	"items": []any{
  		map[string]any{"index": int64(0), ...},
+ 		map[string]any{"added": int64(42), "hello": int64(1)},
+ 		map[string]any{"new-entry": bool(true)},
- 		map[string]any{"hello": string("world"), "removed": int64(42)},
+ 		string("new-non-object-entry"),
  	},
  	"kind": string("test"),
  }`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			var expected, actual unstructured.Unstructured
			require.NoError(t, expected.UnmarshalJSON([]byte(tc.expected)), "unmarshal expected")
			require.NoError(t, actual.UnmarshalJSON([]byte(tc.actual)), "unmarshal actual")
			actualDiff := compareObjects(&expected, &actual)
			t.Logf("Actual diff:\n%s", actualDiff)
			// Upstream go-cmp does not want the diff output to be checked in
			// tests because it is not stable. They intentionally randomly
			// switch between space and non-break space to enforce that
			// (https://github.com/google/go-cmp/issues/366).
			//
			// Therefore the expected diff is merely informative, we only check
			// for empty vs. not empty.
			if tc.expectedDiff == "" {
				require.Empty(t, actualDiff)
			} else {
				require.NotEmpty(t, actualDiff)
			}
		})
	}
}
