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

package nodedeclaredfeatures

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestFeatureMapper(t *testing.T) {
	features := []string{"a", "b", "c", "d"}
	mapper := NewFeatureMapper(features)

	tests := []struct {
		name        string
		input       []string
		expectError bool
		expectedSet sets.Set[int] // indices in features
	}{
		{
			name:        "Empty input",
			input:       nil,
			expectedSet: nil,
		},
		{
			name:        "Single valid feature",
			input:       []string{"a"},
			expectedSet: sets.New(0),
		},
		{
			name:        "Multiple valid features",
			input:       []string{"a", "c"},
			expectedSet: sets.New(0, 2),
		},
		{
			name:        "All features",
			input:       []string{"a", "b", "c", "d"},
			expectedSet: sets.New(0, 1, 2, 3),
		},
		{
			name:        "Unknown feature (start)",
			input:       []string{"0", "a"},
			expectError: true,
			expectedSet: sets.New(0),
		},
		{
			name:        "Unknown feature (middle)",
			input:       []string{"a", "ab", "b"},
			expectError: true,
			expectedSet: sets.New(0, 1),
		},
		{
			name:        "Unknown feature (end)",
			input:       []string{"d", "e"},
			expectError: true,
			expectedSet: sets.New(3),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test TryMap
			fs := mapper.TryMap(tt.input)
			for i, known := range features {
				assert.Equal(t, tt.expectedSet.Has(i), fs.Get(i), "Feature %s (index %d) state mismatch", known, i)
			}

			// Test MapSorted
			fs2, err := mapper.MapSorted(tt.input)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, fs, fs2)
			}

			// Test MustMapSorted
			if tt.expectError {
				assert.Panics(t, func() { mapper.MustMapSorted(tt.input) })
			} else {
				assert.Equal(t, fs, mapper.MustMapSorted(tt.input))
			}

			// Test Unmap
			unmapped := mapper.Unmap(fs)
			assert.Len(t, unmapped, tt.expectedSet.Len())
			if tt.expectError {
				// If there's an error, just verify that the unmapped results of TryMap are
				// contained within the input.
				assert.Subset(t, tt.input, unmapped)
			} else {
				assert.Equal(t, tt.input, unmapped)
			}
		})
	}
}
