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
)

func TestNewBitmap(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		expected int // number of uint64 elements
	}{
		{"Size 0", 0, 0},
		{"Size 1", 1, 1},
		{"Size 63", 63, 1},
		{"Size 64", 64, 1}, // (64+63)/64 = 1
		{"Size 65", 65, 2},
		{"Size 128", 128, 2}, // (128+63)/64 = 2
		{"Size 129", 129, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newBitmap(tt.size)
			assert.Len(t, b, tt.expected)
		})
	}
}

func TestBitmap_GetSet(t *testing.T) {
	b := newBitmap(128)

	// Initially all false
	assert.False(t, b.Get(0))
	assert.False(t, b.Get(63))
	assert.False(t, b.Get(64))
	assert.False(t, b.Get(127))

	// Set bits
	b.Set(0)
	b.Set(63)
	b.Set(64)
	b.Set(127)

	assert.True(t, b.Get(0))
	assert.True(t, b.Get(63))
	assert.True(t, b.Get(64))
	assert.True(t, b.Get(127))

	// Verify other bits are still false
	assert.False(t, b.Get(1))
	assert.False(t, b.Get(62))
	assert.False(t, b.Get(65))

	// Error bounds.
	assert.PanicsWithValue(t, "bitmap index out of range: -1", func() {
		b.Set(-1)
	})
	assert.PanicsWithValue(t, "bitmap index out of range: -1", func() {
		b.Get(-1)
	})
	assert.PanicsWithValue(t, "bitmap index out of range: 128", func() {
		b.Set(128)
	})
	assert.PanicsWithValue(t, "bitmap index out of range: 128", func() {
		b.Set(128)
	})
}

func TestBitmap_DifferenceSubset(t *testing.T) {
	tests := []struct {
		name         string
		bSize        int
		otherSize    int
		bSets        []int
		otherSets    []int
		expectedErr  bool
		expectedDiff []int // expect other.Contains IFF expectedDiff is empty.
	}{
		{
			name:        "Mismatch size",
			bSize:       64,
			otherSize:   128,
			expectedErr: true,
		},
		{
			name:         "Disjoint sets",
			bSize:        64,
			otherSize:    64,
			bSets:        []int{1, 2},
			otherSets:    []int{3, 4},
			expectedDiff: []int{1, 2},
		},
		{
			name:         "Overlap",
			bSize:        64,
			otherSize:    64,
			bSets:        []int{1, 2, 3},
			otherSets:    []int{2, 3, 4},
			expectedDiff: []int{1}, // 2 and 3 removed
		},
		{
			name:         "Identical",
			bSize:        64,
			otherSize:    64,
			bSets:        []int{1, 2},
			otherSets:    []int{1, 2},
			expectedDiff: []int{},
		},
		{
			name:         "Subset",
			bSize:        64,
			otherSize:    64,
			bSets:        []int{1, 2},
			otherSets:    []int{1, 2, 3},
			expectedDiff: []int{},
		},
		{
			name:         "Superset",
			bSize:        64,
			otherSize:    64,
			bSets:        []int{1, 2, 3},
			otherSets:    []int{1, 2},
			expectedDiff: []int{3},
		},
		{
			name:         "Empty contains Empty",
			bSize:        64,
			otherSize:    64,
			expectedDiff: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newBitmap(tt.bSize)
			for _, i := range tt.bSets {
				b.Set(i)
			}
			other := newBitmap(tt.otherSize)
			for _, i := range tt.otherSets {
				other.Set(i)
			}

			// Test Difference
			diff, err := b.Difference(other)
			if tt.expectedErr {
				assert.Error(t, err)
				assert.Nil(t, diff)
			} else {
				require.NoError(t, err)
				for _, i := range tt.expectedDiff {
					assert.True(t, diff.Get(i), "Difference: Index %d should be set", i)
				}
				// Verify count of set bits for Difference
				count := 0
				for i := 0; i < tt.bSize; i++ {
					if diff.Get(i) {
						count++
					}
				}
				assert.Equal(t, len(tt.expectedDiff), count, "Difference: Unexpected number of bits set")
			}

			// Test IsSubset
			result, err := b.IsSubset(other)
			if tt.expectedErr {
				assert.Error(t, err)
				assert.False(t, result)
			} else {
				require.NoError(t, err)
				assert.Equal(t, len(tt.expectedDiff) == 0, result, "IsSubset result mismatch")
			}
		})
	}
}

func TestBitmap_Equal(t *testing.T) {
	b1 := newBitmap(64)
	b1.Set(1)

	b2 := newBitmap(64)
	b2.Set(1)

	b3 := newBitmap(64)
	b3.Set(2)

	b4 := newBitmap(128) // Different size
	b4.Set(1)

	assert.True(t, b1.Equal(b2))
	assert.False(t, b1.Equal(b3))
	assert.False(t, b1.Equal(b4))
}

func TestBitmap_IsEmpty(t *testing.T) {
	b := newBitmap(64)
	assert.True(t, b.IsEmpty())

	b.Set(10)
	assert.False(t, b.IsEmpty())
}

func TestBitmap_Clone(t *testing.T) {
	b := newBitmap(64)
	b.Set(1)

	clone := b.Clone()
	assert.True(t, b.Equal(clone))

	// Modify clone, original should not change
	clone.Set(2)
	assert.False(t, clone.Equal(b))
	assert.False(t, b.Get(2))
	assert.True(t, clone.Get(2))
}

func TestBitmap_String(t *testing.T) {
	// Size 64 -> 1 uint64
	b := newBitmap(64)
	// 0x0000000000000001 (bit 0 set)
	b.Set(0)
	// Output is hex encoded BigEndian of the uint64
	// uint64(1) -> bytes: [0 0 0 0 0 0 0 1]
	// hex: "0000000000000001"
	assert.Equal(t, "0000000000000001", b.String())

	b2 := newBitmap(64)
	b2.Set(63)
	// 1 << 63 is the highest bit.
	// 0x8000000000000000
	// bytes: [128 0 0 0 0 0 0 0] -> hex "8000000000000000"
	assert.Equal(t, "8000000000000000", b2.String())

	// Multi-word
	b3 := newBitmap(128)
	b3.Set(0)
	b3.Set(64)
	// Array of 2 uint64s.
	// b3[0] has bit 0 set -> "0000000000000001"
	// b3[1] has bit 0 (global 64) set -> "0000000000000001"
	// Output order: iterates slice.
	// "0000000000000001" + "0000000000000001"
	assert.Equal(t, "00000000000000010000000000000001", b3.String())
}
