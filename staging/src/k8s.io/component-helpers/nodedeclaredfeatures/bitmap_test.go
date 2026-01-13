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
	"fmt"
	"testing"
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
			if len(b) != tt.expected {
				t.Errorf("len(b) = %d, want %d", len(b), tt.expected)
			}
		})
	}
}

func TestBitmap_GetSet(t *testing.T) {
	b := newBitmap(128)
	indices := []int{0, 63, 64, 127}

	// Initially all false
	for _, i := range indices {
		if b.Get(i) {
			t.Errorf("expected bit %d to be false initially", i)
		}
	}

	// Set bits
	for _, i := range indices {
		b.Set(i)
	}

	for _, i := range indices {
		if !b.Get(i) {
			t.Errorf("expected bit %d to be true after Set", i)
		}
	}

	// Verify other bits are still false
	for _, i := range []int{1, 62, 65} {
		if b.Get(i) {
			t.Errorf("expected bit %d to be false", i)
		}
	}

	// Error bounds.
	for _, i := range []int{-1, 128} {
		msg := fmt.Sprintf("bitmap index out of range: %d", i)
		t.Run(fmt.Sprintf("Panic_Set_%d", i), func(t *testing.T) {
			defer expectPanic(t, msg)
			b.Set(i)
		})
		t.Run(fmt.Sprintf("Panic_Get_%d", i), func(t *testing.T) {
			defer expectPanic(t, msg)
			b.Get(i)
		})
	}
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
				if err == nil {
					t.Error("expected error, got nil")
				}
				if diff != nil {
					t.Errorf("expected nil diff, got %v", diff)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				for _, i := range tt.expectedDiff {
					if !diff.Get(i) {
						t.Errorf("Difference: Index %d should be set", i)
					}
				}
				// Verify count of set bits for Difference
				count := 0
				for i := 0; i < tt.bSize; i++ {
					if diff.Get(i) {
						count++
					}
				}
				if count != len(tt.expectedDiff) {
					t.Errorf("Difference: Unexpected number of bits set, got %d, want %d", count, len(tt.expectedDiff))
				}
			}

			// Test IsSubset
			result, err := b.IsSubset(other)
			if tt.expectedErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				if result {
					t.Error("expected result to be false on error")
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				expectedResult := len(tt.expectedDiff) == 0
				if result != expectedResult {
					t.Errorf("IsSubset result mismatch: got %v, want %v", result, expectedResult)
				}
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

	if !b1.Equal(b2) {
		t.Error("expected b1 to equal b2")
	}
	if b1.Equal(b3) {
		t.Error("expected b1 not to equal b3")
	}
	if b1.Equal(b4) {
		t.Error("expected b1 not to equal b4")
	}
}

func TestBitmap_IsEmpty(t *testing.T) {
	b := newBitmap(64)
	if !b.IsEmpty() {
		t.Error("expected b to be empty")
	}

	b.Set(10)
	if b.IsEmpty() {
		t.Error("expected b not to be empty")
	}
}

func TestBitmap_Clone(t *testing.T) {
	b := newBitmap(64)
	b.Set(1)

	clone := b.Clone()
	if !b.Equal(clone) {
		t.Error("expected clone to equal original")
	}

	// Modify clone, original should not change
	clone.Set(2)
	if clone.Equal(b) {
		t.Error("expected modified clone not to equal original")
	}
	if b.Get(2) {
		t.Error("expected original not to have bit 2 set")
	}
	if !clone.Get(2) {
		t.Error("expected clone to have bit 2 set")
	}
}

func TestBitmap_String(t *testing.T) {
	// Size 64 -> 1 uint64
	b := newBitmap(64)
	// 0x0000000000000001 (bit 0 set)
	b.Set(0)
	// Output is hex encoded BigEndian of the uint64
	// uint64(1) -> bytes: [0 0 0 0 0 0 0 1]
	// hex: "0000000000000001"
	if got, want := b.String(), "0000000000000001"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	b2 := newBitmap(64)
	b2.Set(63)
	// 1 << 63 is the highest bit.
	// 0x8000000000000000
	// bytes: [128 0 0 0 0 0 0 0] -> hex "8000000000000000"
	if got, want := b2.String(), "8000000000000000"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// Multi-word
	b3 := newBitmap(128)
	b3.Set(0)
	b3.Set(64)
	// Array of 2 uint64s.
	// b3[0] has bit 0 set -> "0000000000000001"
	// b3[1] has bit 0 (global 64) set -> "0000000000000001"
	// Output order: iterates slice.
	// "0000000000000001" + "0000000000000001"
	if got, want := b3.String(), "00000000000000010000000000000001"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func expectPanic(t *testing.T, expected interface{}) {
	r := recover()
	if r == nil {
		t.Errorf("expected panic %v, but did not panic", expected)
	} else if r != expected {
		t.Errorf("expected panic %v, but got %v", expected, r)
	}
}
