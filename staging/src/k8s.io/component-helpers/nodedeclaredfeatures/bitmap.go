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
	"slices"
)

type bitmap struct {
	words []uint64
	size  int
}

// newBitmap creates a new bitmap that can store the number of bits specified by size.
func newBitmap(size int) bitmap {
	if size < 0 {
		panic(fmt.Sprintf("bitmap size must be positive: %d", size))
	}
	return bitmap{make([]uint64, (size+63)/64), size}
}

// Set sets the bit at the specified index.
func (b bitmap) Set(i int) {
	if i < 0 || i >= b.size {
		panic(fmt.Sprintf("bitmap index out of range: %d", i))
	}

	b.words[i/64] |= 1 << (i % 64)
}

// Get returns the bit value at the specified index.
func (b bitmap) Get(i int) bool {
	if i < 0 || i >= b.size {
		panic(fmt.Sprintf("bitmap index out of range: %d", i))
	}

	return b.words[i/64]&(1<<(i%64)) != 0
}

// Difference returns a new bitmap containing bits set in b but not in other.
func (b bitmap) Difference(other bitmap) (bitmap, error) {
	if b.size != other.size {
		return bitmap{}, fmt.Errorf("bitmap size mismatch: %d != %d", b.size, other.size)
	}

	diff := newBitmap(b.size)
	for i := range b.words {
		diff.words[i] = b.words[i] &^ other.words[i]
	}
	return diff, nil
}

// Contains checks whether b is a subset of other.
func (b bitmap) IsSubset(other bitmap) (bool, error) {
	if b.size != other.size {
		return false, fmt.Errorf("bitmap size mismatch; %d != %d", b.size, other.size)
	}
	for i := range b.words {
		if b.words[i]&^other.words[i] != 0 {
			return false, nil
		}
	}
	return true, nil
}

// Equal returns true if b and other have the same bits set.
func (b bitmap) Equal(other bitmap) bool {
	if b.size != other.size {
		return false
	}

	for i := range b.words {
		if b.words[i] != other.words[i] {
			return false
		}
	}
	return true
}

// IsEmpty returns true if no bits are set in the bitmap.
func (b bitmap) IsEmpty() bool {
	for i := range b.words {
		if b.words[i] != 0 {
			return false
		}
	}
	return true
}

// Clone returns a deep copy of the bitmap.
func (b bitmap) Clone() bitmap {
	return bitmap{
		words: slices.Clone(b.words),
		size:  b.size,
	}
}

// String returns a binary representation, with leftmost character representing
// the first (0th) bit.
func (b bitmap) String() string {
	buf := make([]byte, b.size)
	for i := 0; i < b.size; i++ {
		if (b.words[i/64]>>(uint(i)%64))&1 != 0 {
			buf[i] = '1'
		} else {
			buf[i] = '0'
		}
	}
	return string(buf)
}
