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
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"slices"
)

type bitmap []uint64

// newBitmap creates a new bitmap that can store the number of bits specified by size.
func newBitmap(size int) bitmap {
	return make(bitmap, (size+63)/64)
}

// Set sets the bit at the specified index.
func (b bitmap) Set(i int) {
	if i < 0 || i/64+1 > len(b) {
		panic(fmt.Sprintf("bitmap index out of range: %d", i))
	}

	b[i/64] |= 1 << (i % 64)
}

// Get returns the bit value at the specified index.
func (b bitmap) Get(i int) bool {
	if i < 0 || i/64+1 > len(b) {
		panic(fmt.Sprintf("bitmap index out of range: %d", i))
	}

	return b[i/64]&(1<<(i%64)) != 0
}

// Difference returns a new bitmap containing bits set in b but not in other.
func (b bitmap) Difference(other bitmap) (bitmap, error) {
	if len(b) != len(other) {
		return nil, fmt.Errorf("bitmap size mismatch: %d != %d", len(b), len(other))
	}

	diff := make(bitmap, len(b))
	for i := range b {
		diff[i] = b[i] &^ other[i]
	}
	return diff, nil
}

// Contains checks whether b is a subset of other.
func (b bitmap) IsSubset(other bitmap) (bool, error) {
	if len(b) != len(other) {
		return false, fmt.Errorf("bitmap size mismatch; %d != %d", len(b), len(other))
	}
	for i := range b {
		if b[i]&^other[i] != 0 {
			return false, nil
		}
	}
	return true, nil
}

// Equal returns true if b and other have the same bits set.
func (b bitmap) Equal(other bitmap) bool {
	if len(b) != len(other) {
		return false
	}

	for i := range b {
		if b[i] != other[i] {
			return false
		}
	}
	return true
}

// IsEmpty returns true if no bits are set in the bitmap.
func (b bitmap) IsEmpty() bool {
	for i := range b {
		if b[i] != 0 {
			return false
		}
	}
	return true
}

// Clone returns a deep copy of the bitmap.
func (b bitmap) Clone() bitmap {
	return slices.Clone(b)
}

// String returns a hex-encoded representation.
func (b bitmap) String() string {
	// Pre-allocate buffer (16 chars per uint64)
	buf := make([]byte, len(b)*16)

	var scratch [8]byte // Holds byte representation
	for i, n := range b {
		// Convert uint64 to bytes
		binary.BigEndian.PutUint64(scratch[:], n)
		// Encode bytes to 16 hex characters.
		hex.Encode(buf[i*16:], scratch[:])
	}

	return string(buf)
}
