/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package allocator

import (
	"errors"
	"math/big"
	"math/rand"
	"sync"
)

// AllocationBitmap is a contiguous block of resources that can be allocated atomically.
//
// Each resource has an offset.  The internal structure is a bitmap, with a bit for each offset.
//
// If a resource is taken, the bit at that offset is set to one.
// r.count is always equal to the number of set bits and can be recalculated at any time
// by counting the set bits in r.allocated.
//
// TODO: use RLE and compact the allocator to minimize space.
type AllocationBitmap struct {
	// strategy is the strategy for choosing the next available item out of the range
	strategy allocateStrategy
	// max is the maximum size of the usable items in the range
	max int
	// rangeSpec is the range specifier, matching RangeAllocation.Range
	rangeSpec string

	// lock guards the following members
	lock sync.Mutex
	// count is the number of currently allocated elements in the range
	count int
	// allocated is a bit array of the allocated items in the range
	allocated *big.Int
}

// AllocationBitmap implements Interface and Snapshottable
var _ Interface = &AllocationBitmap{}
var _ Snapshottable = &AllocationBitmap{}

// allocateStrategy is a search strategy in the allocation map for a valid item.
type allocateStrategy func(allocated *big.Int, max, count int) (int, bool)

func NewAllocationMap(max int, rangeSpec string) *AllocationBitmap {
	a := AllocationBitmap{
		strategy:  randomScanStrategy,
		allocated: big.NewInt(0),
		count:     0,
		max:       max,
		rangeSpec: rangeSpec,
	}
	return &a
}

// Allocate attempts to reserve the provided item.
// Returns true if it was allocated, false if it was already in use
func (r *AllocationBitmap) Allocate(offset int) (bool, error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	if r.allocated.Bit(offset) == 1 {
		return false, nil
	}
	r.allocated = r.allocated.SetBit(r.allocated, offset, 1)
	r.count++
	return true, nil
}

// AllocateNext reserves one of the items from the pool.
// (0, false, nil) may be returned if there are no items left.
func (r *AllocationBitmap) AllocateNext() (int, bool, error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	next, ok := r.strategy(r.allocated, r.max, r.count)
	if !ok {
		return 0, false, nil
	}
	r.count++
	r.allocated = r.allocated.SetBit(r.allocated, next, 1)
	return next, true, nil
}

// Release releases the item back to the pool. Releasing an
// unallocated item or an item out of the range is a no-op and
// returns no error.
func (r *AllocationBitmap) Release(offset int) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if r.allocated.Bit(offset) == 0 {
		return nil
	}

	r.allocated = r.allocated.SetBit(r.allocated, offset, 0)
	r.count--
	return nil
}

// Has returns true if the provided item is already allocated and a call
// to Allocate(offset) would fail.
func (r *AllocationBitmap) Has(offset int) bool {
	r.lock.Lock()
	defer r.lock.Unlock()

	return r.allocated.Bit(offset) == 1
}

// Free returns the count of items left in the range.
func (r *AllocationBitmap) Free() int {
	r.lock.Lock()
	defer r.lock.Unlock()
	return r.max - r.count
}

// Snapshot saves the current state of the pool.
func (r *AllocationBitmap) Snapshot() (string, []byte) {
	r.lock.Lock()
	defer r.lock.Unlock()

	return r.rangeSpec, r.allocated.Bytes()
}

// Restore restores the pool to the previously captured state.
func (r *AllocationBitmap) Restore(rangeSpec string, data []byte) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if r.rangeSpec != rangeSpec {
		return errors.New("the provided range does not match the current range")
	}

	r.allocated = big.NewInt(0).SetBytes(data)
	r.count = countBits(r.allocated)

	return nil
}

// randomScanStrategy chooses a random address from the provided big.Int, and then
// scans forward looking for the next available address (it will wrap the range if
// necessary).
func randomScanStrategy(allocated *big.Int, max, count int) (int, bool) {
	if count >= max {
		return 0, false
	}
	offset := rand.Intn(max)
	for i := 0; i < max; i++ {
		at := (offset + i) % max
		if allocated.Bit(at) == 0 {
			return at, true
		}
	}
	return 0, false
}
