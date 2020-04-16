/*
Copyright 2015 The Kubernetes Authors.

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
	"bytes"
	"encoding/gob"
	"errors"
	"math/rand"
	"sync"
	"time"
)

// AllocationMap is a map of resources that can be allocated atomically.
//
// Each resource has an offset.  The internal structure is a map, with a key for each offset.
//
type AllocationMap struct {
	// strategy carries the details of how to choose the next available item out of the range
	strategy keyAllocator
	// max is the maximum size of the usable items in the range
	max int
	// rangeSpec is the range specifier, matching RangeAllocation.Range
	rangeSpec string

	// lock guards the following members
	lock sync.Mutex
	// allocated is a map of the allocated items in the range
	allocated map[uint64]bool
}

// AllocationMap implements Interface and Snapshottable
var _ Interface = &AllocationMap{}
var _ Snapshottable = &AllocationMap{}

// keyAllocator represents a search strategy in the allocation map for a valid item.
type keyAllocator interface {
	AllocateKey(allocated map[uint64]bool, max int) (int, bool)
}

// NewAllocation creates an allocation bitmap using the random scan strategy.
func NewAllocation(max int, rangeSpec string) *AllocationMap {
	a := AllocationMap{
		strategy: randomMapScanStrategy{
			rand: rand.New(rand.NewSource(time.Now().UnixNano())),
		},
		allocated: make(map[uint64]bool),
		max:       max,
		rangeSpec: rangeSpec,
	}
	return &a
}

// Allocate attempts to reserve the provided item.
// Returns true if it was allocated, false if it was already in use
func (r *AllocationMap) Allocate(offset int) (bool, error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	if _, ok := r.allocated[uint64(offset)]; ok {
		return false, nil
	}
	r.allocated[uint64(offset)] = true
	return true, nil
}

// AllocateNext reserves one of the items from the pool.
// (0, false, nil) may be returned if there are no items left.
func (r *AllocationMap) AllocateNext() (int, bool, error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	next, ok := r.strategy.AllocateKey(r.allocated, r.max)
	if !ok {
		return 0, false, nil
	}
	r.allocated[uint64(next)] = true
	return next, true, nil
}

// Release releases the item back to the pool. Releasing an
// unallocated item or an item out of the range is a no-op and
// returns no error.
func (r *AllocationMap) Release(offset int) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if _, ok := r.allocated[uint64(offset)]; ok {
		delete(r.allocated, uint64(offset))
	}
	return nil
}

// ForEach calls the provided function for each allocated bit.  The
// AllocationMap may not be modified while this loop is running.
func (r *AllocationMap) ForEach(fn func(int)) {
	r.lock.Lock()
	defer r.lock.Unlock()

	for k := range r.allocated {
		fn(int(k))
	}
}

// Has returns true if the provided item is already allocated and a call
// to Allocate(offset) would fail.
func (r *AllocationMap) Has(offset int) bool {
	r.lock.Lock()
	defer r.lock.Unlock()

	_, ok := r.allocated[uint64(offset)]
	return ok
}

// Free returns the count of items left in the range.
func (r *AllocationMap) Free() int {
	r.lock.Lock()
	defer r.lock.Unlock()
	return r.max - len(r.allocated)
}

// Snapshot saves the current state of the pool.
func (r *AllocationMap) Snapshot() (string, []byte) {
	r.lock.Lock()
	defer r.lock.Unlock()
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(r.allocated); err != nil {
		return "", nil
	}
	return r.rangeSpec, buf.Bytes()
}

// Restore restores the pool to the previously captured state.
func (r *AllocationMap) Restore(rangeSpec string, data []byte) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if r.rangeSpec != rangeSpec {
		return errors.New("the provided range does not match the current range")
	}

	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)

	if err := dec.Decode(&r.allocated); err != nil {
		return err
	}

	return nil
}

// randomScanStrategy chooses a random address from the provided big.Int, and then
// scans forward looking for the next available address (it will wrap the range if
// necessary).
type randomMapScanStrategy struct {
	rand *rand.Rand
}

func (rss randomMapScanStrategy) AllocateKey(allocated map[uint64]bool, max int) (int, bool) {
	if len(allocated) >= max {
		return 0, false
	}
	offset := rss.rand.Intn(max)
	for i := 0; i < max; i++ {
		at := (offset + i) % max
		if _, ok := allocated[uint64(at)]; !ok {
			return at, true
		}
	}
	return 0, false
}

var _ keyAllocator = randomMapScanStrategy{}
