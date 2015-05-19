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

package pool

import (
	mathrand "math/rand"
	"sync"

	"time"
)

// MemoryPoolAllocator is PoolAllocator that is backed by an in-memory collection,
// can't be used if there are multiple processes allocating from the pool (cf EtcdPoolAllocator)
type MemoryPoolAllocator struct {
	driver         PoolDriver
	randomAttempts int

	lock   sync.Mutex // protects 'used' and 'random'
	used   map[string]Allocation
	random *mathrand.Rand
}

// For tests
func (a *MemoryPoolAllocator) DisableRandomAllocation() {
	a.randomAttempts = 0
}

// Init initializes a MemoryPoolAllocator.
func (a *MemoryPoolAllocator) Init(driver PoolDriver) {
	seed := time.Now().UTC().UnixNano()
	a.random = mathrand.New(mathrand.NewSource(seed))

	a.randomAttempts = 1000
	a.used = make(map[string]Allocation)

	a.driver = driver
}

// Allocate allocates a specific entry.
func (a *MemoryPoolAllocator) Allocate(key string, owner string) (bool, error) {
	a.lock.Lock()
	defer a.lock.Unlock()

	allocation, found := a.used[key]
	if found {
		return false, nil
	}
	allocation.Owner = owner
	allocation.Key = key
	a.used[key] = allocation
	return true, nil
}

// AllocateNext allocates and returns a new entry.
func (a *MemoryPoolAllocator) AllocateNext(owner string) (string, error) {
	a.lock.Lock()
	defer a.lock.Unlock()

	// Try randomly first
	for i := 0; i < a.randomAttempts; i++ {
		s := a.driver.PickRandom(a.random)

		allocation, found := a.used[s]
		if !found {
			allocation.Owner = owner
			allocation.Key = s
			a.used[s] = allocation
			return s, nil
		}
	}

	// If that doesn't work, try a linear search
	iterator := a.driver.Iterator()
	for {
		s, found := iterator.Next()
		if !found {
			break
		}
		allocation, found := a.used[s]
		if !found {
			allocation.Owner = owner
			allocation.Key = s
			a.used[s] = allocation
			return s, nil
		}
	}
	return "", nil
}

// Release de-allocates an entry.
func (a *MemoryPoolAllocator) Release(key string) (bool, error) {
	a.lock.Lock()
	defer a.lock.Unlock()

	_, found := a.used[key]
	if !found {
		return false, nil
	}
	delete(a.used, key)
	return true, nil
}

// Count the number of items allocated.  Intended for tests, because an etcd implementation is likely to be slow.
func (a *MemoryPoolAllocator) size() int {
	a.lock.Lock()
	defer a.lock.Unlock()

	n := len(a.used)
	return n
}

// Checks if an entry is allocated.
func (a *MemoryPoolAllocator) ReadAllocation(s string) (*Allocation, uint64, error) {
	a.lock.Lock()
	defer a.lock.Unlock()

	allocation, found := a.used[s]
	if !found {
		return nil, 0, nil
	}

	return &allocation, 0, nil
}

// Release an entry, if still current.
func (a *MemoryPoolAllocator) ReleaseForRepair(allocation Allocation) (bool, error) {
	// We shouldn't need a repair for an in-memory implementation
	panic("Repair functions not implemented for MemoryPoolAllocator")
}

// List all allocations
func (a *MemoryPoolAllocator) ListAllocations() ([]Allocation, error) {
	panic("Not implemented")
}
