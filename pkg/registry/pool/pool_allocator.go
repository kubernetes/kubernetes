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
)

// A PoolAllocator is a helper interface that supports a pool of resources that can be allocated
// TODO: Should an allocation with the same owner succeed?  Probably yes...
type PoolAllocator interface {
	// Allocate allocates a specific entry.
	Allocate(key string, owner string) (bool, error)

	// AllocateNext allocates and returns a new entry.
	AllocateNext(owner string) (string, error)

	// Release de-allocates an entry.
	// TODO: We could take an owner, for safety
	Release(key string) (bool, error)

	// List all allocations, for repair.
	ListAllocations() ([]Allocation, error)

	// Checks if key is locked, for repair.
	ReadAllocation(key string) (*Allocation, uint64, error)

	// Releases allocation, if still current, for repair.
	ReleaseForRepair(allocation Allocation) (bool, error)

	// For tests
	DisableRandomAllocation()
}

// Iterator over the set of items that can be members of the pool
type PoolIterator interface {
	Next() (string, bool)
}

type Allocation struct {
	Key     string
	Owner   string
	Version uint64
}

// A PoolDriver provides the PoolAllocator with the set from which it allocates
type PoolDriver interface {
	// Choose an item from the set
	PickRandom(random *mathrand.Rand) string

	// Iterate across all items in the set
	Iterator() PoolIterator
}
