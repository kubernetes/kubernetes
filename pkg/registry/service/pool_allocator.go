/*
Copyright 2015 Google Inc. All rights reserved.

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

package service

import (
	"fmt"
	math_rand "math/rand"
	"sync"

	"time"
)

// A PoolAllocator is a helper-class that supports a pool of resources that can be allocated
type PoolAllocator struct {
	driver PoolDriver

	lock sync.Mutex // protects 'used' and 'random'

	used           map[string]bool
	randomAttempts int

	random *math_rand.Rand
}

// A very simple string iterator
type StringIterator interface {
	Next() (string, bool)
}

// A PoolDriver provides the PoolAllocator with the set from which it allocates
type PoolDriver interface {
	// Choose an item from the set
	PickRandom(random *math_rand.Rand) string

	// Iterate across all items in the set
	Iterate() StringIterator
}

// Init initializes a new PoolAllocator driver.
func (a *PoolAllocator) Init(driver PoolDriver) {
	seed := time.Now().UTC().UnixNano()
	a.random = math_rand.New(math_rand.NewSource(seed))

	a.randomAttempts = 1000
	a.used = make(map[string]bool)

	a.driver = driver
}

// Allocate allocates a specific entry.  This is useful when recovering saved state.
func (a *PoolAllocator) Allocate(s string) error {
	a.lock.Lock()
	defer a.lock.Unlock()

	inUse := a.used[s]
	if inUse {
		return fmt.Errorf("Entry %s is already allocated", s)
	}
	a.used[s] = true

	return nil
}

// AllocateNext allocates and returns a new entry.
func (a *PoolAllocator) AllocateNext() string {
	a.lock.Lock()
	defer a.lock.Unlock()

	// Try randomly first
	for i := 0; i < a.randomAttempts; i++ {
		s := a.driver.PickRandom(a.random)

		inUse := a.used[s]
		if !inUse {
			a.used[s] = true
			return s
		}
	}

	// If that doesn't work, try a linear search
	iterator := a.driver.Iterate()
	for {
		s, found := iterator.Next()
		if !found {
			break
		}
		inUse := a.used[s]
		if !inUse {
			a.used[s] = true
			return s
		}
	}
	return ""
}

// Release de-allocates an entry.
func (a *PoolAllocator) Release(s string) bool {
	a.lock.Lock()
	defer a.lock.Unlock()

	inUse, found := a.used[s]
	if !found {
		return false
	}
	delete(a.used, s)
	return inUse
}

// Count the number of items allocated.  Intended for tests, because an etcd implementation is likely to be slow.
func (a *PoolAllocator) size() int {
	a.lock.Lock()
	defer a.lock.Unlock()

	n := 0
	for _, used := range a.used {
		if used {
			n++
		}
	}
	return n
}

// Checks if an entry is allocated.  Used by tests.
func (a *PoolAllocator) isAllocated(s string) bool {
	a.lock.Lock()
	defer a.lock.Unlock()

	return a.used[s]
}
