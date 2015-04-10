/*
Copyright 2014 Google Inc. All rights reserved.

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

type PoolAllocator struct {
	driver PoolDriver

	lock sync.Mutex // protects 'used'

	used           map[string]bool
	randomAttempts int

	random *math_rand.Rand
}

type PoolDriver interface {
	PickRandom(seed int) string
	IterateStart() string
	IterateNext(s string) string
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
		seed := a.random.Int()

		s := a.driver.PickRandom(seed)

		inUse := a.used[s]
		if !inUse {
			a.used[s] = true
			return s
		}
	}

	// If that doesn't work, try a linear search
	s := a.driver.IterateStart()
	for s != "" {
		inUse := a.used[s]
		if !inUse {
			a.used[s] = true
			return s
		}

		s = a.driver.IterateNext(s)
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
