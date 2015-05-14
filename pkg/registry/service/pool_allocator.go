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

package service

import (
	"fmt"
	math_rand "math/rand"
	"sync"

	"os"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"strings"
)

// A PoolAllocator is a helper interface that supports a pool of resources that can be allocated
type PoolAllocator interface {
	// Allocate allocates a specific entry.
	Allocate(s string) error

	// AllocateNext allocates and returns a new entry.
	AllocateNext() (string, error)

	// Release de-allocates an entry.
	Release(s string) (bool, error)
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

// MemoryPoolAllocator is PoolAllocator that is backed by an in-memory collection,
// can't be used if there are multiple processes allocating from the pool (cf EtcdPoolAllocator)
type MemoryPoolAllocator struct {
	driver PoolDriver

	lock sync.Mutex // protects 'used' and 'random'

	used           map[string]bool
	randomAttempts int

	random *math_rand.Rand
}

// EtcdPoolAllocator is PoolAllocator that is backed by etcd, so can be used by multiple processes
type EtcdPoolAllocator struct {
	driver PoolDriver

	lock sync.Mutex // protects 'used' and 'random'

	usedCached     map[string]bool
	randomAttempts int

	random *math_rand.Rand

	etcd    *tools.EtcdHelper
	baseKey string
}

// Init initializes a MemoryPoolAllocator.
func (a *MemoryPoolAllocator) Init(driver PoolDriver) {
	seed := time.Now().UTC().UnixNano()
	a.random = math_rand.New(math_rand.NewSource(seed))

	a.randomAttempts = 1000
	a.used = make(map[string]bool)

	a.driver = driver
}

// Init initializes a EtcdPoolAllocator.
func (a *EtcdPoolAllocator) Init(driver PoolDriver, etcd *tools.EtcdHelper, baseKey string) {
	seed := time.Now().UTC().UnixNano()
	seed ^= int64(os.Getpid())
	a.random = math_rand.New(math_rand.NewSource(seed))

	a.randomAttempts = 1000
	a.usedCached = make(map[string]bool)

	a.driver = driver

	a.etcd = etcd
	if !strings.HasSuffix(baseKey, "/") {
		baseKey += "/"
	}
	a.baseKey = baseKey
}

// Allocate allocates a specific entry.
func (a *MemoryPoolAllocator) Allocate(s string) error {
	a.lock.Lock()
	defer a.lock.Unlock()

	inUse := a.used[s]
	if inUse {
		return fmt.Errorf("Entry %s is already allocated", s)
	}
	a.used[s] = true

	return nil
}

// Try to lock the resource identified by s.
// Checks to see if we have a cached allocation first if useCache is true.
// Always updates the cache when it learns information.
func (a *EtcdPoolAllocator) tryLock(s string, useCache bool) (bool, error) {
	if useCache {
		a.lock.Lock()
		usedCached := a.usedCached[s]
		a.lock.Unlock()

		if usedCached {
			return false, nil
		}
	}

	key := a.baseKey + s
	value := "1"

	_, err := a.etcd.Client.Create(key, value, 0)

	created := true
	if err != nil {
		if tools.IsEtcdNodeExist(err) {
			// We failed to obtain the lock
			created = false
		} else {
			return false, fmt.Errorf("Error communicating with etcd: %v", err)
		}
	}

	// Whether or not we locked it, it is locked
	a.lock.Lock()
	a.usedCached[s] = true
	a.lock.Unlock()

	return created, nil
}

// Allocate allocates a specific entry.
func (a *EtcdPoolAllocator) Allocate(s string) error {
	useCache := false
	locked, err := a.tryLock(s, useCache)
	if err != nil {
		return err
	}
	if !locked {
		return fmt.Errorf("Entry %s is already allocated", s)
	}
	return nil
}

// AllocateNext allocates and returns a new entry.
func (a *MemoryPoolAllocator) AllocateNext() (string, error) {
	a.lock.Lock()
	defer a.lock.Unlock()

	// Try randomly first
	for i := 0; i < a.randomAttempts; i++ {
		s := a.driver.PickRandom(a.random)

		inUse := a.used[s]
		if !inUse {
			a.used[s] = true
			return s, nil
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
			return s, nil
		}
	}
	return "", nil
}

// AllocateNext allocates and returns a new entry.
func (a *EtcdPoolAllocator) AllocateNext() (string, error) {
	// Try randomly first
	for _, useCache := range []bool{true, false} {
		for i := 0; i < a.randomAttempts; i++ {
			// random is not documented to be thread-safe
			a.lock.Lock()
			s := a.driver.PickRandom(a.random)
			a.lock.Unlock()

			locked, err := a.tryLock(s, useCache)
			if err != nil {
				return "", err
			}
			if locked {
				return s, nil
			}
		}
	}

	// If that doesn't work, try a linear search
	useCache := false
	iterator := a.driver.Iterate()
	for {
		s, found := iterator.Next()
		if !found {
			break
		}
		locked, err := a.tryLock(s, useCache)
		if err != nil {
			return "", err
		}
		if locked {
			return s, nil
		}
	}
	return "", nil
}

// Release de-allocates an entry.
func (a *MemoryPoolAllocator) Release(s string) (bool, error) {
	a.lock.Lock()
	defer a.lock.Unlock()

	inUse, found := a.used[s]
	if !found {
		return false, nil
	}
	delete(a.used, s)
	return inUse, nil
}

// Release de-allocates an entry.
func (a *EtcdPoolAllocator) Release(s string) (bool, error) {
	key := a.baseKey + s

	recursive := false
	_, err := a.etcd.Client.Delete(key, recursive)

	deleted := true
	if err != nil {
		if tools.IsEtcdNotFound(err) {
			deleted = false
		} else {
			return false, fmt.Errorf("Error communicating with etcd: %v", err)
		}
	}

	// Whether or not we deleted it, it is deleted
	a.lock.Lock()
	a.usedCached[s] = false
	a.lock.Unlock()

	return deleted, nil
}

// Count the number of items allocated.  Intended for tests, because an etcd implementation is likely to be slow.
func (a *MemoryPoolAllocator) size() int {
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
func (a *MemoryPoolAllocator) isAllocated(s string) bool {
	a.lock.Lock()
	defer a.lock.Unlock()

	return a.used[s]
}
