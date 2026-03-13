/*
Copyright 2023 The Kubernetes Authors.

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

package synctrack

import (
	"sync"
	"sync/atomic"
)

// Lazy defers the computation of `Evaluate` to when it is necessary. It is
// possible that Evaluate will be called in parallel from multiple goroutines.
type Lazy[T any] struct {
	Evaluate func() (T, error)

	cache atomic.Pointer[cacheEntry[T]]
}

type cacheEntry[T any] struct {
	eval   func() (T, error)
	lock   sync.RWMutex
	result *T
}

func (e *cacheEntry[T]) get() (T, error) {
	if cur := func() *T {
		e.lock.RLock()
		defer e.lock.RUnlock()
		return e.result
	}(); cur != nil {
		return *cur, nil
	}

	e.lock.Lock()
	defer e.lock.Unlock()
	if e.result != nil {
		return *e.result, nil
	}
	r, err := e.eval()
	if err == nil {
		e.result = &r
	}
	return r, err
}

func (z *Lazy[T]) newCacheEntry() *cacheEntry[T] {
	return &cacheEntry[T]{eval: z.Evaluate}
}

// Notify should be called when something has changed necessitating a new call
// to Evaluate.
func (z *Lazy[T]) Notify() { z.cache.Swap(z.newCacheEntry()) }

// Get should be called to get the current result of a call to Evaluate. If the
// current cached value is stale (due to a call to Notify), then Evaluate will
// be called synchronously. If subsequent calls to Get happen (without another
// Notify), they will all wait for the same return value.
//
// Error returns are not cached and will cause multiple calls to evaluate!
func (z *Lazy[T]) Get() (T, error) {
	e := z.cache.Load()
	if e == nil {
		// Since we don't force a constructor, nil is a possible value.
		// If multiple Gets race to set this, the swap makes sure only
		// one wins.
		z.cache.CompareAndSwap(nil, z.newCacheEntry())
		e = z.cache.Load()
	}
	return e.get()
}
