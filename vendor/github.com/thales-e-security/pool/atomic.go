/*
Copyright 2017 Google Inc.

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
	"sync"
	"sync/atomic"
	"time"
)

// AtomicInt32 is a wrapper with a simpler interface around atomic.(Add|Store|Load|CompareAndSwap)Int32 functions.
type AtomicInt32 struct {
	int32
}

// NewAtomicInt32 initializes a new AtomicInt32 with a given value.
func NewAtomicInt32(n int32) AtomicInt32 {
	return AtomicInt32{n}
}

// Add atomically adds n to the value.
func (i *AtomicInt32) Add(n int32) int32 {
	return atomic.AddInt32(&i.int32, n)
}

// Set atomically sets n as new value.
func (i *AtomicInt32) Set(n int32) {
	atomic.StoreInt32(&i.int32, n)
}

// Get atomically returns the current value.
func (i *AtomicInt32) Get() int32 {
	return atomic.LoadInt32(&i.int32)
}

// CompareAndSwap atomatically swaps the old with the new value.
func (i *AtomicInt32) CompareAndSwap(oldval, newval int32) (swapped bool) {
	return atomic.CompareAndSwapInt32(&i.int32, oldval, newval)
}

// AtomicInt64 is a wrapper with a simpler interface around atomic.(Add|Store|Load|CompareAndSwap)Int64 functions.
type AtomicInt64 struct {
	int64
}

// NewAtomicInt64 initializes a new AtomicInt64 with a given value.
func NewAtomicInt64(n int64) AtomicInt64 {
	return AtomicInt64{n}
}

// Add atomically adds n to the value.
func (i *AtomicInt64) Add(n int64) int64 {
	return atomic.AddInt64(&i.int64, n)
}

// Set atomically sets n as new value.
func (i *AtomicInt64) Set(n int64) {
	atomic.StoreInt64(&i.int64, n)
}

// Get atomically returns the current value.
func (i *AtomicInt64) Get() int64 {
	return atomic.LoadInt64(&i.int64)
}

// CompareAndSwap atomatically swaps the old with the new value.
func (i *AtomicInt64) CompareAndSwap(oldval, newval int64) (swapped bool) {
	return atomic.CompareAndSwapInt64(&i.int64, oldval, newval)
}

// AtomicDuration is a wrapper with a simpler interface around atomic.(Add|Store|Load|CompareAndSwap)Int64 functions.
type AtomicDuration struct {
	int64
}

// NewAtomicDuration initializes a new AtomicDuration with a given value.
func NewAtomicDuration(duration time.Duration) AtomicDuration {
	return AtomicDuration{int64(duration)}
}

// Add atomically adds duration to the value.
func (d *AtomicDuration) Add(duration time.Duration) time.Duration {
	return time.Duration(atomic.AddInt64(&d.int64, int64(duration)))
}

// Set atomically sets duration as new value.
func (d *AtomicDuration) Set(duration time.Duration) {
	atomic.StoreInt64(&d.int64, int64(duration))
}

// Get atomically returns the current value.
func (d *AtomicDuration) Get() time.Duration {
	return time.Duration(atomic.LoadInt64(&d.int64))
}

// CompareAndSwap atomatically swaps the old with the new value.
func (d *AtomicDuration) CompareAndSwap(oldval, newval time.Duration) (swapped bool) {
	return atomic.CompareAndSwapInt64(&d.int64, int64(oldval), int64(newval))
}

// AtomicBool gives an atomic boolean variable.
type AtomicBool struct {
	int32
}

// NewAtomicBool initializes a new AtomicBool with a given value.
func NewAtomicBool(n bool) AtomicBool {
	if n {
		return AtomicBool{1}
	}
	return AtomicBool{0}
}

// Set atomically sets n as new value.
func (i *AtomicBool) Set(n bool) {
	if n {
		atomic.StoreInt32(&i.int32, 1)
	} else {
		atomic.StoreInt32(&i.int32, 0)
	}
}

// Get atomically returns the current value.
func (i *AtomicBool) Get() bool {
	return atomic.LoadInt32(&i.int32) != 0
}

// CompareAndSwap atomatically swaps the old with the new value.
func (i *AtomicBool) CompareAndSwap(o, n bool) bool {
	var old, new int32
	if o {
		old = 1
	}
	if n {
		new = 1
	}
	return atomic.CompareAndSwapInt32(&i.int32, old, new)
}

// AtomicString gives you atomic-style APIs for string, but
// it's only a convenience wrapper that uses a mutex. So, it's
// not as efficient as the rest of the atomic types.
type AtomicString struct {
	mu  sync.Mutex
	str string
}

// Set atomically sets str as new value.
func (s *AtomicString) Set(str string) {
	s.mu.Lock()
	s.str = str
	s.mu.Unlock()
}

// Get atomically returns the current value.
func (s *AtomicString) Get() string {
	s.mu.Lock()
	str := s.str
	s.mu.Unlock()
	return str
}

// CompareAndSwap atomatically swaps the old with the new value.
func (s *AtomicString) CompareAndSwap(oldval, newval string) (swqpped bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.str == oldval {
		s.str = newval
		return true
	}
	return false
}
