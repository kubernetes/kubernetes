/*
Copyright 2022 The Kubernetes Authors.

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

package runtime

import (
	"sync"
	"sync/atomic"
)

// AllocatorPool simply stores Allocator objects to avoid additional memory allocations
// by caching created but unused items for later reuse, relieving pressure on the garbage collector.
//
// Usage:
//
//	memoryAllocator := runtime.AllocatorPool.Get().(*runtime.Allocator)
//	defer runtime.PutAllocator(memoryAllocator)
//
// A note for future:
//
//	consider introducing multiple pools for storing buffers of different sizes
//	perhaps this could allow us to be more efficient.
var AllocatorPool = sync.Pool{
	New: func() interface{} {
		return &Allocator{}
	},
}

// maxPooledBufferCapacity bounds the buffer capacity an Allocator may retain
// while parked in AllocatorPool; zero means unbounded. An Allocator's buffer
// grows to the largest object it has ever encoded and never shrinks, so
// without a bound a burst of large responses leaves every pooled allocator
// holding a multi-megabyte buffer for the lifetime of the process. Encodes
// larger than the bound still work; their buffers are simply released to the
// GC instead of being pooled.
//
// The bound is process-wide and unbounded by default so that library
// consumers see no behavior change; servers configure it at startup with
// SetMaxPooledBufferCapacity.
var maxPooledBufferCapacity atomic.Int64

// SetMaxPooledBufferCapacity sets the largest buffer capacity, in bytes, that
// PutAllocator returns to AllocatorPool. Zero (the default) or a negative
// value disables the bound. It is safe to call concurrently with PutAllocator;
// it is intended to be called once at process startup.
func SetMaxPooledBufferCapacity(n int) {
	if n < 0 {
		n = 0
	}
	maxPooledBufferCapacity.Store(int64(n))
}

// MaxPooledBufferCapacity returns the bound set by SetMaxPooledBufferCapacity,
// or zero if the pool is unbounded.
func MaxPooledBufferCapacity() int {
	return int(maxPooledBufferCapacity.Load())
}

// PutAllocator returns a MemoryAllocator previously obtained from
// AllocatorPool. When a bound is configured with SetMaxPooledBufferCapacity,
// buffers that grew beyond it are dropped before pooling so that steady-state
// memory retained by the pool stays bounded. Allocators of other types are
// ignored.
func PutAllocator(m MemoryAllocator) {
	a, ok := m.(*Allocator)
	if !ok {
		return
	}
	if limit := maxPooledBufferCapacity.Load(); limit > 0 && int64(cap(a.buf)) > limit {
		a.buf = nil
	}
	AllocatorPool.Put(a)
}

// Allocator knows how to allocate memory
// It exists to make the cost of object serialization cheaper.
// In some cases, it allows for allocating memory only once and then reusing it.
// This approach puts less load on GC and leads to less fragmented memory in general.
type Allocator struct {
	buf []byte
}

var _ MemoryAllocator = &Allocator{}

// Allocate reserves memory for n bytes only if the underlying array doesn't have enough capacity
// otherwise it returns previously allocated block of memory.
//
// Note that the returned array is not zeroed, it is the caller's
// responsibility to clean the memory if needed.
func (a *Allocator) Allocate(n uint64) []byte {
	if uint64(cap(a.buf)) >= n {
		a.buf = a.buf[:n]
		return a.buf
	}
	// grow the buffer
	var size uint64
	if maxPooledBufferCapacity.Load() > 0 {
		// With a pool bound configured, grow geometrically but never past
		// what the request or plain doubling needs. Callers pass exact sizes
		// (the protobuf encoder sizes the object first), so the historical
		// 2*cap+n formula over-allocates by up to 3x the object size, and the
		// excess is what the pool then retains.
		size = max(n, uint64(2*cap(a.buf)))
	} else {
		size = uint64(2*cap(a.buf)) + n
	}
	a.buf = make([]byte, size)
	a.buf = a.buf[:n]
	return a.buf
}

// SimpleAllocator a wrapper around make([]byte)
// conforms to the MemoryAllocator interface
type SimpleAllocator struct{}

var _ MemoryAllocator = &SimpleAllocator{}

func (sa *SimpleAllocator) Allocate(n uint64) []byte {
	return make([]byte, n)
}
