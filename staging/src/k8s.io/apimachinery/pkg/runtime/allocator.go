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
)

// AllocatorPool simply stores Allocator objects to avoid additional memory allocations
// by caching created but unused items for later reuse, relieving pressure on the garbage collector.
//
// Usage:
//
//	memoryAllocator := runtime.AllocatorPool.Get().(*runtime.Allocator)
//	defer runtime.AllocatorPool.Put(memoryAllocator)
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
	size := uint64(2*cap(a.buf)) + n
	a.buf = make([]byte, size, size)
	a.buf = a.buf[:n]
	return a.buf
}

// SimpleAllocator a wrapper around make([]byte)
// conforms to the MemoryAllocator interface
type SimpleAllocator struct{}

var _ MemoryAllocator = &SimpleAllocator{}

func (sa *SimpleAllocator) Allocate(n uint64) []byte {
	return make([]byte, n, n)
}
