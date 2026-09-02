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
	"math/rand"
	"testing"
)

func TestAllocatorRandomInputs(t *testing.T) {
	maxBytes := 5 * 1000000 // 5 MB
	iterations := rand.Intn(10000) + 10
	target := &Allocator{}

	for i := 0; i < iterations; i++ {
		bytesToAllocate := rand.Intn(maxBytes)
		buff := target.Allocate(uint64(bytesToAllocate))
		if cap(buff) < bytesToAllocate {
			t.Fatalf("expected the buffer to allocate: %v bytes whereas it allocated: %v bytes", bytesToAllocate, cap(buff))
		}
		if len(buff) != bytesToAllocate {
			t.Fatalf("unexpected length of the buffer, expected: %v, got: %v", bytesToAllocate, len(buff))
		}
	}
}

func TestAllocatorNeverShrinks(t *testing.T) {
	target := &Allocator{}
	initialSize := 1000000 // 1MB
	initialBuff := target.Allocate(uint64(initialSize))
	if cap(initialBuff) < initialSize {
		t.Fatalf("unexpected size of the buffer, expected at least 1MB, got: %v", cap(initialBuff))
	}

	for i := initialSize; i > 0; i = i / 10 {
		newBuff := target.Allocate(uint64(i))
		if cap(newBuff) < initialSize {
			t.Fatalf("allocator is now allowed to shrink memory")
		}
		if len(newBuff) != i {
			t.Fatalf("unexpected length of the buffer, expected: %v, got: %v", i, len(newBuff))
		}
	}
}

// withMaxPooledBufferCapacity sets the pool bound for the duration of a test.
func withMaxPooledBufferCapacity(t *testing.T, n int) {
	t.Helper()
	previous := MaxPooledBufferCapacity()
	SetMaxPooledBufferCapacity(n)
	t.Cleanup(func() { SetMaxPooledBufferCapacity(previous) })
}

func TestPutAllocatorDropsOversizedBuffer(t *testing.T) {
	const limit = 64 * 1024
	withMaxPooledBufferCapacity(t, limit)
	target := &Allocator{}
	target.Allocate(limit + 1)
	PutAllocator(target)
	if target.buf != nil {
		t.Fatalf("expected the oversized buffer to be dropped before pooling, got capacity: %v", cap(target.buf))
	}
}

func TestPutAllocatorKeepsSmallBuffer(t *testing.T) {
	const limit = 64 * 1024
	withMaxPooledBufferCapacity(t, limit)
	target := &Allocator{}
	target.Allocate(limit / 2)
	PutAllocator(target)
	if cap(target.buf) < limit/2 {
		t.Fatalf("expected the buffer to be retained, got capacity: %v", cap(target.buf))
	}
}

func TestPutAllocatorUnboundedByDefault(t *testing.T) {
	for _, n := range []int{0, -1} {
		withMaxPooledBufferCapacity(t, n)
		if got := MaxPooledBufferCapacity(); got != 0 {
			t.Fatalf("SetMaxPooledBufferCapacity(%d): got bound %d, want 0", n, got)
		}
		target := &Allocator{}
		target.Allocate(4 * 1024 * 1024)
		PutAllocator(target)
		if cap(target.buf) < 4*1024*1024 {
			t.Fatalf("expected an unbounded pool to retain the buffer, got capacity: %v", cap(target.buf))
		}
	}
}

func TestPutAllocatorIgnoresOtherAllocators(t *testing.T) {
	// must not panic
	PutAllocator(&SimpleAllocator{})
	PutAllocator(nil)
}

func TestAllocatorZero(t *testing.T) {
	target := &Allocator{}
	initialSize := 1000000 // 1MB
	buff := target.Allocate(uint64(initialSize))
	if cap(buff) < initialSize {
		t.Fatalf("unexpected size of the buffer, expected at least 1MB, got: %v", cap(buff))
	}
	if len(buff) != initialSize {
		t.Fatalf("unexpected length of the buffer, expected: %v, got: %v", initialSize, len(buff))
	}

	buff = target.Allocate(0)
	if cap(buff) < initialSize {
		t.Fatalf("unexpected size of the buffer, expected at least 1MB, got: %v", cap(buff))
	}
	if len(buff) != 0 {
		t.Fatalf("unexpected length of the buffer, expected: 0, got: %v", len(buff))
	}
}
