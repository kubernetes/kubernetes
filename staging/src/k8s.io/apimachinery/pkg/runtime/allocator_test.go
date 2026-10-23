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

// TestAllocatorGrowthPolicy verifies the growth formula in both modes across
// all orderings of (n, 2*cap, maxPool): with a pool bound configured a buffer
// grows via max(n, min(2*cap, maxPool)); without one the legacy 2*cap+n
// formula is preserved.
func TestAllocatorGrowthPolicy(t *testing.T) {
	for _, tc := range []struct {
		name     string
		bound    int
		requests []uint64
		wantCaps []int
	}{
		{
			// Covers: 0<n<=maxPool (initial), cap<n<2*cap<=maxPool (2*cap wins),
			// n<=cap (reuse), cap<2*cap<=n<=maxPool (n wins within maxPool).
			name:     "bounded: growth within maxPool (2*cap wins vs n wins)",
			bound:    1000,
			requests: []uint64{100, 130, 200, 50, 500},
			wantCaps: []int{100, 200, 200, 200, 500},
		},
		{
			// Covers: cap<n<=maxPool<2*cap (maxPool clamps doubling),
			// cap==maxPool<n<2*cap (n wins), n<=cap above maxPool (reuse),
			// maxPool<cap<n<2*cap (exact-n growth above maxPool, no doubling),
			// maxPool<cap<2*cap<=n (exact-n growth above maxPool).
			name:     "bounded: doubling clamped at maxPool, then exact-n growth above maxPool",
			bound:    300,
			requests: []uint64{200, 260, 350, 300, 400, 900},
			wantCaps: []int{200, 300, 350, 350, 400, 900},
		},
		{
			// Covers: cap<maxPool<n<2*cap (2*cap clamped to maxPool, so n wins over 2*cap).
			name:     "bounded: cap < maxPool < n < 2*cap allocates n rather than 2*cap",
			bound:    300,
			requests: []uint64{200, 350},
			wantCaps: []int{200, 350},
		},
		{
			// Covers: cap<2*cap<=maxPool<n (n jumps past both 2*cap and maxPool).
			name:     "bounded: cap < 2*cap <= maxPool < n allocates n",
			bound:    300,
			requests: []uint64{100, 350},
			wantCaps: []int{100, 350},
		},
		{
			// Covers: cap<=maxPool<2*cap<=n (both 2*cap and n exceed maxPool, n >= 2*cap).
			name:     "bounded: cap <= maxPool < 2*cap <= n allocates n",
			bound:    150,
			requests: []uint64{100, 250},
			wantCaps: []int{100, 250},
		},
		{
			// Covers: 0=cap=2*cap<maxPool<n (very first allocation already exceeds maxPool).
			name:     "bounded: initial allocation already exceeds maxPool",
			bound:    300,
			requests: []uint64{350},
			wantCaps: []int{350},
		},
		{
			// Covers: unbounded (maxPool==0) preserves legacy 2*cap+n growth.
			name:     "unbounded: preserves legacy 2*cap+n growth",
			bound:    0,
			requests: []uint64{100, 130, 50, 330, 331},
			wantCaps: []int{100, 330, 330, 330, 991},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withMaxPooledBufferCapacity(t, tc.bound)
			target := &Allocator{}
			for i, n := range tc.requests {
				buf := target.Allocate(n)
				if uint64(len(buf)) != n {
					t.Fatalf("request %d: got len %d, want %d", i, len(buf), n)
				}
				if cap(buf) != tc.wantCaps[i] {
					t.Fatalf("request %d (n=%d): got cap %d, want %d", i, n, cap(buf), tc.wantCaps[i])
				}
			}
		})
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
