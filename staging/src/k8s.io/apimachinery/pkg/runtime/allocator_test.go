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
