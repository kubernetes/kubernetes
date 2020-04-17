/*
Copyright 2015 The Kubernetes Authors.

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

package allocator

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestAllocate(t *testing.T) {
	max := 10
	m := NewAllocationMap(max, "test")

	if _, ok, _ := m.AllocateNext(); !ok {
		t.Fatalf("unexpected error")
	}
	if m.count != 1 {
		t.Errorf("expect to get %d, but got %d", 1, m.count)
	}
	if f := m.Free(); f != max-1 {
		t.Errorf("expect to get %d, but got %d", max-1, f)
	}
}

func TestAllocateMax(t *testing.T) {
	max := 10
	m := NewAllocationMap(max, "test")
	for i := 0; i < max; i++ {
		if _, ok, _ := m.AllocateNext(); !ok {
			t.Fatalf("unexpected error")
		}
	}

	if _, ok, _ := m.AllocateNext(); ok {
		t.Errorf("unexpected success")
	}
	if f := m.Free(); f != 0 {
		t.Errorf("expect to get %d, but got %d", 0, f)
	}
}

func TestAllocateError(t *testing.T) {
	m := NewAllocationMap(10, "test")
	if ok, _ := m.Allocate(3); !ok {
		t.Errorf("error allocate offset %v", 3)
	}
	if ok, _ := m.Allocate(3); ok {
		t.Errorf("unexpected success")
	}
}

func TestRelease(t *testing.T) {
	offset := 3
	m := NewAllocationMap(10, "test")
	if ok, _ := m.Allocate(offset); !ok {
		t.Errorf("error allocate offset %v", offset)
	}

	if !m.Has(offset) {
		t.Errorf("expect offset %v allocated", offset)
	}

	if err := m.Release(offset); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if m.Has(offset) {
		t.Errorf("expect offset %v not allocated", offset)
	}
}

func TestForEach(t *testing.T) {
	testCases := []sets.Int{
		sets.NewInt(),
		sets.NewInt(0),
		sets.NewInt(0, 2, 5, 9),
		sets.NewInt(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
	}

	for i, tc := range testCases {
		m := NewAllocationMap(10, "test")
		for offset := range tc {
			if ok, _ := m.Allocate(offset); !ok {
				t.Errorf("[%d] error allocate offset %v", i, offset)
			}
			if !m.Has(offset) {
				t.Errorf("[%d] expect offset %v allocated", i, offset)
			}
		}
		calls := sets.NewInt()
		m.ForEach(func(i int) {
			calls.Insert(i)
		})
		if len(calls) != len(tc) {
			t.Errorf("[%d] expected %d calls, got %d", i, len(tc), len(calls))
		}
		if !calls.Equal(tc) {
			t.Errorf("[%d] expected calls to equal testcase: %v vs %v", i, calls.List(), tc.List())
		}
	}
}

func TestSnapshotAndRestore(t *testing.T) {
	offset := 3
	m := NewAllocationMap(10, "test")
	if ok, _ := m.Allocate(offset); !ok {
		t.Errorf("error allocate offset %v", offset)
	}
	spec, bytes := m.Snapshot()

	m2 := NewAllocationMap(10, "test")
	err := m2.Restore(spec, bytes)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if m2.count != 1 {
		t.Errorf("expect count to %d, but got %d", 0, m.count)
	}
	if !m2.Has(offset) {
		t.Errorf("expect offset %v allocated", offset)
	}
}

func TestContiguousAllocation(t *testing.T) {
	max := 10
	m := NewContiguousAllocationMap(max, "test")

	for i := 0; i < max; i++ {
		next, ok, _ := m.AllocateNext()
		if !ok {
			t.Fatalf("unexpected error")
		}
		if next != i {
			t.Fatalf("expect next to %d, but got %d", i, next)
		}
	}

	if _, ok, _ := m.AllocateNext(); ok {
		t.Errorf("unexpected success")
	}
}

func TestBitmapSizeSnapshotAndRestore(t *testing.T) {
	// 1M IPs
	max := 1000000
	m := NewAllocationMap(max, "test")
	// Allocate all
	start := time.Now()
	for i := 0; i < max; i++ {
		_, ok, _ := m.AllocateNext()
		if !ok {
			t.Fatalf("unexpected error")
		}

	}
	elapsed := time.Since(start)
	fmt.Printf("Fill the map took %s\n", elapsed)

	start = time.Now()
	spec, bytes := m.Snapshot()
	elapsed = time.Since(start)
	fmt.Printf("Size snapshot: %d bytes in %v\n", len(bytes), elapsed)

	m2 := NewAllocationMap(max, "test")
	// Check the time to load a snapshot
	start = time.Now()
	err := m2.Restore(spec, bytes)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	elapsed = time.Since(start)
	fmt.Printf("Load the snapshot took %s\n", elapsed)

	if m2.count != max {
		t.Errorf("expect count to %d, but got %d", 0, m.count)
	}

}

func benchmarkBitmap(max int, b *testing.B) {
	m := NewAllocationMap(max, "test")
	for n := 0; n < b.N; n++ {
		m.AllocateNext()
	}
}

func BenchmarkBitmap1000(b *testing.B)     { benchmarkBitmap(1000, b) }
func BenchmarkBitmap10000(b *testing.B)    { benchmarkBitmap(10000, b) }
func BenchmarkBitmap100000(b *testing.B)   { benchmarkBitmap(100000, b) }
func BenchmarkBitmap1000000(b *testing.B)  { benchmarkBitmap(1000000, b) }
func BenchmarkBitmap10000000(b *testing.B) { benchmarkBitmap(10000000, b) }
