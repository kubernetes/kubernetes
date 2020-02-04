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
	// Max value obtained from const MaxHostsSubnet = int64(16777216)
	// "k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	max := 16777216 / 50
	m := NewAllocationMap(max, "test")
	startTime := time.Now()
	for i := 0; i < max; i++ {
		if _, ok, _ := m.AllocateNext(); !ok {
			t.Fatalf("unexpected error")
		}
	}
	endTime := time.Now()

	if _, ok, _ := m.AllocateNext(); ok {
		t.Errorf("unexpected success")
	}
	if f := m.Free(); f != 0 {
		t.Errorf("expect to get %d, but got %d", 0, f)
	}
	t.Logf("%d entries allocated in %s seconds", max, endTime.Sub(startTime).String())

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

func TestRoaringAllocate(t *testing.T) {
	max := 10
	m := NewRoaringAllocationMap(max, "test")

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

func TestRoaringAllocateMax(t *testing.T) {
	// Max value obtained from const MaxHostsSubnet = int64(16777216)
	// "k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	max := 16777216 / 50
	m := NewRoaringAllocationMap(max, "test")
	startTime := time.Now()
	for i := 0; i < max; i++ {
		if _, ok, _ := m.AllocateNext(); !ok {
			t.Fatalf("unexpected error")
		}
	}
	endTime := time.Now()
	if _, ok, _ := m.AllocateNext(); ok {
		t.Errorf("unexpected success")
	}
	if f := m.Free(); f != 0 {
		t.Errorf("expect to get %d, but got %d", 0, f)
	}
	t.Logf("%d entries allocated in %s seconds using %d bytes", max, endTime.Sub(startTime).String(), m.allocated.GetSizeInBytes())
}

func TestRoaringAllocateError(t *testing.T) {
	m := NewRoaringAllocationMap(10, "test")
	if ok, _ := m.Allocate(3); !ok {
		t.Errorf("error allocate offset %v", 3)
	}
	if ok, _ := m.Allocate(3); ok {
		t.Errorf("unexpected success")
	}
}

func TestRoaringRelease(t *testing.T) {
	offset := 3
	m := NewRoaringAllocationMap(10, "test")
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

func TestRoaringForEach(t *testing.T) {
	testCases := []sets.Int{
		sets.NewInt(),
		sets.NewInt(0),
		sets.NewInt(0, 2, 5, 9),
		sets.NewInt(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
	}

	for i, tc := range testCases {
		m := NewRoaringAllocationMap(10, "test")
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

func TestRoaringSnapshotAndRestore(t *testing.T) {
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

func TestRoaringContiguousAllocation(t *testing.T) {
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

func allocateAllBitmap(max int, m Interface) {
	for i := 0; i < max; i++ {
		m.AllocateNext()
	}
}

func benchmarkContiguousAllocationBitmap(b *testing.B, max int) {
	m := NewContiguousAllocationMap(max, "test")
	for n := 0; n < b.N; n++ {
		allocateAllBitmap(max, m)
	}
}

func benchmarkRandomBitmap(b *testing.B, max int) {
	m := NewAllocationMap(max, "test")
	for n := 0; n < b.N; n++ {
		allocateAllBitmap(max, m)
	}
}
func benchmarkRoaringBitmap(b *testing.B, max int) {
	m := NewRoaringAllocationMap(max, "test")
	for n := 0; n < b.N; n++ {
		allocateAllBitmap(max, m)
	}
}

// Bitmaps Contiguous Allocation
func BenchmarkContiguousAllocate10(b *testing.B)     { benchmarkContiguousAllocationBitmap(b, 10) }
func BenchmarkContiguousAllocate100(b *testing.B)    { benchmarkContiguousAllocationBitmap(b, 100) }
func BenchmarkContiguousAllocate1000(b *testing.B)   { benchmarkContiguousAllocationBitmap(b, 1000) }
func BenchmarkContiguousAllocate10000(b *testing.B)  { benchmarkContiguousAllocationBitmap(b, 10000) }
func BenchmarkContiguousAllocate100000(b *testing.B) { benchmarkContiguousAllocationBitmap(b, 100000) }

// Bitmaps Random Allocation
func BenchmarkRandomAllocate10(b *testing.B)     { benchmarkRandomBitmap(b, 10) }
func BenchmarkRandomAllocate100(b *testing.B)    { benchmarkRandomBitmap(b, 100) }
func BenchmarkRandomAllocate1000(b *testing.B)   { benchmarkRandomBitmap(b, 1000) }
func BenchmarkRandomAllocate10000(b *testing.B)  { benchmarkRandomBitmap(b, 10000) }
func BenchmarkRandomAllocate100000(b *testing.B) { benchmarkRandomBitmap(b, 100000) }

// Roaring bitmaps
func BenchmarkRoaringAllocate10(b *testing.B)     { benchmarkRoaringBitmap(b, 10) }
func BenchmarkRoaringAllocate100(b *testing.B)    { benchmarkRoaringBitmap(b, 100) }
func BenchmarkRoaringAllocate1000(b *testing.B)   { benchmarkRoaringBitmap(b, 1000) }
func BenchmarkRoaringAllocate10000(b *testing.B)  { benchmarkRoaringBitmap(b, 10000) }
func BenchmarkRoaringAllocate100000(b *testing.B) { benchmarkRoaringBitmap(b, 100000) }
