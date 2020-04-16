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

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestMapAllocate(t *testing.T) {
	max := 10
	m := NewAllocation(max, "test")

	if _, ok, _ := m.AllocateNext(); !ok {
		t.Fatalf("unexpected error")
	}
	if len(m.allocated) != 1 {
		t.Errorf("expect to get %d, but got %d", 1, len(m.allocated))
	}
	if f := m.Free(); f != max-1 {
		t.Errorf("expect to get %d, but got %d", max-1, f)
	}
}

func TestMapAllocateMax(t *testing.T) {
	max := 10
	m := NewAllocation(max, "test")
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

func TestMapAllocateError(t *testing.T) {
	m := NewAllocation(10, "test")
	if ok, _ := m.Allocate(3); !ok {
		t.Errorf("error allocate offset %v", 3)
	}
	if ok, _ := m.Allocate(3); ok {
		t.Errorf("unexpected success")
	}
}

func TestMapRelease(t *testing.T) {
	offset := 3
	m := NewAllocation(10, "test")
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

func TestMapForEach(t *testing.T) {
	testCases := []sets.Int{
		sets.NewInt(),
		sets.NewInt(0),
		sets.NewInt(0, 2, 5, 9),
		sets.NewInt(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
	}

	for i, tc := range testCases {
		m := NewAllocation(10, "test")
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

func TestMapSnapshotAndRestore(t *testing.T) {
	offset := 3
	m := NewAllocation(10, "test")
	if ok, _ := m.Allocate(offset); !ok {
		t.Errorf("error allocate offset %v", offset)
	}
	spec, bytes := m.Snapshot()

	m2 := NewAllocation(10, "test")
	err := m2.Restore(spec, bytes)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(m2.allocated) != 1 {
		t.Errorf("expect count to %d, but got %d", 0, len(m.allocated))
	}
	if !m2.Has(offset) {
		t.Errorf("expect offset %v allocated", offset)
	}
}

func TestMapContiguousAllocation(t *testing.T) {
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

func benchmarkMap(max int, b *testing.B) {
	m := NewAllocationMap(max, "test")
	for n := 0; n < b.N; n++ {
		m.AllocateNext()
	}
}

func BenchmarkMap1000(b *testing.B)     { benchmarkMap(1000, b) }
func BenchmarkMap10000(b *testing.B)    { benchmarkMap(10000, b) }
func BenchmarkMap100000(b *testing.B)   { benchmarkMap(100000, b) }
func BenchmarkMap1000000(b *testing.B)  { benchmarkMap(1000000, b) }
func BenchmarkMap10000000(b *testing.B) { benchmarkMap(10000000, b) }
