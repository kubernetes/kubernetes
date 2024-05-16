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

func TestAllocate(t *testing.T) {
	testCases := []struct {
		name      string
		allocator func(max int, rangeSpec string, reserved int) *AllocationBitmap
		max       int
		reserved  int
	}{
		{
			name:      "NewAllocationMap",
			allocator: NewAllocationMapWithOffset,
			max:       32,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max < 16",
			allocator: NewAllocationMapWithOffset,
			max:       8,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max > 16",
			allocator: NewAllocationMapWithOffset,
			max:       128,
			reserved:  16,
		},
		{
			name:      "NewAllocationMapWithOffset max > 256",
			allocator: NewAllocationMapWithOffset,
			max:       1024,
			reserved:  64,
		},
		{
			name:      "NewAllocationMapWithOffset max value",
			allocator: NewAllocationMapWithOffset,
			max:       65535,
			reserved:  256,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.allocator(tc.max, "test", tc.reserved)

			if _, ok, _ := m.AllocateNext(); !ok {
				t.Fatalf("unexpected error")
			}
			if m.count != 1 {
				t.Errorf("expect to get %d, but got %d", 1, m.count)
			}
			if f := m.Free(); f != tc.max-1 {
				t.Errorf("expect to get %d, but got %d", tc.max-1, f)
			}
		})
	}
}

func TestAllocateMax(t *testing.T) {
	testCases := []struct {
		name      string
		allocator func(max int, rangeSpec string, reserved int) *AllocationBitmap
		max       int
		reserved  int
	}{
		{
			name:      "NewAllocationMap",
			allocator: NewAllocationMapWithOffset,
			max:       32,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max < 16",
			allocator: NewAllocationMapWithOffset,
			max:       8,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max > 16",
			allocator: NewAllocationMapWithOffset,
			max:       128,
			reserved:  16,
		},
		{
			name:      "NewAllocationMapWithOffset max > 256",
			allocator: NewAllocationMapWithOffset,
			max:       1024,
			reserved:  64,
		},
		{
			name:      "NewAllocationMapWithOffset max value",
			allocator: NewAllocationMapWithOffset,
			max:       65535,
			reserved:  256,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.allocator(tc.max, "test", tc.reserved)
			for i := 0; i < tc.max; i++ {
				if ok, err := m.Allocate(i); !ok || err != nil {
					t.Fatalf("unexpected error")
				}
			}
			if _, ok, _ := m.AllocateNext(); ok {
				t.Errorf("unexpected success")
			}

			if ok, err := m.Allocate(tc.max); ok || err == nil {
				t.Fatalf("unexpected allocation")
			}

			if f := m.Free(); f != 0 {
				t.Errorf("expect to get %d, but got %d", 0, f)
			}
		})
	}
}

func TestAllocateNextMax(t *testing.T) {
	testCases := []struct {
		name      string
		allocator func(max int, rangeSpec string, reserved int) *AllocationBitmap
		max       int
		reserved  int
	}{
		{
			name:      "NewAllocationMap",
			allocator: NewAllocationMapWithOffset,
			max:       32,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max < 16",
			allocator: NewAllocationMapWithOffset,
			max:       8,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max > 16",
			allocator: NewAllocationMapWithOffset,
			max:       128,
			reserved:  16,
		},
		{
			name:      "NewAllocationMapWithOffset max > 256",
			allocator: NewAllocationMapWithOffset,
			max:       1024,
			reserved:  64,
		},
		{
			name:      "NewAllocationMapWithOffset max value",
			allocator: NewAllocationMapWithOffset,
			max:       65535,
			reserved:  256,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.allocator(tc.max, "test", tc.reserved)
			for i := 0; i < tc.max; i++ {
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
		})
	}
}
func TestAllocateError(t *testing.T) {
	testCases := []struct {
		name      string
		allocator func(max int, rangeSpec string, reserved int) *AllocationBitmap
		max       int
		reserved  int
	}{
		{
			name:      "NewAllocationMap",
			allocator: NewAllocationMapWithOffset,
			max:       32,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max < 16",
			allocator: NewAllocationMapWithOffset,
			max:       8,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max > 16",
			allocator: NewAllocationMapWithOffset,
			max:       128,
			reserved:  16,
		},
		{
			name:      "NewAllocationMapWithOffset max > 256",
			allocator: NewAllocationMapWithOffset,
			max:       1024,
			reserved:  64,
		},
		{
			name:      "NewAllocationMapWithOffset max value",
			allocator: NewAllocationMapWithOffset,
			max:       65535,
			reserved:  256,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.allocator(tc.max, "test", tc.reserved)
			if ok, _ := m.Allocate(3); !ok {
				t.Errorf("error allocate offset %v", 3)
			}
			if ok, _ := m.Allocate(3); ok {
				t.Errorf("unexpected success")
			}
		})
	}
}

func TestRelease(t *testing.T) {
	testCases := []struct {
		name      string
		allocator func(max int, rangeSpec string, reserved int) *AllocationBitmap
		max       int
		reserved  int
	}{
		{
			name:      "NewAllocationMap",
			allocator: NewAllocationMapWithOffset,
			max:       32,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max < 16",
			allocator: NewAllocationMapWithOffset,
			max:       8,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max > 16",
			allocator: NewAllocationMapWithOffset,
			max:       128,
			reserved:  16,
		},
		{
			name:      "NewAllocationMapWithOffset max > 256",
			allocator: NewAllocationMapWithOffset,
			max:       1024,
			reserved:  64,
		},
		{
			name:      "NewAllocationMapWithOffset max value",
			allocator: NewAllocationMapWithOffset,
			max:       65535,
			reserved:  256,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.allocator(tc.max, "test", tc.reserved)
			offset := 3
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
		})
	}

}

func TestForEach(t *testing.T) {
	testCases := []struct {
		name      string
		allocator func(max int, rangeSpec string, reserved int) *AllocationBitmap
		max       int
		reserved  int
	}{
		{
			name:      "NewAllocationMap",
			allocator: NewAllocationMapWithOffset,
			max:       32,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max < 16",
			allocator: NewAllocationMapWithOffset,
			max:       8,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max > 16",
			allocator: NewAllocationMapWithOffset,
			max:       128,
			reserved:  16,
		},
		{
			name:      "NewAllocationMapWithOffset max > 256",
			allocator: NewAllocationMapWithOffset,
			max:       1024,
			reserved:  64,
		},
		{
			name:      "NewAllocationMapWithOffset max value",
			allocator: NewAllocationMapWithOffset,
			max:       65535,
			reserved:  256,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			subTests := []sets.Int{
				sets.NewInt(),
				sets.NewInt(0),
				sets.NewInt(0, 2, 5),
				sets.NewInt(0, 1, 2, 3, 4, 5, 6, 7),
			}

			for i, ts := range subTests {
				m := tc.allocator(tc.max, "test", tc.reserved)
				for offset := range ts {
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
				if len(calls) != len(ts) {
					t.Errorf("[%d] expected %d calls, got %d", i, len(ts), len(calls))
				}
				if !calls.Equal(ts) {
					t.Errorf("[%d] expected calls to equal testcase: %v vs %v", i, calls.List(), ts.List())
				}
			}
		})
	}
}

func TestSnapshotAndRestore(t *testing.T) {
	testCases := []struct {
		name      string
		allocator func(max int, rangeSpec string, reserved int) *AllocationBitmap
		max       int
		reserved  int
	}{
		{
			name:      "NewAllocationMap",
			allocator: NewAllocationMapWithOffset,
			max:       32,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max < 16",
			allocator: NewAllocationMapWithOffset,
			max:       8,
			reserved:  0,
		},
		{
			name:      "NewAllocationMapWithOffset max > 16",
			allocator: NewAllocationMapWithOffset,
			max:       128,
			reserved:  16,
		},
		{
			name:      "NewAllocationMapWithOffset max > 256",
			allocator: NewAllocationMapWithOffset,
			max:       1024,
			reserved:  64,
		},
		{
			name:      "NewAllocationMapWithOffset max value",
			allocator: NewAllocationMapWithOffset,
			max:       65535,
			reserved:  256,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.allocator(tc.max, "test", tc.reserved)
			offset := 3
			if ok, _ := m.Allocate(offset); !ok {
				t.Errorf("error allocate offset %v", offset)
			}
			spec, bytes := m.Snapshot()

			m2 := tc.allocator(10, "test", tc.reserved)
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
		})
	}

}

// TestAllocateMaxReserved should allocate first values greater or equal than the reserved values
func TestAllocateMax_BitmapReserved(t *testing.T) {
	max := 128
	dynamicOffset := 16

	// just to double check off by one errors
	allocated := 0
	// modify if necessary
	m := NewAllocationMapWithOffset(max, "test", dynamicOffset)
	for i := 0; i < max-dynamicOffset; i++ {
		if _, ok, _ := m.AllocateNext(); !ok {
			t.Fatalf("unexpected error")
		}
		allocated++
	}

	if f := m.Free(); f != dynamicOffset {
		t.Errorf("expect to get %d, but got %d", dynamicOffset-1, f)
	}

	for i := 0; i < dynamicOffset; i++ {
		if m.Has(i) {
			t.Errorf("unexpected allocated value %d", i)
		}
	}
	// it should allocate one value of the reserved block
	if _, ok, _ := m.AllocateNext(); !ok {
		t.Fatalf("unexpected error")
	}
	allocated++
	if allocated != m.count {
		t.Errorf("expect to get %d, but got %d", allocated, m.count)
	}

	if m.count != max-dynamicOffset+1 {
		t.Errorf("expect to get %d, but got %d", max-dynamicOffset+1, m.count)
	}
	if f := m.Free(); f != max-allocated {
		t.Errorf("expect to get %d, but got %d", max-allocated, f)
	}
}

func TestPreAllocateReservedFull_BitmapReserved(t *testing.T) {
	max := 128
	dynamicOffset := 16
	// just to double check off by one errors
	allocated := 0
	m := NewAllocationMapWithOffset(max, "test", dynamicOffset)
	// Allocate all possible values except the reserved
	for i := dynamicOffset; i < max; i++ {
		if ok, _ := m.Allocate(i); !ok {
			t.Errorf("error allocate i %v", i)
		} else {
			allocated++
		}
	}
	// Allocate all the values of the reserved block except one
	for i := 0; i < dynamicOffset-1; i++ {
		if ok, _ := m.Allocate(i); !ok {
			t.Errorf("error allocate i %v", i)
		} else {
			allocated++
		}
	}

	// there should be only one free value
	if f := m.Free(); f != 1 {
		t.Errorf("expect to get %d, but got %d", 1, f)
	}
	// check if the last free value is in the lower band
	count := 0
	for i := 0; i < dynamicOffset; i++ {
		if !m.Has(i) {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected one remaining free value, got %d", count)
	}

	if _, ok, _ := m.AllocateNext(); !ok {
		t.Errorf("unexpected allocation error")
	} else {
		allocated++
	}
	if f := m.Free(); f != 0 {
		t.Errorf("expect to get %d, but got %d", max-1, f)
	}

	if _, ok, _ := m.AllocateNext(); ok {
		t.Errorf("unexpected success")
	}
	if m.count != allocated {
		t.Errorf("expect to get %d, but got %d", max, m.count)
	}
	if f := m.Free(); f != 0 {
		t.Errorf("expect to get %d, but got %d", max-1, f)
	}
}

func TestAllocateUniqueness(t *testing.T) {
	max := 128
	dynamicOffset := 16
	uniqueAllocated := map[int]bool{}
	m := NewAllocationMapWithOffset(max, "test", dynamicOffset)

	// Allocate all the values in both the dynamic and reserved blocks
	for i := 0; i < max; i++ {
		alloc, ok, _ := m.AllocateNext()
		if !ok {
			t.Fatalf("unexpected error")
		}
		if _, ok := uniqueAllocated[alloc]; ok {
			t.Fatalf("unexpected allocated value %d", alloc)
		} else {
			uniqueAllocated[alloc] = true
		}
	}

	if max != len(uniqueAllocated) {
		t.Errorf("expect to get %d, but got %d", max, len(uniqueAllocated))
	}
}
