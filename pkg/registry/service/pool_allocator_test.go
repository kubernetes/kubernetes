/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package service

import (
	math_rand "math/rand"
	"testing"
)

type stringSliceIterator struct {
	items []string
	count int
}

func (it *stringSliceIterator) Next() (string, bool) {
	it.count++
	if it.count > len(it.items) {
		return "", false
	}
	return it.items[it.count-1], true
}

type testPoolDriver struct {
	items []string
}

func (d *testPoolDriver) PickRandom(random *math_rand.Rand) string {
	i := random.Int() % len(d.items)
	return d.items[i]
}

func (d *testPoolDriver) Iterate() StringIterator {
	return &stringSliceIterator{items: d.items}
}

func (d *testPoolDriver) IterateNext(last string) string {
	// Note this won't do well with duplicates!
	for i, s := range d.items {
		if s == last {
			next := i + 1
			if next == len(d.items) {
				return ""
			} else {
				return d.items[next]
			}
		}
	}
	return ""
}

func TestPoolAllocatorAllocate(t *testing.T) {
	var pa PoolAllocator

	driver := &testPoolDriver{items: []string{"1", "2", "3"}}
	pa.Init(driver)

	if err := pa.Allocate("0"); err != nil {
		// TODO: Maybe it should?
		t.Errorf("PoolAllocator does not know what items are valid for pool")
	}

	if err := pa.Allocate("1"); err != nil {
		t.Errorf("expected success, got %s", err)
	}

	if pa.Allocate("1") == nil {
		t.Errorf("expected failure")
	}
}

func TestPoolAllocatorAllocateNext(t *testing.T) {
	var pa PoolAllocator

	driver := &testPoolDriver{items: []string{"a", "b", "c"}}
	pa.Init(driver)

	// Turn off random allocation attempts, so we just allocate in sequence
	pa.randomAttempts = 0

	v1 := pa.AllocateNext()
	if v1 != "a" {
		t.Errorf("expected 'a', got %s", v1)
	}

	v2 := pa.AllocateNext()
	if v2 != "b" {
		t.Errorf("expected 'b', got %s", v2)
	}

	v3 := pa.AllocateNext()
	if v3 != "c" {
		t.Errorf("expected 'c', got %s", v3)
	}

	v4 := pa.AllocateNext()
	if v4 != "" {
		t.Errorf("Expected '' - allocator is full")
	}
}

func TestPoolAllocatorRelease(t *testing.T) {
	var pa PoolAllocator

	driver := &testPoolDriver{items: []string{"a", "b", "c"}}
	pa.Init(driver)

	pa.randomAttempts = 0

	ok := pa.Release("a")
	if ok {
		t.Errorf("Expected an error")
	}

	pa.AllocateNext()
	v2 := pa.AllocateNext()
	pa.AllocateNext()

	ok = pa.Release(v2)
	if !ok {
		t.Error("Expected release to succeed")
	}

	v2_again := pa.AllocateNext()
	if v2_again != v2 {
		t.Errorf("Expected %s, got %s", v2, v2_again)
	}

	v4 := pa.AllocateNext()
	if v4 != "" {
		t.Errorf("Expected '' - allocator is full")
	}
}
