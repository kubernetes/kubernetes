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

package pool

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

type TestPoolDriver struct {
	Items []string
}

func (d *TestPoolDriver) PickRandom(random *math_rand.Rand) string {
	i := random.Int() % len(d.Items)
	return d.Items[i]
}

func (d *TestPoolDriver) Iterator() PoolIterator {
	return &stringSliceIterator{items: d.Items}
}

func TestPoolAllocatorAllocate(t *testing.T, pa PoolAllocator) {
	if _, err := pa.Allocate("z", "owner1"); err != nil {
		// TODO: Maybe it should know?
		t.Errorf("PoolAllocator does not know what items are valid for pool")
	}

	ok, err := pa.Allocate("a", "owner1")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if !ok {
		t.Errorf("expected success")
	}

	ok, err = pa.Allocate("a", "owner1")

	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if ok {
		t.Errorf("expected failure")
	}
}

func TestPoolAllocatorAllocateNext(t *testing.T, pa PoolAllocator) {
	v1, err := pa.AllocateNext("owner1")
	if err != nil {
		t.Errorf("expected 'a', got error: %v", err)
	}
	if v1 != "a" {
		t.Errorf("expected 'a', got %s", v1)
	}

	v2, err := pa.AllocateNext("owner1")
	if err != nil {
		t.Errorf("expected 'b', got error: %v", err)
	}
	if v2 != "b" {
		t.Errorf("expected 'b', got %s", v2)
	}

	v3, err := pa.AllocateNext("owner1")
	if err != nil {
		t.Errorf("expected 'c', got error: %v", err)
	}
	if v3 != "c" {
		t.Errorf("expected 'c', got %s", v3)
	}

	v4, err := pa.AllocateNext("owner1")
	if err != nil {
		t.Errorf("expected no error - (though allocator is full)")
	}
	if v4 != "" {
		t.Errorf("Expected '' - allocator is full")
	}
}

func TestPoolAllocatorRelease(t *testing.T, pa PoolAllocator) {
	ok, err := pa.Release("a")
	if err != nil {
		t.Errorf("expected !ok, got error: %v", err)
	}
	if ok {
		t.Errorf("Expected !ok")
	}

	pa.AllocateNext("owner1")
	v2, _ := pa.AllocateNext("owner1")
	pa.AllocateNext("owner1")

	ok, err = pa.Release(v2)
	if err != nil {
		t.Errorf("expected ok, got error: %v", err)
	}
	if !ok {
		t.Error("Expected release to succeed")
	}

	v2_again, err := pa.AllocateNext("owner1")
	if err != nil {
		t.Errorf("expected %s, got error: %v", v2, err)
	}
	if v2_again != v2 {
		t.Errorf("Expected %s, got %s", v2, v2_again)
	}

	v4, err := pa.AllocateNext("owner1")
	if err != nil {
		t.Errorf("expected no error - though allocator is full")
	}
	if v4 != "" {
		t.Errorf("Expected '' - allocator is full")
	}
}
