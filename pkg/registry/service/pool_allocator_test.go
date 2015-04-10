/*
Copyright 2014 Google Inc. All rights reserved.

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
	"net"
	"testing"
)

type testPoolDriver struct {
	items []string
}

func (d *testPoolDriver) PickRandom(seed int) string {
	i := seed % len(d.items)
	return d.items[i]
}

func (d *testPoolDriver) IterateStart() string {
	return d.items[0]
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

func TestAllocate(t *testing.T) {
	var pa PoolAllocator

	driver := &testPoolDriver{items: {"1", "2", "3"}}
	pa.Init(driver)

	if err := pa.Allocate("0"); err != nil {
		// TODO: Maybe it should?
		t.Errorf("PoolAllocator does not know what items are valid for pool")
	}

	if err := ipa.Allocate("1"); err != nil {
		t.Errorf("expected success, got %s", err)
	}

	if ipa.Allocate("1") == nil {
		t.Errorf("expected failure")
	}
}

func TestAllocateNext(t *testing.T) {
	var pa PoolAllocator

	driver := &testPoolDriver{items: {"a", "b", "c"}}
	pa.Init(driver)

	// Turn off random allocation attempts, so we just allocate in sequence
	pa.randomAttempts = 0

	v1, err := pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if v1 != "1" {
		t.Errorf("expected 'a', got %s", v1)
	}

	v2, err := pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if v2 != "b" {
		t.Errorf("expected 'b', got %s", v2)
	}

	v3, err := pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if v3 != "c" {
		t.Errorf("expected 'c', got %s", v3)
	}

	_, err = pa.AllocateNext()
	if err == nil {
		t.Errorf("Expected nil - allocator is full")
	}
}

func TestRelease(t *testing.T) {
	var pa PoolAllocator

	driver := &testPoolDriver{items: {"a", "b", "c"}}
	pa.Init(driver)

	ipa.randomAttempts = 0

	err := ipa.Release("a")
	if err == nil {
		t.Errorf("Expected an error")
	}

	v1, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	v2, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	_, err = ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}

	err = ipa.Release(v2)
	if err != nil {
		t.Error(err)
	}

	v2_again, err := ipa.AllocateNext()
	if v2_again != v2 {
		t.Errorf("Expected %s, got %s", v2, v2_again)
	}

	_, err = pa.AllocateNext()
	if err == nil {
		t.Errorf("Expected nil - allocator is full")
	}
}
