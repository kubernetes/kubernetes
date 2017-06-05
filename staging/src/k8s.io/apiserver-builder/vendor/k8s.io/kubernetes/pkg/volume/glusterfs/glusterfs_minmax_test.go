/*
Copyright 2016 The Kubernetes Authors.

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

package glusterfs

import (
	"testing"
)

func TestNewFree(t *testing.T) {
	min := 1
	max := 10

	m, err := NewMinMaxAllocator(min, max)
	if err != nil {
		t.Errorf("error creating new allocator: '%v'", err)
	}

	if f := m.Free(); f != (max - min + 1) {
		t.Errorf("expect to get %d free, but got %d", (max - min + 1), f)
	}
}

func TestNewInvalidRange(t *testing.T) {
	if _, err := NewMinMaxAllocator(10, 1); err != ErrInvalidRange {
		t.Errorf("expect to get Error '%v', got '%v'", ErrInvalidRange, err)
	}
}

func TestSetRange(t *testing.T) {
	min := 1
	max := 10

	m, err := NewMinMaxAllocator(min, max)
	if err != nil {
		t.Errorf("error creating new allocator: '%v'", err)
	}

	if err = m.SetRange(10, 1); err != ErrInvalidRange {
		t.Errorf("expected to get error '%v', got '%v'", ErrInvalidRange, err)
	}

	if err = m.SetRange(1, 2); err != nil {
		t.Errorf("error setting range: '%v'", err)
	}

	if f := m.Free(); f != 2 {
		t.Errorf("expect to get %d free, but got %d", 2, f)
	}

	if ok, _ := m.Allocate(1); !ok {
		t.Errorf("error allocate offset %v", 1)
	}

	if f := m.Free(); f != 1 {
		t.Errorf("expect to get 1 free, but got %d", f)
	}

	if err = m.SetRange(1, 1); err != nil {
		t.Errorf("error setting range: '%v'", err)
	}

	if f := m.Free(); f != 0 {
		t.Errorf("expect to get 0 free, but got %d", f)
	}

	if err = m.SetRange(2, 2); err != nil {
		t.Errorf("error setting range: '%v'", err)
	}

	if f := m.Free(); f != 1 {
		t.Errorf("expect to get 1 free, but got %d", f)
	}
}

func TestAllocateNext(t *testing.T) {
	min := 1
	max := 10

	m, err := NewMinMaxAllocator(min, max)
	if err != nil {
		t.Errorf("error creating new allocator: '%v'", err)
	}

	el, ok, _ := m.AllocateNext()
	if !ok {
		t.Fatalf("unexpected error")
	}

	if !m.Has(el) {
		t.Errorf("expect element %v allocated", el)
	}

	if f := m.Free(); f != (max-min+1)-1 {
		t.Errorf("expect to get %d free, but got %d", (max-min+1)-1, f)
	}
}

func TestAllocateMax(t *testing.T) {
	min := 1
	max := 10

	m, err := NewMinMaxAllocator(min, max)
	if err != nil {
		t.Errorf("error creating new allocator: '%v'", err)
	}

	for i := 1; i <= max; i++ {
		if _, ok, _ := m.AllocateNext(); !ok {
			t.Fatalf("unexpected error")
		}
	}

	if _, ok, _ := m.AllocateNext(); ok {
		t.Errorf("unexpected success")
	}

	if f := m.Free(); f != 0 {
		t.Errorf("expect to get %d free, but got %d", 0, f)
	}
}

func TestAllocate(t *testing.T) {
	min := 1
	max := 10
	offset := 3

	m, err := NewMinMaxAllocator(min, max)
	if err != nil {
		t.Errorf("error creating new allocator: '%v'", err)
	}

	if ok, err := m.Allocate(offset); !ok {
		t.Errorf("error allocate offset %v: %v", offset, err)
	}

	if !m.Has(offset) {
		t.Errorf("expect element %v allocated", offset)
	}

	if f := m.Free(); f != (max-min+1)-1 {
		t.Errorf("expect to get %d free, but got %d", (max-min+1)-1, f)
	}
}

func TestAllocateConflict(t *testing.T) {
	min := 1
	max := 10
	offset := 3

	m, err := NewMinMaxAllocator(min, max)
	if err != nil {
		t.Errorf("error creating new allocator: '%v'", err)
	}

	if ok, err := m.Allocate(offset); !ok {
		t.Errorf("error allocate offset %v: %v", offset, err)
	}

	ok, err := m.Allocate(offset)
	if ok {
		t.Errorf("unexpected success")
	}
	if err != ErrConflict {
		t.Errorf("expected error '%v', got '%v'", ErrConflict, err)
	}
}

func TestAllocateOutOfRange(t *testing.T) {
	min := 1
	max := 10
	offset := 11

	m, err := NewMinMaxAllocator(min, max)
	if err != nil {
		t.Errorf("error creating new allocator: '%v'", err)
	}

	ok, err := m.Allocate(offset)
	if ok {
		t.Errorf("unexpected success")
	}
	if err != ErrOutOfRange {
		t.Errorf("expected error '%v', got '%v'", ErrOutOfRange, err)
	}
}

func TestRelease(t *testing.T) {
	min := 1
	max := 10
	offset := 3

	m, err := NewMinMaxAllocator(min, max)
	if err != nil {
		t.Errorf("error creating new allocator: '%v'", err)
	}

	if ok, err := m.Allocate(offset); !ok {
		t.Errorf("error allocate offset %v: %v", offset, err)
	}

	if !m.Has(offset) {
		t.Errorf("expect offset %v allocated", offset)
	}

	if err = m.Release(offset); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if m.Has(offset) {
		t.Errorf("expect offset %v not allocated", offset)
	}
}
