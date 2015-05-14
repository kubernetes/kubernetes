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
	"strconv"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestPortPoolIterator(t *testing.T) {
	pr, err := util.ParsePortRange("100-200")
	if err != nil {
		t.Error(err)
	}

	ppd := &portPoolDriver{portRange: *pr}
	it := ppd.Iterate()
	ports := []string{}
	for {
		s, found := it.Next()
		if !found {
			break
		}
		ports = append(ports, s)
	}

	if len(ports) != 101 {
		t.Errorf("unexpected size of port range %d != 101", len(ports))
	}

	for i := 0; i < 101; i++ {
		expected := strconv.Itoa(100 + i)
		if ports[i] != expected {
			t.Errorf("Unexpected port @%d.  expected=%s, actual=%s", i, expected, ports[i])
		}
	}
}

func TestPortAllocatorNew(t *testing.T) {
	if NewPortAllocator(&util.PortRange{Base: 0, Size: 0}, nil) != nil {
		t.Errorf("expected nil for empty port range")
	}
	if NewPortAllocator(&util.PortRange{Base: 100, Size: 1}, nil) != nil {
		t.Errorf("expected nil for too-small port range")
	}
	pr, err := util.ParsePortRange("100-200")
	if err != nil {
		t.Error(err)
	}
	pa := NewPortAllocator(pr, nil)
	if pa == nil {
		t.Errorf("expected non-nil")
	}
	if pa.pool.(*MemoryPoolAllocator).size() != 0 {
		t.Errorf("wrong size() for pa.pool")
	}
}

func TestPortAllocatorAllocate(t *testing.T) {
	pr, err := util.ParsePortRange("100-200")
	if err != nil {
		t.Error(err)
	}
	pa := NewPortAllocator(pr, nil)
	if pa == nil {
		t.Errorf("expected non-nil")
	}

	if err := pa.Allocate(99); err == nil {
		t.Errorf("expected failure")
	}

	if err := pa.Allocate(-1); err == nil {
		t.Errorf("expected failure")
	}

	if err := pa.Allocate(100); err != nil {
		t.Errorf("expected success, got %s", err)
	}

	if pa.Allocate(100) == nil {
		t.Errorf("expected failure")
	}
}

func TestPortAllocatorAllocateNext(t *testing.T) {
	pr, err := util.ParsePortRange("100-200")
	if err != nil {
		t.Error(err)
	}
	pa := NewPortAllocator(pr, nil)
	if pa == nil {
		t.Errorf("expected non-nil")
	}

	// Turn off random allocation attempts, so we just allocate in sequence
	pa.pool.(*MemoryPoolAllocator).randomAttempts = 0

	p1, err := pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if p1 != 100 {
		t.Errorf("expected 100, got %d", p1)
	}

	p2, err := pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if p2 != 101 {
		t.Errorf("expected 101, got %s", p2)
	}

	// Burn a bunch of ports.
	for i := 3; i <= 100; i++ {
		_, err = pa.AllocateNext()
		if err != nil {
			t.Error(err)
		}
	}

	p101, err := pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if p101 != 200 {
		t.Errorf("expected 200, got %s", p101)
	}

	_, err = pa.AllocateNext()
	if err == nil {
		t.Errorf("Expected nil - allocator is full")
	}
}

func TestPortAllocatorRelease(t *testing.T) {
	pr, err := util.ParsePortRange("100-200")
	if err != nil {
		t.Error(err)
	}
	pa := NewPortAllocator(pr, nil)
	if pa == nil {
		t.Errorf("expected non-nil")
	}

	// Turn off random allocation attempts, so we just allocate in sequence
	pa.pool.(*MemoryPoolAllocator).randomAttempts = 0

	err = pa.Release(50)
	if err == nil {
		t.Errorf("Expected an error")
	}

	p1, err := pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	p2, err := pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	_, err = pa.AllocateNext()
	if err != nil {
		t.Error(err)
	}

	err = pa.Release(p2)
	if err != nil {
		t.Error(err)
	}

	p4, err := pa.AllocateNext()
	if p4 != p2 {
		t.Errorf("Expected %d, got %d", p2, p4)
	}

	// Burn a bunch of addresses.
	for i := 4; i <= 101; i++ {
		pa.AllocateNext()
	}
	_, err = pa.AllocateNext()
	if err == nil {
		t.Errorf("Expected an error")
	}
	pa.Release(p1)

	p1_again, err := pa.AllocateNext()
	if p1_again != p1 {
		t.Errorf("Expected %d, got %d", p1, p1_again)
	}
}
