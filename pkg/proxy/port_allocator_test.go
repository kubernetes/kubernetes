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

package proxy

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/util"
)

func TestRangeAllocatorEmpty(t *testing.T) {
	r := &util.PortRange{}
	r.Set("0-0")
	defer func() {
		if rv := recover(); rv == nil {
			t.Fatalf("expected panic because of empty port range: %+v", r)
		}
	}()
	_ = newPortRangeAllocator(*r)
}

func TestRangeAllocatorFullyAllocated(t *testing.T) {
	r := &util.PortRange{}
	r.Set("1-1")
	a := newPortRangeAllocator(*r)
	p, err := a.AllocateNext()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != 1 {
		t.Fatalf("unexpected allocated port: %d", p)
	}

	_, err = a.AllocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}

	a.Release(p)
	p, err = a.AllocateNext()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != 1 {
		t.Fatalf("unexpected allocated port: %d", p)
	}

	_, err = a.AllocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}
}

func TestRangeAllocator_RandomishAllocation(t *testing.T) {
	r := &util.PortRange{}
	r.Set("1-100")
	a := newPortRangeAllocator(*r)

	// allocate all the ports
	var err error
	ports := make([]int, 100, 100)
	for i := 0; i < 100; i++ {
		ports[i], err = a.AllocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	// release them all
	for i := 0; i < 100; i++ {
		a.Release(ports[i])
	}

	// allocate the ports again
	rports := make([]int, 100, 100)
	for i := 0; i < 100; i++ {
		rports[i], err = a.AllocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if reflect.DeepEqual(ports, rports) {
		t.Fatalf("expected re-allocated ports to be in a somewhat random order")
	}
}
