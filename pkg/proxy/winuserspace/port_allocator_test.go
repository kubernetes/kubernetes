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

package winuserspace

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/net"
)

func TestRangeAllocatorEmpty(t *testing.T) {
	r := &net.PortRange{}
	r.Set("0-0")
	defer func() {
		if rv := recover(); rv == nil {
			t.Fatalf("expected panic because of empty port range: %#v", r)
		}
	}()
	_ = newPortRangeAllocator(*r, true)
}

func TestRangeAllocatorFullyAllocated(t *testing.T) {
	r := &net.PortRange{}
	r.Set("1-1")
	// Don't auto-fill ports, we'll manually turn the crank
	pra := newPortRangeAllocator(*r, false)
	a := pra.(*rangeAllocator)

	// Fill in the one available port
	if !a.fillPortsOnce() {
		t.Fatalf("Expected to be able to fill ports")
	}

	// There should be no ports available
	if a.fillPortsOnce() {
		t.Fatalf("Expected to be unable to fill ports")
	}

	p, err := a.AllocateNext()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != 1 {
		t.Fatalf("unexpected allocated port: %d", p)
	}

	a.lock.Lock()
	if bit := a.used.Bit(p - a.Base); bit != 1 {
		a.lock.Unlock()
		t.Fatalf("unexpected used bit for allocated port: %d", p)
	}
	a.lock.Unlock()

	_, err = a.AllocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}

	a.Release(p)
	a.lock.Lock()
	if bit := a.used.Bit(p - a.Base); bit != 0 {
		a.lock.Unlock()
		t.Fatalf("unexpected used bit for allocated port: %d", p)
	}
	a.lock.Unlock()

	// Fill in the one available port
	if !a.fillPortsOnce() {
		t.Fatalf("Expected to be able to fill ports")
	}

	p, err = a.AllocateNext()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != 1 {
		t.Fatalf("unexpected allocated port: %d", p)
	}
	a.lock.Lock()
	if bit := a.used.Bit(p - a.Base); bit != 1 {
		a.lock.Unlock()
		t.Fatalf("unexpected used bit for allocated port: %d", p)
	}
	a.lock.Unlock()

	_, err = a.AllocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}
}

func TestRangeAllocator_RandomishAllocation(t *testing.T) {
	r := &net.PortRange{}
	r.Set("1-100")
	pra := newPortRangeAllocator(*r, false)
	a := pra.(*rangeAllocator)

	// allocate all the ports
	var err error
	ports := make([]int, 100, 100)
	for i := 0; i < 100; i++ {
		if !a.fillPortsOnce() {
			t.Fatalf("Expected to be able to fill ports")
		}
		ports[i], err = a.AllocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if ports[i] < 1 || ports[i] > 100 {
			t.Fatalf("unexpected allocated port: %d", ports[i])
		}
		a.lock.Lock()
		if bit := a.used.Bit(ports[i] - a.Base); bit != 1 {
			a.lock.Unlock()
			t.Fatalf("unexpected used bit for allocated port: %d", ports[i])
		}
		a.lock.Unlock()
	}

	if a.fillPortsOnce() {
		t.Fatalf("Expected to be unable to fill ports")
	}

	// release them all
	for i := 0; i < 100; i++ {
		a.Release(ports[i])
		a.lock.Lock()
		if bit := a.used.Bit(ports[i] - a.Base); bit != 0 {
			a.lock.Unlock()
			t.Fatalf("unexpected used bit for allocated port: %d", ports[i])
		}
		a.lock.Unlock()
	}

	// allocate the ports again
	rports := make([]int, 100, 100)
	for i := 0; i < 100; i++ {
		if !a.fillPortsOnce() {
			t.Fatalf("Expected to be able to fill ports")
		}
		rports[i], err = a.AllocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if rports[i] < 1 || rports[i] > 100 {
			t.Fatalf("unexpected allocated port: %d", rports[i])
		}
		a.lock.Lock()
		if bit := a.used.Bit(rports[i] - a.Base); bit != 1 {
			a.lock.Unlock()
			t.Fatalf("unexpected used bit for allocated port: %d", rports[i])
		}
		a.lock.Unlock()
	}

	if a.fillPortsOnce() {
		t.Fatalf("Expected to be unable to fill ports")
	}

	if reflect.DeepEqual(ports, rports) {
		t.Fatalf("expected re-allocated ports to be in a somewhat random order")
	}
}
