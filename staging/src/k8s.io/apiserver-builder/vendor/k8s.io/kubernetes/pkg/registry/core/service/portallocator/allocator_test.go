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

package portallocator

import (
	"testing"

	"strconv"

	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
)

func TestAllocate(t *testing.T) {
	pr, err := net.ParsePortRange("10000-10200")
	if err != nil {
		t.Fatal(err)
	}
	r := NewPortAllocator(*pr)
	if f := r.Free(); f != 201 {
		t.Errorf("unexpected free %d", f)
	}
	if f := r.Used(); f != 0 {
		t.Errorf("unexpected used %d", f)
	}
	found := sets.NewString()
	count := 0
	for r.Free() > 0 {
		p, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ %d: %v", count, err)
		}
		count++
		if !pr.Contains(p) {
			t.Fatalf("allocated %d which is outside of %v", p, pr)
		}
		if found.Has(strconv.Itoa(p)) {
			t.Fatalf("allocated %d twice @ %d", p, count)
		}
		found.Insert(strconv.Itoa(p))
	}
	if _, err := r.AllocateNext(); err != ErrFull {
		t.Fatal(err)
	}

	released := 10005
	if err := r.Release(released); err != nil {
		t.Fatal(err)
	}
	if f := r.Free(); f != 1 {
		t.Errorf("unexpected free %d", f)
	}
	if f := r.Used(); f != 200 {
		t.Errorf("unexpected used %d", f)
	}
	p, err := r.AllocateNext()
	if err != nil {
		t.Fatal(err)
	}
	if released != p {
		t.Errorf("unexpected %d : %d", p, released)
	}

	if err := r.Release(released); err != nil {
		t.Fatal(err)
	}

	err = r.Allocate(1)
	if _, ok := err.(*ErrNotInRange); !ok {
		t.Fatal(err)
	}

	if err := r.Allocate(10001); err != ErrAllocated {
		t.Fatal(err)
	}

	err = r.Allocate(20000)
	if _, ok := err.(*ErrNotInRange); !ok {
		t.Fatal(err)
	}

	err = r.Allocate(10201)
	if _, ok := err.(*ErrNotInRange); !ok {
		t.Fatal(err)
	}
	if f := r.Free(); f != 1 {
		t.Errorf("unexpected free %d", f)
	}
	if f := r.Used(); f != 200 {
		t.Errorf("unexpected used %d", f)
	}
	if err := r.Allocate(released); err != nil {
		t.Fatal(err)
	}
	if f := r.Free(); f != 0 {
		t.Errorf("unexpected free %d", f)
	}
	if f := r.Used(); f != 201 {
		t.Errorf("unexpected used %d", f)
	}
}

func TestForEach(t *testing.T) {
	pr, err := net.ParsePortRange("10000-10200")
	if err != nil {
		t.Fatal(err)
	}

	testCases := []sets.Int{
		sets.NewInt(),
		sets.NewInt(10000),
		sets.NewInt(10000, 10200),
		sets.NewInt(10000, 10099, 10200),
	}

	for i, tc := range testCases {
		r := NewPortAllocator(*pr)

		for port := range tc {
			if err := r.Allocate(port); err != nil {
				t.Errorf("[%d] error allocating port %v: %v", i, port, err)
			}
			if !r.Has(port) {
				t.Errorf("[%d] expected port %v allocated", i, port)
			}
		}

		calls := sets.NewInt()
		r.ForEach(func(port int) {
			calls.Insert(port)
		})
		if len(calls) != len(tc) {
			t.Errorf("[%d] expected %d calls, got %d", i, len(tc), len(calls))
		}
		if !calls.Equal(tc) {
			t.Errorf("[%d] expected calls to equal testcase: %v vs %v", i, calls.List(), tc.List())
		}
	}
}

func TestSnapshot(t *testing.T) {
	pr, err := net.ParsePortRange("10000-10200")
	if err != nil {
		t.Fatal(err)
	}
	r := NewPortAllocator(*pr)
	ports := []int{}
	for i := 0; i < 10; i++ {
		port, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		ports = append(ports, port)
	}

	var dst api.RangeAllocation
	err = r.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	pr2, err := net.ParsePortRange(dst.Range)
	if err != nil {
		t.Fatal(err)
	}

	if pr.String() != pr2.String() {
		t.Fatalf("mismatched networks: %s : %s", pr, pr2)
	}

	otherPr, err := net.ParsePortRange("200-300")
	if err != nil {
		t.Fatal(err)
	}
	other := NewPortAllocator(*otherPr)
	if err := r.Restore(*otherPr, dst.Data); err != ErrMismatchedNetwork {
		t.Fatal(err)
	}
	other = NewPortAllocator(*pr2)
	if err := other.Restore(*pr2, dst.Data); err != nil {
		t.Fatal(err)
	}

	for _, n := range ports {
		if !other.Has(n) {
			t.Errorf("restored range does not have %d", n)
		}
	}
	if other.Free() != r.Free() {
		t.Errorf("counts do not match: %d", other.Free())
	}
}

func TestNewFromSnapshot(t *testing.T) {
	pr, err := net.ParsePortRange("200-300")
	if err != nil {
		t.Fatal(err)
	}
	r := NewPortAllocator(*pr)
	allocated := []int{}
	for i := 0; i < 50; i++ {
		p, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		allocated = append(allocated, p)
	}

	snapshot := api.RangeAllocation{}
	if err = r.Snapshot(&snapshot); err != nil {
		t.Fatal(err)
	}

	r, err = NewFromSnapshot(&snapshot)
	if err != nil {
		t.Fatal(err)
	}

	if x := r.Free(); x != 51 {
		t.Fatalf("expected 51 free ports, got %d", x)
	}
	if x := r.Used(); x != 50 {
		t.Fatalf("expected 50 used port, got %d", x)
	}

	for _, p := range allocated {
		if !r.Has(p) {
			t.Fatalf("expected port to be allocated, but it was not")
		}
	}
}
