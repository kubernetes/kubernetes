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

package portallocator

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"strconv"
)

func TestAllocate(t *testing.T) {
	pr, err := util.ParsePortRange("10000-10200")
	if err != nil {
		t.Fatal(err)
	}
	r := NewPortAllocator(*pr)
	if f := r.Free(); f != 201 {
		t.Errorf("unexpected free %d", f)
	}
	found := util.NewStringSet()
	count := 0
	for r.Free() > 0 {
		p, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ %d: %v", count, err)
		}
		count++
		if !pr.Contains(p) {
			t.Fatalf("allocated %s which is outside of %s", p, pr)
		}
		if found.Has(strconv.Itoa(p)) {
			t.Fatalf("allocated %s twice @ %d", p, count)
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
	p, err := r.AllocateNext()
	if err != nil {
		t.Fatal(err)
	}
	if released != p {
		t.Errorf("unexpected %s : %s", p, released)
	}

	if err := r.Release(released); err != nil {
		t.Fatal(err)
	}
	if err := r.Allocate(1); err != ErrNotInRange {
		t.Fatal(err)
	}
	if err := r.Allocate(10001); err != ErrAllocated {
		t.Fatal(err)
	}
	if err := r.Allocate(20000); err != ErrNotInRange {
		t.Fatal(err)
	}
	if err := r.Allocate(10201); err != ErrNotInRange {
		t.Fatal(err)
	}
	if f := r.Free(); f != 1 {
		t.Errorf("unexpected free %d", f)
	}
	if err := r.Allocate(released); err != nil {
		t.Fatal(err)
	}
	if f := r.Free(); f != 0 {
		t.Errorf("unexpected free %d", f)
	}
}

func TestSnapshot(t *testing.T) {
	pr, err := util.ParsePortRange("10000-10200")
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

	pr2, err := util.ParsePortRange(dst.Range)
	if err != nil {
		t.Fatal(err)
	}

	if pr.String() != pr2.String() {
		t.Fatalf("mismatched networks: %s : %s", pr, pr2)
	}

	otherPr, err := util.ParsePortRange("200-300")
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
			t.Errorf("restored range does not have %s", n)
		}
	}
	if other.Free() != r.Free() {
		t.Errorf("counts do not match: %d", other.Free())
	}
}
