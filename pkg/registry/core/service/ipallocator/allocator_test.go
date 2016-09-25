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

package ipallocator

import (
	"net"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestAllocate(t *testing.T) {
	_, cidr, err := net.ParseCIDR("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}
	r := NewCIDRRange(cidr)
	t.Logf("base: %v", r.base.Bytes())
	if f := r.Free(); f != 254 {
		t.Errorf("unexpected free %d", f)
	}
	found := sets.NewString()
	count := 0
	for r.Free() > 0 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ %d: %v", count, err)
		}
		count++
		if !cidr.Contains(ip) {
			t.Fatalf("allocated %s which is outside of %s", ip, cidr)
		}
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice @ %d", ip, count)
		}
		found.Insert(ip.String())
	}
	if _, err := r.AllocateNext(); err != ErrFull {
		t.Fatal(err)
	}

	released := net.ParseIP("192.168.1.5")
	if err := r.Release(released); err != nil {
		t.Fatal(err)
	}
	if f := r.Free(); f != 1 {
		t.Errorf("unexpected free %d", f)
	}
	ip, err := r.AllocateNext()
	if err != nil {
		t.Fatal(err)
	}
	if !released.Equal(ip) {
		t.Errorf("unexpected %s : %s", ip, released)
	}

	if err := r.Release(released); err != nil {
		t.Fatal(err)
	}
	if err := r.Allocate(net.ParseIP("192.168.0.1")); err != ErrNotInRange {
		t.Fatal(err)
	}
	if err := r.Allocate(net.ParseIP("192.168.1.1")); err != ErrAllocated {
		t.Fatal(err)
	}
	if err := r.Allocate(net.ParseIP("192.168.1.0")); err != ErrNotInRange {
		t.Fatal(err)
	}
	if err := r.Allocate(net.ParseIP("192.168.1.255")); err != ErrNotInRange {
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

func TestAllocateTiny(t *testing.T) {
	_, cidr, err := net.ParseCIDR("192.168.1.0/32")
	if err != nil {
		t.Fatal(err)
	}
	r := NewCIDRRange(cidr)
	if f := r.Free(); f != 0 {
		t.Errorf("free: %d", f)
	}
	if _, err := r.AllocateNext(); err != ErrFull {
		t.Error(err)
	}
}

func TestAllocateSmall(t *testing.T) {
	_, cidr, err := net.ParseCIDR("192.168.1.240/30")
	if err != nil {
		t.Fatal(err)
	}
	r := NewCIDRRange(cidr)
	if f := r.Free(); f != 2 {
		t.Errorf("free: %d", f)
	}
	found := sets.NewString()
	for i := 0; i < 2; i++ {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		if found.Has(ip.String()) {
			t.Fatalf("already reserved: %s", ip)
		}
		found.Insert(ip.String())
	}
	for s := range found {
		if !r.Has(net.ParseIP(s)) {
			t.Fatalf("missing: %s", s)
		}
		if err := r.Allocate(net.ParseIP(s)); err != ErrAllocated {
			t.Fatal(err)
		}
	}
	for i := 0; i < 100; i++ {
		if _, err := r.AllocateNext(); err != ErrFull {
			t.Fatalf("suddenly became not-full: %#v", r)
		}
	}

	if r.Free() != 0 && r.max != 2 {
		t.Fatalf("unexpected range: %v", r)
	}

	t.Logf("allocated: %v", found)
}

func TestRangeSize(t *testing.T) {
	testCases := map[string]int64{
		"192.168.1.0/24": 256,
		"192.168.1.0/32": 1,
		"192.168.1.0/31": 2,
	}
	for k, v := range testCases {
		_, cidr, err := net.ParseCIDR(k)
		if err != nil {
			t.Fatal(err)
		}
		if size := RangeSize(cidr); size != v {
			t.Errorf("%s should have a range size of %d, got %d", k, v, size)
		}
	}
}

func TestSnapshot(t *testing.T) {
	_, cidr, err := net.ParseCIDR("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}
	r := NewCIDRRange(cidr)
	ip := []net.IP{}
	for i := 0; i < 10; i++ {
		n, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		ip = append(ip, n)
	}

	var dst api.RangeAllocation
	err = r.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	_, network, err := net.ParseCIDR(dst.Range)
	if err != nil {
		t.Fatal(err)
	}

	if !network.IP.Equal(cidr.IP) || network.Mask.String() != cidr.Mask.String() {
		t.Fatalf("mismatched networks: %s : %s", network, cidr)
	}

	_, otherCidr, err := net.ParseCIDR("192.168.2.0/24")
	if err != nil {
		t.Fatal(err)
	}
	other := NewCIDRRange(otherCidr)
	if err := r.Restore(otherCidr, dst.Data); err != ErrMismatchedNetwork {
		t.Fatal(err)
	}
	other = NewCIDRRange(network)
	if err := other.Restore(network, dst.Data); err != nil {
		t.Fatal(err)
	}

	for _, n := range ip {
		if !other.Has(n) {
			t.Errorf("restored range does not have %s", n)
		}
	}
	if other.Free() != r.Free() {
		t.Errorf("counts do not match: %d", other.Free())
	}
}
