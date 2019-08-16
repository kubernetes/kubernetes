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

	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestAllocate(t *testing.T) {
	testCases := []struct {
		name             string
		cidr             string
		free             int
		released         string
		outOfRange1      string
		outOfRange2      string
		outOfRange3      string
		alreadyAllocated string
	}{
		{
			name:             "IPv4",
			cidr:             "192.168.1.0/24",
			free:             254,
			released:         "192.168.1.5",
			outOfRange1:      "192.168.0.1",
			outOfRange2:      "192.168.1.0",
			outOfRange3:      "192.168.1.255",
			alreadyAllocated: "192.168.1.1",
		},
		{
			name:             "IPv6",
			cidr:             "2001:db8:1::/48",
			free:             65534,
			released:         "2001:db8:1::5",
			outOfRange1:      "2001:db8::1",
			outOfRange2:      "2001:db8:1::",
			outOfRange3:      "2001:db8:1::ffff",
			alreadyAllocated: "2001:db8:1::1",
		},
	}
	for _, tc := range testCases {
		_, cidr, err := net.ParseCIDR(tc.cidr)
		if err != nil {
			t.Fatal(err)
		}
		r, err := NewCIDRRange(cidr)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("base: %v", r.base.Bytes())
		if f := r.Free(); f != tc.free {
			t.Errorf("Test %s unexpected free %d", tc.name, f)
		}
		if f := r.Used(); f != 0 {
			t.Errorf("Test %s unexpected used %d", tc.name, f)
		}
		found := sets.NewString()
		count := 0
		for r.Free() > 0 {
			ip, err := r.AllocateNext()
			if err != nil {
				t.Fatalf("Test %s error @ %d: %v", tc.name, count, err)
			}
			count++
			if !cidr.Contains(ip) {
				t.Fatalf("Test %s allocated %s which is outside of %s", tc.name, ip, cidr)
			}
			if found.Has(ip.String()) {
				t.Fatalf("Test %s allocated %s twice @ %d", tc.name, ip, count)
			}
			found.Insert(ip.String())
		}
		if _, err := r.AllocateNext(); err != ErrFull {
			t.Fatal(err)
		}

		released := net.ParseIP(tc.released)
		if err := r.Release(released); err != nil {
			t.Fatal(err)
		}
		if f := r.Free(); f != 1 {
			t.Errorf("Test %s unexpected free %d", tc.name, f)
		}
		if f := r.Used(); f != (tc.free - 1) {
			t.Errorf("Test %s unexpected free %d", tc.name, f)
		}
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		if !released.Equal(ip) {
			t.Errorf("Test %s unexpected %s : %s", tc.name, ip, released)
		}

		if err := r.Release(released); err != nil {
			t.Fatal(err)
		}
		err = r.Allocate(net.ParseIP(tc.outOfRange1))
		if _, ok := err.(*ErrNotInRange); !ok {
			t.Fatal(err)
		}
		if err := r.Allocate(net.ParseIP(tc.alreadyAllocated)); err != ErrAllocated {
			t.Fatal(err)
		}
		err = r.Allocate(net.ParseIP(tc.outOfRange2))
		if _, ok := err.(*ErrNotInRange); !ok {
			t.Fatal(err)
		}
		err = r.Allocate(net.ParseIP(tc.outOfRange3))
		if _, ok := err.(*ErrNotInRange); !ok {
			t.Fatal(err)
		}
		if f := r.Free(); f != 1 {
			t.Errorf("Test %s unexpected free %d", tc.name, f)
		}
		if f := r.Used(); f != (tc.free - 1) {
			t.Errorf("Test %s unexpected free %d", tc.name, f)
		}
		if err := r.Allocate(released); err != nil {
			t.Fatal(err)
		}
		if f := r.Free(); f != 0 {
			t.Errorf("Test %s unexpected free %d", tc.name, f)
		}
		if f := r.Used(); f != tc.free {
			t.Errorf("Test %s unexpected free %d", tc.name, f)
		}
	}
}

func TestAllocateTiny(t *testing.T) {
	_, cidr, err := net.ParseCIDR("192.168.1.0/32")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewCIDRRange(cidr)
	if err != nil {
		t.Fatal(err)
	}
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
	r, err := NewCIDRRange(cidr)
	if err != nil {
		t.Fatal(err)
	}
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
	testCases := []struct {
		name  string
		cidr  string
		addrs int64
	}{
		{
			name:  "supported IPv4 cidr",
			cidr:  "192.168.1.0/24",
			addrs: 256,
		},
		{
			name:  "unsupported IPv4 cidr",
			cidr:  "192.168.1.0/1",
			addrs: 0,
		},
		{
			name:  "supported IPv6 cidr",
			cidr:  "2001:db8::/48",
			addrs: 65536,
		},
		{
			name:  "unsupported IPv6 mask",
			cidr:  "2001:db8::/1",
			addrs: 0,
		},
	}

	for _, tc := range testCases {
		_, cidr, err := net.ParseCIDR(tc.cidr)
		if err != nil {
			t.Errorf("failed to parse cidr for test %s, unexpected error: '%s'", tc.name, err)
		}
		if size := RangeSize(cidr); size != tc.addrs {
			t.Errorf("test %s failed. %s should have a range size of %d, got %d",
				tc.name, tc.cidr, tc.addrs, size)
		}
	}
}

func TestForEach(t *testing.T) {
	_, cidr, err := net.ParseCIDR("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}

	testCases := []sets.String{
		sets.NewString(),
		sets.NewString("192.168.1.1"),
		sets.NewString("192.168.1.1", "192.168.1.254"),
		sets.NewString("192.168.1.1", "192.168.1.128", "192.168.1.254"),
	}

	for i, tc := range testCases {
		r, err := NewCIDRRange(cidr)
		if err != nil {
			t.Fatal(err)
		}
		for ips := range tc {
			ip := net.ParseIP(ips)
			if err := r.Allocate(ip); err != nil {
				t.Errorf("[%d] error allocating IP %v: %v", i, ip, err)
			}
			if !r.Has(ip) {
				t.Errorf("[%d] expected IP %v allocated", i, ip)
			}
		}
		calls := sets.NewString()
		r.ForEach(func(ip net.IP) {
			calls.Insert(ip.String())
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
	_, cidr, err := net.ParseCIDR("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewCIDRRange(cidr)
	if err != nil {
		t.Fatal(err)
	}
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
	other, err := NewCIDRRange(otherCidr)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Restore(otherCidr, dst.Data); err != ErrMismatchedNetwork {
		t.Fatal(err)
	}
	other, err = NewCIDRRange(network)
	if err != nil {
		t.Fatal(err)
	}
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

func TestNewFromSnapshot(t *testing.T) {
	_, cidr, err := net.ParseCIDR("192.168.0.0/24")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewCIDRRange(cidr)
	if err != nil {
		t.Fatal(err)
	}
	allocated := []net.IP{}
	for i := 0; i < 128; i++ {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		allocated = append(allocated, ip)
	}

	snapshot := api.RangeAllocation{}
	if err = r.Snapshot(&snapshot); err != nil {
		t.Fatal(err)
	}

	r, err = NewFromSnapshot(&snapshot)
	if err != nil {
		t.Fatal(err)
	}

	if x := r.Free(); x != 126 {
		t.Fatalf("expected 126 free IPs, got %d", x)
	}
	if x := r.Used(); x != 128 {
		t.Fatalf("expected 128 used IPs, got %d", x)
	}

	for _, ip := range allocated {
		if !r.Has(ip) {
			t.Fatalf("expected IP to be allocated, but it was not")
		}
	}
}
