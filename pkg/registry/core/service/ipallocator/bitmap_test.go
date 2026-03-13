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
	"fmt"
	"net"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/metrics/testutil"
	api "k8s.io/kubernetes/pkg/apis/core"
	netutils "k8s.io/utils/net"
)

func TestAllocate(t *testing.T) {
	testCases := []struct {
		name             string
		cidr             string
		family           api.IPFamily
		free             int
		released         string
		outOfRange       []string
		alreadyAllocated string
	}{
		{
			name:     "IPv4",
			cidr:     "192.168.1.0/24",
			family:   api.IPv4Protocol,
			free:     254,
			released: "192.168.1.5",
			outOfRange: []string{
				"192.168.0.1",   // not in 192.168.1.0/24
				"192.168.1.0",   // reserved (base address)
				"192.168.1.255", // reserved (broadcast address)
				"192.168.2.2",   // not in 192.168.1.0/24
			},
			alreadyAllocated: "192.168.1.1",
		},
		{
			name:     "IPv4 large",
			cidr:     "10.0.0.0/15",
			family:   api.IPv4Protocol,
			free:     131070,
			released: "10.0.0.5",
			outOfRange: []string{
				"10.0.0.0",      // reserved (base address)
				"10.15.255.255", // reserved (broadcast address)
				"10.255.255.2",  // not in range
			},
			alreadyAllocated: "10.0.0.1",
		},
		{
			name:     "IPv6",
			cidr:     "2001:db8:1::/48",
			family:   api.IPv6Protocol,
			free:     65535,
			released: "2001:db8:1::5",
			outOfRange: []string{
				"2001:db8::1",     // not in 2001:db8:1::/48
				"2001:db8:1::",    // reserved (base address)
				"2001:db8:1::1:0", // not in the low 16 bits of 2001:db8:1::/48
				"2001:db8:2::2",   // not in 2001:db8:1::/48
			},
			alreadyAllocated: "2001:db8:1::1",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, cidr, err := netutils.ParseCIDRSloppy(tc.cidr)
			if err != nil {
				t.Fatal(err)
			}
			r, err := NewInMemory(cidr)
			if err != nil {
				t.Fatal(err)
			}
			if f := r.Free(); f != tc.free {
				t.Errorf("[%s] wrong free: expected %d, got %d", tc.name, tc.free, f)
			}

			rCIDR := r.CIDR()
			if rCIDR.String() != tc.cidr {
				t.Errorf("[%s] wrong CIDR: expected %v, got %v", tc.name, tc.cidr, rCIDR.String())
			}

			if r.IPFamily() != tc.family {
				t.Errorf("[%s] wrong IP family: expected %v, got %v", tc.name, tc.family, r.IPFamily())
			}

			if f := r.Used(); f != 0 {
				t.Errorf("[%s]: wrong used: expected %d, got %d", tc.name, 0, f)
			}
			found := sets.NewString()
			count := 0
			for r.Free() > 0 {
				ip, err := r.AllocateNext()
				if err != nil {
					t.Fatalf("[%s] error @ %d: %v", tc.name, count, err)
				}
				count++
				if !cidr.Contains(ip) {
					t.Fatalf("[%s] allocated %s which is outside of %s", tc.name, ip, cidr)
				}
				if found.Has(ip.String()) {
					t.Fatalf("[%s] allocated %s twice @ %d", tc.name, ip, count)
				}
				found.Insert(ip.String())
			}
			if _, err := r.AllocateNext(); err != ErrFull {
				t.Fatal(err)
			}

			released := netutils.ParseIPSloppy(tc.released)
			if err := r.Release(released); err != nil {
				t.Fatal(err)
			}
			if f := r.Free(); f != 1 {
				t.Errorf("[%s] wrong free: expected %d, got %d", tc.name, 1, f)
			}
			if f := r.Used(); f != (tc.free - 1) {
				t.Errorf("[%s] wrong free: expected %d, got %d", tc.name, tc.free-1, f)
			}
			ip, err := r.AllocateNext()
			if err != nil {
				t.Fatal(err)
			}
			if !released.Equal(ip) {
				t.Errorf("[%s] unexpected %s : %s", tc.name, ip, released)
			}

			if err := r.Release(released); err != nil {
				t.Fatal(err)
			}
			for _, outOfRange := range tc.outOfRange {
				err = r.Allocate(netutils.ParseIPSloppy(outOfRange))
				if _, ok := err.(*ErrNotInRange); !ok {
					t.Fatal(err)
				}
			}
			if err := r.Allocate(netutils.ParseIPSloppy(tc.alreadyAllocated)); err != ErrAllocated {
				t.Fatal(err)
			}
			if f := r.Free(); f != 1 {
				t.Errorf("[%s] wrong free: expected %d, got %d", tc.name, 1, f)
			}
			if f := r.Used(); f != (tc.free - 1) {
				t.Errorf("[%s] wrong free: expected %d, got %d", tc.name, tc.free-1, f)
			}
			if err := r.Allocate(released); err != nil {
				t.Fatal(err)
			}
			if f := r.Free(); f != 0 {
				t.Errorf("[%s] wrong free: expected %d, got %d", tc.name, 0, f)
			}
			if f := r.Used(); f != tc.free {
				t.Errorf("[%s] wrong free: expected %d, got %d", tc.name, tc.free, f)
			}
		})
	}
}

func TestAllocateTiny(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.0/32")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewInMemory(cidr)
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

func TestAllocateReserved(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.0/25")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewInMemory(cidr)
	if err != nil {
		t.Fatal(err)
	}
	// allocate all addresses on the dynamic block
	// subnet /25 = 128 ; dynamic block size is min(max(16,128/16),256) = 16
	dynamicOffset := calculateRangeOffset(cidr)
	dynamicBlockSize := r.max - dynamicOffset
	for i := 0; i < dynamicBlockSize; i++ {
		if _, err := r.AllocateNext(); err != nil {
			t.Errorf("Unexpected error trying to allocate: %v", err)
		}
	}
	for i := dynamicOffset; i < r.max; i++ {
		ip := fmt.Sprintf("192.168.1.%d", i+1)
		if !r.Has(netutils.ParseIPSloppy(ip)) {
			t.Errorf("IP %s expected to be allocated", ip)
		}
	}
	if f := r.Free(); f != dynamicOffset {
		t.Errorf("expected %d free addresses, got %d", dynamicOffset, f)
	}
	// allocate all addresses on the static block
	for i := 0; i < dynamicOffset; i++ {
		ip := fmt.Sprintf("192.168.1.%d", i+1)
		if err := r.Allocate(netutils.ParseIPSloppy(ip)); err != nil {
			t.Errorf("Unexpected error trying to allocate IP %s: %v", ip, err)
		}
	}
	if f := r.Free(); f != 0 {
		t.Errorf("expected free equal to 0 got: %d", f)
	}
	// release one address in the allocated block and another a new one randomly
	if err := r.Release(netutils.ParseIPSloppy("192.168.1.10")); err != nil {
		t.Fatalf("Unexpected error trying to release ip 192.168.1.10: %v", err)
	}
	if _, err := r.AllocateNext(); err != nil {
		t.Error(err)
	}
	if f := r.Free(); f != 0 {
		t.Errorf("expected free equal to 0 got: %d", f)
	}
}

func TestAllocateSmall(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.240/30")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewInMemory(cidr)
	if err != nil {
		t.Fatal(err)
	}
	if f := r.Free(); f != 2 {
		t.Errorf("expected free equal to 2 got: %d", f)
	}
	found := sets.NewString()
	for i := 0; i < 2; i++ {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		if found.Has(ip.String()) {
			t.Fatalf("address %s has been already allocated", ip)
		}
		found.Insert(ip.String())
	}
	for s := range found {
		if !r.Has(netutils.ParseIPSloppy(s)) {
			t.Fatalf("missing: %s", s)
		}
		if err := r.Allocate(netutils.ParseIPSloppy(s)); err != ErrAllocated {
			t.Fatal(err)
		}
	}
	for i := 0; i < 100; i++ {
		if _, err := r.AllocateNext(); err != ErrFull {
			t.Fatalf("suddenly became not-full: %#v", r)
		}
	}

	if f := r.Free(); f != 0 {
		t.Errorf("expected free equal to 0 got: %d", f)
	}

	if r.max != 2 {
		t.Fatalf("expected range equal to 2, got: %v", r)
	}
}

func TestForEach(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.0/24")
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
		r, err := NewInMemory(cidr)
		if err != nil {
			t.Fatal(err)
		}
		for ips := range tc {
			ip := netutils.ParseIPSloppy(ips)
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
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewInMemory(cidr)
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

	_, network, err := netutils.ParseCIDRSloppy(dst.Range)
	if err != nil {
		t.Fatal(err)
	}

	if !network.IP.Equal(cidr.IP) || network.Mask.String() != cidr.Mask.String() {
		t.Fatalf("mismatched networks: %s : %s", network, cidr)
	}

	_, otherCidr, err := netutils.ParseCIDRSloppy("192.168.2.0/24")
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewInMemory(otherCidr)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Restore(otherCidr, dst.Data); err != ErrMismatchedNetwork {
		t.Fatal(err)
	}
	other, err := NewInMemory(network)
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
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.0.0/24")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewInMemory(cidr)
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

func TestClusterIPMetrics(t *testing.T) {
	clearMetrics()
	// create IPv4 allocator
	cidrIPv4 := "10.0.0.0/24"
	_, clusterCIDRv4, _ := netutils.ParseCIDRSloppy(cidrIPv4)
	a, err := NewInMemory(clusterCIDRv4)
	if err != nil {
		t.Fatalf("unexpected error creating CidrSet: %v", err)
	}
	a.EnableMetrics()
	// create IPv6 allocator
	cidrIPv6 := "2001:db8::/112"
	_, clusterCIDRv6, _ := netutils.ParseCIDRSloppy(cidrIPv6)
	b, err := NewInMemory(clusterCIDRv6)
	b.EnableMetrics()
	if err != nil {
		t.Fatalf("unexpected error creating CidrSet: %v", err)
	}

	// Check initial state
	em := testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, cidrIPv4, em)
	em = testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, cidrIPv6, em)

	// allocate 2 IPv4 addresses
	found := sets.NewString()
	for i := 0; i < 2; i++ {
		ip, err := a.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		if found.Has(ip.String()) {
			t.Fatalf("already reserved: %s", ip)
		}
		found.Insert(ip.String())
	}

	em = testMetrics{
		free:      252,
		used:      2,
		allocated: 2,
		errors:    0,
	}
	expectMetrics(t, cidrIPv4, em)

	// try to allocate the same IP addresses
	for s := range found {
		if !a.Has(netutils.ParseIPSloppy(s)) {
			t.Fatalf("missing: %s", s)
		}
		if err := a.Allocate(netutils.ParseIPSloppy(s)); err != ErrAllocated {
			t.Fatal(err)
		}
	}
	em = testMetrics{
		free:      252,
		used:      2,
		allocated: 2,
		errors:    2,
	}
	expectMetrics(t, cidrIPv4, em)

	// release the addresses allocated
	for s := range found {
		if !a.Has(netutils.ParseIPSloppy(s)) {
			t.Fatalf("missing: %s", s)
		}
		if err := a.Release(netutils.ParseIPSloppy(s)); err != nil {
			t.Fatal(err)
		}
	}
	em = testMetrics{
		free:      254,
		used:      0,
		allocated: 2,
		errors:    2,
	}
	expectMetrics(t, cidrIPv4, em)

	// allocate 264 addresses for each allocator
	// the full range and 10 more (254 + 10 = 264) for IPv4
	for i := 0; i < 264; i++ {
		a.AllocateNext()
		b.AllocateNext()
	}
	em = testMetrics{
		free:      0,
		used:      254,
		allocated: 256, // this is a counter, we already had 2 allocations and we did 254 more
		errors:    12,
	}
	expectMetrics(t, cidrIPv4, em)
	em = testMetrics{
		free:      65271, // IPv6 clusterIP range is capped to 2^16 and consider the broadcast address as valid
		used:      264,
		allocated: 264,
		errors:    0,
	}
	expectMetrics(t, cidrIPv6, em)
}

func TestClusterIPAllocatedMetrics(t *testing.T) {
	clearMetrics()
	// create IPv4 allocator
	cidrIPv4 := "10.0.0.0/25"
	_, clusterCIDRv4, _ := netutils.ParseCIDRSloppy(cidrIPv4)
	a, err := NewInMemory(clusterCIDRv4)
	if err != nil {
		t.Fatalf("unexpected error creating CidrSet: %v", err)
	}
	a.EnableMetrics()

	em := testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, cidrIPv4, em)

	// allocate 2 dynamic IPv4 addresses
	found := sets.NewString()
	for i := 0; i < 2; i++ {
		ip, err := a.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		if found.Has(ip.String()) {
			t.Fatalf("already reserved: %s", ip)
		}
		found.Insert(ip.String())
	}

	dynamic_allocated, err := testutil.GetCounterMetricValue(clusterIPAllocations.WithLabelValues(cidrIPv4, "dynamic"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocations.Name, err)
	}
	if dynamic_allocated != 2 {
		t.Fatalf("Expected 2 received %f", dynamic_allocated)
	}

	// try to allocate the same IP addresses
	for s := range found {
		if !a.Has(netutils.ParseIPSloppy(s)) {
			t.Fatalf("missing: %s", s)
		}
		if err := a.Allocate(netutils.ParseIPSloppy(s)); err != ErrAllocated {
			t.Fatal(err)
		}
	}

	static_errors, err := testutil.GetCounterMetricValue(clusterIPAllocationErrors.WithLabelValues(cidrIPv4, "static"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocationErrors.Name, err)
	}
	if static_errors != 2 {
		t.Fatalf("Expected 2 received %f", dynamic_allocated)
	}
}

func TestMetricsDisabled(t *testing.T) {
	// create metrics enabled allocator
	cidrIPv4 := "10.0.0.0/24"
	_, clusterCIDRv4, _ := netutils.ParseCIDRSloppy(cidrIPv4)
	a, err := NewInMemory(clusterCIDRv4)
	if err != nil {
		t.Fatalf("unexpected error creating CidrSet: %v", err)
	}
	a.EnableMetrics()

	// create metrics disabled allocator with same CIDR
	// this metrics should be ignored
	b, err := NewInMemory(clusterCIDRv4)
	if err != nil {
		t.Fatalf("unexpected error creating CidrSet: %v", err)
	}

	// Check initial state
	em := testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, cidrIPv4, em)

	// allocate in metrics enabled allocator
	for i := 0; i < 100; i++ {
		_, err := a.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
	}
	em = testMetrics{
		free:      154,
		used:      100,
		allocated: 100,
		errors:    0,
	}
	expectMetrics(t, cidrIPv4, em)

	// allocate in metrics disabled allocator
	for i := 0; i < 200; i++ {
		_, err := b.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
	}
	// the metrics should not be changed
	expectMetrics(t, cidrIPv4, em)
}

// Metrics helpers
func clearMetrics() {
	clusterIPAllocated.Reset()
	clusterIPAvailable.Reset()
	clusterIPAllocations.Reset()
	clusterIPAllocationErrors.Reset()
}

type testMetrics struct {
	free      float64
	used      float64
	allocated float64
	errors    float64
}

func expectMetrics(t *testing.T, label string, em testMetrics) {
	var m testMetrics
	var err error
	m.free, err = testutil.GetGaugeMetricValue(clusterIPAvailable.WithLabelValues(label))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAvailable.Name, err)
	}
	m.used, err = testutil.GetGaugeMetricValue(clusterIPAllocated.WithLabelValues(label))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocated.Name, err)
	}
	static_allocated, err := testutil.GetCounterMetricValue(clusterIPAllocations.WithLabelValues(label, "static"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocations.Name, err)
	}
	static_errors, err := testutil.GetCounterMetricValue(clusterIPAllocationErrors.WithLabelValues(label, "static"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocationErrors.Name, err)
	}
	dynamic_allocated, err := testutil.GetCounterMetricValue(clusterIPAllocations.WithLabelValues(label, "dynamic"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocations.Name, err)
	}
	dynamic_errors, err := testutil.GetCounterMetricValue(clusterIPAllocationErrors.WithLabelValues(label, "dynamic"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocationErrors.Name, err)
	}

	m.allocated = static_allocated + dynamic_allocated
	m.errors = static_errors + dynamic_errors

	if m != em {
		t.Fatalf("metrics error: expected %v, received %v", em, m)
	}
}

func TestDryRun(t *testing.T) {
	testCases := []struct {
		name   string
		cidr   string
		family api.IPFamily
	}{{
		name:   "IPv4",
		cidr:   "192.168.1.0/24",
		family: api.IPv4Protocol,
	}, {
		name:   "IPv6",
		cidr:   "2001:db8:1::/48",
		family: api.IPv6Protocol,
	}}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, cidr, err := netutils.ParseCIDRSloppy(tc.cidr)
			if err != nil {
				t.Fatalf("unexpected failure: %v", err)
			}
			r, err := NewInMemory(cidr)
			if err != nil {
				t.Fatalf("unexpected failure: %v", err)
			}

			baseUsed := r.Used()

			rCIDR := r.DryRun().CIDR()
			if rCIDR.String() != tc.cidr {
				t.Errorf("allocator returned a different cidr")
			}

			if r.DryRun().IPFamily() != tc.family {
				t.Errorf("allocator returned wrong IP family")
			}

			expectUsed := func(t *testing.T, r *Range, expect int) {
				t.Helper()
				if u := r.Used(); u != expect {
					t.Errorf("unexpected used count: got %d, wanted %d", u, expect)
				}
			}
			expectUsed(t, r, baseUsed)

			err = r.DryRun().Allocate(netutils.AddIPOffset(netutils.BigForIP(cidr.IP), 1))
			if err != nil {
				t.Fatalf("unexpected failure: %v", err)
			}
			expectUsed(t, r, baseUsed)

			_, err = r.DryRun().AllocateNext()
			if err != nil {
				t.Fatalf("unexpected failure: %v", err)
			}
			expectUsed(t, r, baseUsed)

			if err := r.DryRun().Release(cidr.IP); err != nil {
				t.Fatalf("unexpected failure: %v", err)
			}
			expectUsed(t, r, baseUsed)
		})
	}
}

func Test_calculateRangeOffset(t *testing.T) {
	// default $min = 16, $max = 256 and $step = 16.
	tests := []struct {
		name string
		cidr string
		want int
	}{
		{
			name: "full mask IPv4",
			cidr: "192.168.1.1/32",
			want: 0,
		},
		{
			name: "full mask IPv6",
			cidr: "fd00::1/128",
			want: 0,
		},
		{
			name: "very small mask IPv4",
			cidr: "192.168.1.1/30",
			want: 0,
		},
		{
			name: "very small mask IPv6",
			cidr: "fd00::1/126",
			want: 0,
		},
		{
			name: "small mask IPv4",
			cidr: "192.168.1.1/28",
			want: 0,
		},
		{
			name: "small mask IPv4",
			cidr: "192.168.1.1/27",
			want: 16,
		},
		{
			name: "small mask IPv6",
			cidr: "fd00::1/124",
			want: 0,
		},
		{
			name: "small mask IPv6",
			cidr: "fd00::1/122",
			want: 16,
		},
		{
			name: "medium mask IPv4",
			cidr: "192.168.1.1/22",
			want: 64,
		},
		{
			name: "medium mask IPv6",
			cidr: "fd00::1/118",
			want: 64,
		},
		{
			name: "large mask IPv4",
			cidr: "192.168.1.1/8",
			want: 256,
		},
		{
			name: "large mask IPv6",
			cidr: "fd00::1/12",
			want: 256,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			_, cidr, err := netutils.ParseCIDRSloppy(tt.cidr)
			if err != nil {
				t.Fatalf("Unexpected error parsing CIDR %s: %v", tt.cidr, err)
			}
			if got := calculateRangeOffset(cidr); got != tt.want {
				t.Errorf("DynamicRangeOffset() = %v, want %v", got, tt.want)
			}
		})
	}
}

// cpu: Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz
// BenchmarkAllocateNextIPv4
// BenchmarkAllocateNextIPv4-24    	 1175304	       870.9 ns/op	    1337 B/op	      11 allocs/op
func BenchmarkAllocateNextIPv4Size1048574(b *testing.B) {
	_, cidr, err := netutils.ParseCIDRSloppy("10.0.0.0/12")
	if err != nil {
		b.Fatal(err)
	}
	r, err := NewInMemory(cidr)
	if err != nil {
		b.Fatal(err)
	}
	for n := 0; n < b.N; n++ {
		r.AllocateNext()
	}
}

// This is capped to 65535
// cpu: Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz
// BenchmarkAllocateNextIPv6
// BenchmarkAllocateNextIPv6-24    	 5779431	       194.0 ns/op	      18 B/op	       2 allocs/op
func BenchmarkAllocateNextIPv6Size65535(b *testing.B) {
	_, cidr, err := netutils.ParseCIDRSloppy("fd00::/24")
	if err != nil {
		b.Fatal(err)
	}
	r, err := NewInMemory(cidr)
	if err != nil {
		b.Fatal(err)
	}
	for n := 0; n < b.N; n++ {
		r.AllocateNext()
	}
}
