/*
Copyright 2022 The Kubernetes Authors.

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
	"math"
	"net"
	"net/netip"
	"reflect"
	"testing"
	"time"

	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/component-base/metrics/testutil"
	api "k8s.io/kubernetes/pkg/apis/core"
	netutils "k8s.io/utils/net"
)

func newTestAllocator(cidr *net.IPNet) (*Allocator, error) {
	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, 0*time.Second)
	ipInformer := informerFactory.Networking().V1().IPAddresses()
	ipStore := ipInformer.Informer().GetIndexer()

	client.PrependReactor("create", "ipaddresses", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		ip := action.(k8stesting.CreateAction).GetObject().(*networkingv1.IPAddress)
		_, exists, err := ipStore.GetByKey(ip.Name)
		if exists && err != nil {
			return false, nil, fmt.Errorf("ip already exist")
		}
		ip.Generation = 1
		err = ipStore.Add(ip)
		return false, ip, err
	}))
	client.PrependReactor("delete", "ipaddresses", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		name := action.(k8stesting.DeleteAction).GetName()
		obj, exists, err := ipStore.GetByKey(name)
		ip := &networkingv1.IPAddress{}
		if exists && err == nil {
			ip = obj.(*networkingv1.IPAddress)
			err = ipStore.Delete(ip)
		}
		return false, ip, err
	}))

	c, err := NewIPAllocator(cidr, client.NetworkingV1(), ipInformer)
	if err != nil {
		return nil, err
	}
	c.ipAddressSynced = func() bool { return true }
	return c, nil
}

func TestAllocateIPAllocator(t *testing.T) {
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
			name:     "IPv6",
			cidr:     "2001:db8:1::/116",
			free:     4095,
			released: "2001:db8:1::5",
			outOfRange: []string{
				"2001:db8::1",   // not in 2001:db8:1::/48
				"2001:db8:1::",  // reserved (base address)
				"2001:db8:2::2", // not in 2001:db8:1::/48
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
			r, err := newTestAllocator(cidr)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Destroy()
			if f := r.Free(); f != tc.free {
				t.Errorf("[%s] wrong free: expected %d, got %d", tc.name, tc.free, f)
			}

			if f := r.Used(); f != 0 {
				t.Errorf("[%s]: wrong used: expected %d, got %d", tc.name, 0, f)
			}
			found := sets.NewString()
			count := 0
			for r.Free() > 0 {
				ip, err := r.AllocateNext()
				if err != nil {
					t.Fatalf("[%s] error @ free: %d used: %d count: %d: %v", tc.name, r.Free(), r.Used(), count, err)
				}
				count++
				//if !cidr.Contains(ip) {
				//	t.Fatalf("[%s] allocated %s which is outside of %s", tc.name, ip, cidr)
				//}
				if found.Has(ip.String()) {
					t.Fatalf("[%s] allocated %s twice @ %d", tc.name, ip, count)
				}
				found.Insert(ip.String())
			}
			if _, err := r.AllocateNext(); err == nil {
				t.Fatal(err)
			}

			if !found.Has(tc.released) {
				t.Fatalf("not allocated address to be releases %s found %d", tc.released, len(found))
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
				if err == nil {
					t.Fatalf("unexpacted allocating of %s", outOfRange)
				}
			}
			if err := r.Allocate(netutils.ParseIPSloppy(tc.alreadyAllocated)); err == nil {
				t.Fatalf("unexpected allocation of %s", tc.alreadyAllocated)
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

func TestAllocateTinyIPAllocator(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.0/32")
	if err != nil {
		t.Fatal(err)
	}

	r, err := newTestAllocator(cidr)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()

	if f := r.Free(); f != 0 {
		t.Errorf("free: %d", f)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Error(err)
	}
}

func TestAllocateReservedIPAllocator(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.0/25")
	if err != nil {
		t.Fatal(err)
	}
	r, err := newTestAllocator(cidr)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()
	// allocate all addresses on the dynamic block
	// subnet /25 = 128 ; dynamic block size is min(max(16,128/16),256) = 16
	dynamicOffset := calculateRangeOffset(cidr)
	dynamicBlockSize := int(r.size) - dynamicOffset
	for i := 0; i < dynamicBlockSize; i++ {
		_, err := r.AllocateNext()
		if err != nil {
			t.Errorf("Unexpected error trying to allocate: %v", err)
		}
	}
	for i := dynamicOffset; i < int(r.size); i++ {
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

func TestAllocateSmallIPAllocator(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.240/30")
	if err != nil {
		t.Fatal(err)
	}
	r, err := newTestAllocator(cidr)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()

	if f := r.Free(); f != 2 {
		t.Errorf("expected free equal to 2 got: %d", f)
	}
	found := sets.NewString()
	for i := 0; i < 2; i++ {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error allocating %s try %d : %v", ip, i, err)
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
		if err := r.Allocate(netutils.ParseIPSloppy(s)); err == nil {
			t.Fatal(err)
		}
	}
	if f := r.Free(); f != 0 {
		t.Errorf("expected free equal to 0 got: %d", f)
	}

	for i := 0; i < 100; i++ {
		if ip, err := r.AllocateNext(); err == nil {
			t.Fatalf("suddenly became not-full: %s", ip.String())
		}
	}

}

func TestForEachIPAllocator(t *testing.T) {
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
		r, err := newTestAllocator(cidr)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Destroy()

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

func TestIPAllocatorClusterIPMetrics(t *testing.T) {
	clearMetrics()
	// create IPv4 allocator
	cidrIPv4 := "10.0.0.0/24"
	_, clusterCIDRv4, _ := netutils.ParseCIDRSloppy(cidrIPv4)
	a, err := newTestAllocator(clusterCIDRv4)
	if err != nil {
		t.Fatal(err)
	}
	a.EnableMetrics()
	// create IPv6 allocator
	cidrIPv6 := "2001:db8::/112"
	_, clusterCIDRv6, _ := netutils.ParseCIDRSloppy(cidrIPv6)
	b, err := newTestAllocator(clusterCIDRv6)
	if err != nil {
		t.Fatalf("unexpected error creating CidrSet: %v", err)
	}
	b.EnableMetrics()

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

func TestIPAllocatorClusterIPAllocatedMetrics(t *testing.T) {
	clearMetrics()
	// create IPv4 allocator
	cidrIPv4 := "10.0.0.0/25"
	_, clusterCIDRv4, _ := netutils.ParseCIDRSloppy(cidrIPv4)
	a, err := newTestAllocator(clusterCIDRv4)
	if err != nil {
		t.Fatal(err)
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

func Test_addOffsetAddress(t *testing.T) {
	tests := []struct {
		name    string
		address netip.Addr
		offset  uint64
		want    netip.Addr
	}{
		{
			name:    "IPv4 offset 0",
			address: netip.MustParseAddr("192.168.0.0"),
			offset:  0,
			want:    netip.MustParseAddr("192.168.0.0"),
		},
		{
			name:    "IPv4 offset 0 not nibble boundary",
			address: netip.MustParseAddr("192.168.0.11"),
			offset:  0,
			want:    netip.MustParseAddr("192.168.0.11"),
		},
		{
			name:    "IPv4 offset 1",
			address: netip.MustParseAddr("192.168.0.0"),
			offset:  1,
			want:    netip.MustParseAddr("192.168.0.1"),
		},
		{
			name:    "IPv4 offset 1 not nibble boundary",
			address: netip.MustParseAddr("192.168.0.11"),
			offset:  1,
			want:    netip.MustParseAddr("192.168.0.12"),
		},
		{
			name:    "IPv6 offset 1",
			address: netip.MustParseAddr("fd00:1:2:3::"),
			offset:  1,
			want:    netip.MustParseAddr("fd00:1:2:3::1"),
		},
		{
			name:    "IPv6 offset 1 not nibble boundary",
			address: netip.MustParseAddr("fd00:1:2:3::a"),
			offset:  1,
			want:    netip.MustParseAddr("fd00:1:2:3::b"),
		},
		{
			name:    "IPv4 offset last",
			address: netip.MustParseAddr("192.168.0.0"),
			offset:  255,
			want:    netip.MustParseAddr("192.168.0.255"),
		},
		{
			name:    "IPv6 offset last",
			address: netip.MustParseAddr("fd00:1:2:3::"),
			offset:  0x7FFFFFFFFFFFFFFF,
			want:    netip.MustParseAddr("fd00:1:2:3:7FFF:FFFF:FFFF:FFFF"),
		},
		{
			name:    "IPv4 offset middle",
			address: netip.MustParseAddr("192.168.0.0"),
			offset:  128,
			want:    netip.MustParseAddr("192.168.0.128"),
		},
		{
			name:    "IPv4 with leading zeros",
			address: netip.MustParseAddr("0.0.1.8"),
			offset:  138,
			want:    netip.MustParseAddr("0.0.1.146"),
		},
		{
			name:    "IPv6 with leading zeros",
			address: netip.MustParseAddr("00fc::1"),
			offset:  255,
			want:    netip.MustParseAddr("fc::100"),
		},
		{
			name:    "IPv6 offset 255",
			address: netip.MustParseAddr("2001:db8:1::101"),
			offset:  255,
			want:    netip.MustParseAddr("2001:db8:1::200"),
		},
		{
			name:    "IPv6 offset 1025",
			address: netip.MustParseAddr("fd00:1:2:3::"),
			offset:  1025,
			want:    netip.MustParseAddr("fd00:1:2:3::401"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := addOffsetAddress(tt.address, tt.offset)
			if !reflect.DeepEqual(got, tt.want) || err != nil {
				t.Errorf("offsetAddress() = %v, want %v", got, tt.want)
			}
			// double check to avoid mistakes on the hardcoded values
			// avoid large numbers or it will timeout the test
			if tt.offset < 2048 {
				want := tt.address
				var i uint64
				for i = 0; i < tt.offset; i++ {
					want = want.Next()
				}
				if !reflect.DeepEqual(got, tt.want) || err != nil {
					t.Errorf("offsetAddress() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func Test_broadcastAddress(t *testing.T) {
	tests := []struct {
		name   string
		subnet netip.Prefix
		want   netip.Addr
	}{
		{
			name:   "ipv4",
			subnet: netip.MustParsePrefix("192.168.0.0/24"),
			want:   netip.MustParseAddr("192.168.0.255"),
		},
		{
			name:   "ipv4 no nibble boundary",
			subnet: netip.MustParsePrefix("10.0.0.0/12"),
			want:   netip.MustParseAddr("10.15.255.255"),
		},
		{
			name:   "ipv6",
			subnet: netip.MustParsePrefix("fd00:1:2:3::/64"),
			want:   netip.MustParseAddr("fd00:1:2:3:FFFF:FFFF:FFFF:FFFF"),
		},
		{
			name:   "ipv6 00fc::/112",
			subnet: netip.MustParsePrefix("00fc::/112"),
			want:   netip.MustParseAddr("fc::ffff"),
		},
		{
			name:   "ipv6 fc00::/112",
			subnet: netip.MustParsePrefix("fc00::/112"),
			want:   netip.MustParseAddr("fc00::ffff"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := broadcastAddress(tt.subnet); !reflect.DeepEqual(got, tt.want) || err != nil {
				t.Errorf("broadcastAddress() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_hostsPerNetwork(t *testing.T) {
	testCases := []struct {
		name  string
		cidr  string
		addrs uint64
	}{
		{
			name:  "supported IPv4 cidr",
			cidr:  "192.168.1.0/24",
			addrs: 254,
		},
		{
			name:  "single IPv4 host",
			cidr:  "192.168.1.0/32",
			addrs: 0,
		},
		{
			name:  "small IPv4 cidr",
			cidr:  "192.168.1.0/31",
			addrs: 0,
		},
		{
			name:  "very large IPv4 cidr",
			cidr:  "0.0.0.0/1",
			addrs: math.MaxInt32 - 1,
		},
		{
			name:  "full IPv4 range",
			cidr:  "0.0.0.0/0",
			addrs: math.MaxUint32 - 1,
		},
		{
			name:  "supported IPv6 cidr",
			cidr:  "2001:db2::/112",
			addrs: 65535,
		},
		{
			name:  "single IPv6 host",
			cidr:  "2001:db8::/128",
			addrs: 0,
		},
		{
			name:  "small IPv6 cidr",
			cidr:  "2001:db8::/127",
			addrs: 1,
		},
		{
			name:  "largest IPv6 for Int64",
			cidr:  "2001:db8::/65",
			addrs: math.MaxInt64,
		},
		{
			name:  "largest IPv6 for Uint64",
			cidr:  "2001:db8::/64",
			addrs: math.MaxUint64,
		},
		{
			name:  "very large IPv6 cidr",
			cidr:  "2001:db8::/1",
			addrs: math.MaxUint64,
		},
	}

	for _, tc := range testCases {
		_, cidr, err := netutils.ParseCIDRSloppy(tc.cidr)
		if err != nil {
			t.Errorf("failed to parse cidr for test %s, unexpected error: '%s'", tc.name, err)
		}
		if size := hostsPerNetwork(cidr); size != tc.addrs {
			t.Errorf("test %s failed. %s should have a range size of %d, got %d",
				tc.name, tc.cidr, tc.addrs, size)
		}
	}
}

func Test_ipIterator(t *testing.T) {
	tests := []struct {
		name   string
		first  netip.Addr
		last   netip.Addr
		offset uint64
		want   []string
	}{
		{
			name:   "start from first address small range",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.2"),
			offset: 0,
			want:   []string{"192.168.0.1", "192.168.0.2"},
		}, {
			name:   "start from last address small range",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.2"),
			offset: 1,
			want:   []string{"192.168.0.2", "192.168.0.1"},
		}, {
			name:   "start from offset out of range address small range",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.2"),
			offset: 10,
			want:   []string{"192.168.0.1", "192.168.0.2"},
		}, {
			name:   "start from first address",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.7"),
			offset: 0,
			want:   []string{"192.168.0.1", "192.168.0.2", "192.168.0.3", "192.168.0.4", "192.168.0.5", "192.168.0.6", "192.168.0.7"},
		}, {
			name:   "start from middle address",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.7"),
			offset: 2,
			want:   []string{"192.168.0.3", "192.168.0.4", "192.168.0.5", "192.168.0.6", "192.168.0.7", "192.168.0.1", "192.168.0.2"},
		}, {
			name:   "start from last address",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.7"),
			offset: 6,
			want:   []string{"192.168.0.7", "192.168.0.1", "192.168.0.2", "192.168.0.3", "192.168.0.4", "192.168.0.5", "192.168.0.6"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := []string{}
			iterator := ipIterator(tt.first, tt.last, tt.offset)

			for {
				ip := iterator()
				if !ip.IsValid() {
					break
				}
				got = append(got, ip.String())
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ipIterator() = %v, want %v", got, tt.want)
			}
			// check the iterator is fully stopped
			for i := 0; i < 5; i++ {
				if ip := iterator(); ip.IsValid() {
					t.Errorf("iterator should not return more addresses: %v", ip)
				}
			}
		})
	}
}

func Test_ipIterator_Number(t *testing.T) {
	tests := []struct {
		name   string
		first  netip.Addr
		last   netip.Addr
		offset uint64
		want   uint64
	}{
		{
			name:   "start from first address small range",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.2"),
			offset: 0,
			want:   2,
		}, {
			name:   "start from last address small range",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.2"),
			offset: 1,
			want:   2,
		}, {
			name:   "start from offset out of range small range",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.2"),
			offset: 10,
			want:   2,
		}, {
			name:   "start from first address",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.7"),
			offset: 0,
			want:   7,
		}, {
			name:   "start from middle address",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.7"),
			offset: 2,
			want:   7,
		}, {
			name:   "start from last address",
			first:  netip.MustParseAddr("192.168.0.1"),
			last:   netip.MustParseAddr("192.168.0.7"),
			offset: 6,
			want:   7,
		}, {
			name:   "start from first address large range",
			first:  netip.MustParseAddr("2001:db8:1::101"),
			last:   netip.MustParseAddr("2001:db8:1::fff"),
			offset: 0,
			want:   3839,
		}, {
			name:   "start from address in the middle",
			first:  netip.MustParseAddr("2001:db8:1::101"),
			last:   netip.MustParseAddr("2001:db8:1::fff"),
			offset: 255,
			want:   3839,
		}, {
			name:   "start from last address",
			first:  netip.MustParseAddr("2001:db8:1::101"),
			last:   netip.MustParseAddr("2001:db8:1::fff"),
			offset: 3838,
			want:   3839,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got uint64
			iterator := ipIterator(tt.first, tt.last, tt.offset)

			for {
				ip := iterator()
				if !ip.IsValid() {
					break
				}
				got++
			}
			if got != tt.want {
				t.Errorf("ipIterator() = %d, want %d", got, tt.want)
			}
			// check the iterator is fully stopped
			for i := 0; i < 5; i++ {
				if ip := iterator(); ip.IsValid() {
					t.Errorf("iterator should not return more addresses: %v", ip)
				}
			}
		})
	}
}

func TestAllocateNextFC(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("fc::/112")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("CIDR %s", cidr)

	r, err := newTestAllocator(cidr)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()
	ip, err := r.AllocateNext()
	if err != nil {
		t.Fatalf("wrong ip %s : %v", ip, err)
	}
	t.Log(ip.String())
}

func BenchmarkIPAllocatorAllocateNextIPv4Size1048574(b *testing.B) {
	_, cidr, err := netutils.ParseCIDRSloppy("10.0.0.0/12")
	if err != nil {
		b.Fatal(err)
	}
	r, err := newTestAllocator(cidr)
	if err != nil {
		b.Fatal(err)
	}
	defer r.Destroy()

	for n := 0; n < b.N; n++ {
		r.AllocateNext()
	}
}

func BenchmarkIPAllocatorAllocateNextIPv6Size65535(b *testing.B) {
	_, cidr, err := netutils.ParseCIDRSloppy("fd00::/120")
	if err != nil {
		b.Fatal(err)
	}
	r, err := newTestAllocator(cidr)
	if err != nil {
		b.Fatal(err)
	}
	defer r.Destroy()

	for n := 0; n < b.N; n++ {
		r.AllocateNext()
	}
}
