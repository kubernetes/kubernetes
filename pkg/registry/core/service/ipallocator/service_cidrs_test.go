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
	"context"
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/netip"
	"reflect"
	"testing"
	"time"

	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/util/workqueue"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/iptree"
	netutils "k8s.io/utils/net"
)

func newTestAllocator(cidr *net.IPNet) (*Allocator, error) {
	svcCIDR := &networkingv1alpha1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: "default-cidr",
		},
	}
	isIPv6 := false
	if netutils.IsIPv4CIDR(cidr) {
		svcCIDR.Spec.IPv4 = cidr.String()
	} else {
		svcCIDR.Spec.IPv6 = cidr.String()
		isIPv6 = true
	}

	client := fake.NewSimpleClientset(svcCIDR)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	cidrInformer := informerFactory.Networking().V1alpha1().ServiceCIDRs()
	ipInformer := informerFactory.Networking().V1alpha1().IPAddresses()

	cidrStore := cidrInformer.Informer().GetStore()
	err := cidrStore.Add(svcCIDR)
	if err != nil {
		return nil, err
	}
	client.PrependReactor("create", "servicecidrs", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		cidr := action.(k8stesting.CreateAction).GetObject().(*networkingv1alpha1.ServiceCIDR)
		_, exists, err := cidrStore.GetByKey(cidr.Name)
		if exists && err != nil {
			return false, nil, fmt.Errorf("ip already exist")
		}
		cidr.Generation = 1
		err = cidrStore.Add(cidr)
		return false, cidr, err
	}))
	ipStore := ipInformer.Informer().GetStore()
	client.PrependReactor("create", "ipaddresses", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		ip := action.(k8stesting.CreateAction).GetObject().(*networkingv1alpha1.IPAddress)
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
		ip := &networkingv1alpha1.IPAddress{}
		if exists && err == nil {
			ip = obj.(*networkingv1alpha1.IPAddress)
			err = ipStore.Delete(ip)
		}
		return false, ip, err
	}))

	c := &Allocator{
		client:            client.NetworkingV1alpha1(),
		isIPv6:            isIPv6,
		rand:              rand.New(rand.NewSource(time.Now().UnixNano())),
		startCh:           make(chan struct{}),
		stopCh:            make(chan struct{}),
		serviceCIDRLister: cidrInformer.Lister(),
		serviceCIDRSynced: cidrInformer.Informer().GetController().HasSynced,
		ipAddressLister:   ipInformer.Lister(),
		ipAddressSynced:   ipInformer.Informer().GetController().HasSynced,
		queue:             workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "ipAllocator"),
		tree:              iptree.New(isIPv6),
	}

	informerFactory.Start(c.stopCh)
	go c.run()
	<-c.startCh
	return c, nil
}

func TestAllocateServiceCIDRs(t *testing.T) {
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

func TestAllocateTinyServiceCIDRs(t *testing.T) {
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

func TestAllocateReservedServiceCIDRs(t *testing.T) {
	t.Skipf("TODO unsupported static subrange offset")
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
	dynamicOffset := 16
	dynamicBlockSize := 126 - dynamicOffset
	for i := 0; i < dynamicBlockSize; i++ {
		if _, err := r.AllocateNext(); err != nil {
			t.Errorf("Unexpected error trying to allocate: %v", err)
		}
	}
	for i := dynamicOffset; i < 126; i++ {
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

func TestAllocateSmallServiceCIDRs(t *testing.T) {
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

func TestResizeServiceCIDRs(t *testing.T) {
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.0.0/24")
	if err != nil {
		t.Fatal(err)
	}
	r, err := newTestAllocator(cidr)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()

	size := int(rangeSize(netip.MustParsePrefix("192.168.0.0/24")))

	if f := r.Free(); f != size {
		t.Errorf("expected free equal to 254 got: %d", f)
	}
	for i := 0; i < size; i++ {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error allocating %s try %d : %v", ip, i, err)
		}
	}

	if f := r.Free(); f != 0 {
		t.Errorf("expected free equal to 0 got: %d", f)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Error(err)
	}
	newSvcCIDR := &networkingv1alpha1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: "default-cidr2",
		},
		Spec: networkingv1alpha1.ServiceCIDRSpec{
			IPv4: "192.168.0.0/22",
		},
	}
	_, err = r.client.ServiceCIDRs().Create(context.Background(), newSvcCIDR, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.addServiceCIDR(newSvcCIDR)
	time.Sleep(1 * time.Second)
	newSize := int(rangeSize(netip.MustParsePrefix("192.168.0.0/22")))
	if f := r.Free(); f != newSize-size {
		t.Errorf("expected free equal to %d got: %d", newSize-size, f)
	}
	for i := 0; i < newSize-size; i++ {
		if ip, err := r.AllocateNext(); err != nil {
			t.Fatalf("error allocating %s try %d : %v", ip, i, err)
		}
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Error(err)
	}
}

func TestForEachServiceCIDRs(t *testing.T) {
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

func BenchmarkServiceCIDRsAllocateNextIPv4Size1048574(b *testing.B) {
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

func BenchmarkServiceCIDRsAllocateNextIPv6Size65535(b *testing.B) {
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

func Test_rangeSize(t *testing.T) {
	tests := []struct {
		name string
		cidr netip.Prefix
		want int64
	}{
		{
			name: "full mask ipv4",
			cidr: netip.MustParsePrefix("192.168.0.0/32"),
			want: 0,
		},
		{
			name: "full maks ipv6",
			cidr: netip.MustParsePrefix("fd00:1::2/128"),
			want: 0,
		},
		{
			name: "small ipv4",
			cidr: netip.MustParsePrefix("192.168.0.0/31"),
			want: 0,
		},
		{
			name: "small ipv6",
			cidr: netip.MustParsePrefix("fd00:1:2::/127"),
			want: 1,
		},
		{
			name: "normal ipv4",
			cidr: netip.MustParsePrefix("10.0.0.0/24"),
			want: 254,
		},
		{
			name: "normal ipv6",
			cidr: netip.MustParsePrefix("fd00:1:2::/120"),
			want: 255,
		},
		{
			name: "large ipv6",
			cidr: netip.MustParsePrefix("fd00:1:2::/48"),
			want: math.MaxInt64,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := rangeSize(tt.cidr); got != tt.want {
				t.Errorf("prefixNumberHosts() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_offsetAddress(t *testing.T) {
	tests := []struct {
		name   string
		subnet netip.Prefix
		offset int64
		want   netip.Addr
	}{
		{
			name:   "IPv4 offset 0",
			subnet: netip.MustParsePrefix("192.168.0.0/24"),
			offset: 0,
			want:   netip.MustParseAddr("192.168.0.0"),
		},
		{
			name:   "IPv4 offset 1",
			subnet: netip.MustParsePrefix("192.168.0.0/24"),
			offset: 1,
			want:   netip.MustParseAddr("192.168.0.1"),
		},
		{
			name:   "IPv6 offset 1",
			subnet: netip.MustParsePrefix("fd00:1:2:3::/64"),
			offset: 1,
			want:   netip.MustParseAddr("fd00:1:2:3::1"),
		},
		{
			name:   "IPv4 offset last",
			subnet: netip.MustParsePrefix("192.168.0.0/24"),
			offset: 255,
			want:   netip.MustParseAddr("192.168.0.255"),
		},
		{
			name:   "IPv6 offset last",
			subnet: netip.MustParsePrefix("fd00:1:2:3::/64"),
			offset: 0x7FFFFFFFFFFFFFFF,
			want:   netip.MustParseAddr("fd00:1:2:3:7FFF:FFFF:FFFF:FFFF"),
		},
		{
			name:   "IPv4 offset middle",
			subnet: netip.MustParsePrefix("192.168.0.0/24"),
			offset: 128,
			want:   netip.MustParseAddr("192.168.0.128"),
		},
		{
			name:   "IPv6 offset random",
			subnet: netip.MustParsePrefix("fd00:1:2:3::/64"),
			offset: 1025,
			want:   netip.MustParseAddr("fd00:1:2:3::401"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getIndexedAddr(tt.subnet, tt.offset); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("offsetAddress() = %v, want %v", got, tt.want)
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := broadcastAddress(tt.subnet); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("broadcastAddress() = %v, want %v", got, tt.want)
			}
		})
	}
}
