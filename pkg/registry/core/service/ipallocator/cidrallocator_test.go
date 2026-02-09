/*
Copyright 2023 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"net/netip"
	"testing"
	"time"

	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	networkingv1fake "k8s.io/client-go/kubernetes/typed/networking/v1/fake"
	k8stesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/features"
	netutils "k8s.io/utils/net"
)

func newTestMetaAllocator() (*MetaAllocator, error) {
	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, 0*time.Second)
	serviceCIDRInformer := informerFactory.Networking().V1().ServiceCIDRs()
	serviceCIDRStore := serviceCIDRInformer.Informer().GetIndexer()
	ipInformer := informerFactory.Networking().V1().IPAddresses()
	ipStore := ipInformer.Informer().GetIndexer()

	client.PrependReactor("create", "servicecidrs", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		cidr := action.(k8stesting.CreateAction).GetObject().(*networkingv1.ServiceCIDR)
		_, exists, err := serviceCIDRStore.GetByKey(cidr.Name)
		if exists && err != nil {
			return false, nil, fmt.Errorf("cidr already exist")
		}
		cidr.Generation = 1
		err = serviceCIDRStore.Add(cidr)
		return false, cidr, err
	}))
	client.PrependReactor("delete", "servicecidrs", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		name := action.(k8stesting.DeleteAction).GetName()
		obj, exists, err := serviceCIDRStore.GetByKey(name)
		cidr := &networkingv1.ServiceCIDR{}
		if exists && err == nil {
			cidr = obj.(*networkingv1.ServiceCIDR)
			err = serviceCIDRStore.Delete(cidr)
		}
		return false, cidr, err
	}))

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

	c := newMetaAllocator(client.NetworkingV1(), serviceCIDRInformer, ipInformer, false, nil)

	c.serviceCIDRSynced = func() bool { return true }
	c.ipAddressSynced = func() bool { return true }
	go c.run()
	return c, nil
}

func TestCIDRAllocateMultiple(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, true)
	r, err := newTestMetaAllocator()
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

	cidr := newServiceCIDR("test", "192.168.0.0/28")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr)
	// wait for the cidr to be processed and set the informer synced
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			t.Logf("unexpected error %v", err)
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	found := sets.NewString()
	count := 0
	for r.Free() > 0 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ free: %d count: %d: %v", r.Free(), count, err)
		}
		count++
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice: %d", ip, count)
		}
		found.Insert(ip.String())
	}
	if count != 14 {
		t.Fatalf("expected 14 IPs got %d", count)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Fatal(err)
	}

	cidr2 := newServiceCIDR("test2", "10.0.0.0/28")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr2, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr2)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("10.0.0.11"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// allocate one IP from the new allocator
	err = r.Allocate(netutils.ParseIPSloppy("10.0.0.11"))
	if err != nil {
		t.Fatalf("error allocating IP 10.0.0.11 from new allocator: %v", err)
	}
	count++
	for r.Free() > 0 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ free: %d count: %d: %v", r.Free(), count, err)
		}
		count++
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice: %d", ip, count)
		}
		found.Insert(ip.String())
	}
	if count != 28 {
		t.Fatalf("expected 28 IPs got %d", count)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Fatal(err)
	}

}

func TestCIDRAllocateShadow(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, true)
	r, err := newTestMetaAllocator()
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

	cidr := newServiceCIDR("test", "192.168.1.0/24")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr)
	// wait for the cidr to be processed and set the informer synced
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.1.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// can not allocate the subnet IP from the new allocator
	err = r.Allocate(netutils.ParseIPSloppy("192.168.1.0"))
	if err == nil {
		t.Fatalf("unexpected allocation for IP 192.168.1.0")
	}

	if f := r.Used(); f != 0 {
		t.Errorf("used: %d", f)
	}

	cidr2 := newServiceCIDR("test2", "192.168.0.0/16")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr2, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr2)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// allocate one IP from the new allocator
	err = r.Allocate(netutils.ParseIPSloppy("192.168.1.0"))
	if err != nil {
		t.Fatalf("error allocating IP 192.168.1.0 from new allocator: %v", err)
	}

	if f := r.Used(); f != 1 {
		t.Errorf("used: %d", f)
	}

}

func TestCIDRAllocateGrow(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, true)
	r, err := newTestMetaAllocator()
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

	cidr := newServiceCIDR("test", "192.168.0.0/28")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	found := sets.NewString()
	count := 0
	for r.Free() > 0 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ free: %d count: %d: %v", r.Free(), count, err)
		}
		count++
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice: %d", ip, count)
		}
		found.Insert(ip.String())
	}
	if count != 14 {
		t.Fatalf("expected 14 IPs got %d", count)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Fatal(err)
	}

	cidr2 := newServiceCIDR("test2", "192.168.0.0/24")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr2, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr2)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.253"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	for r.Free() > 0 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ free: %d count: %d: %v", r.Free(), count, err)
		}
		count++
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice: %d", ip, count)
		}
		found.Insert(ip.String())
	}
	if count != 254 {
		t.Fatalf("expected 254 IPs got %d", count)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Fatal(err)
	}

}

func TestCIDRAllocateShrink(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, true)
	r, err := newTestMetaAllocator()
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

	cidr := newServiceCIDR("test", "192.168.0.0/24")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	found := sets.NewString()
	count := 0
	for r.Free() > 0 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ free: %d count: %d: %v", r.Free(), count, err)
		}
		count++
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice: %d", ip, count)
		}
		found.Insert(ip.String())
	}
	if count != 254 {
		t.Fatalf("expected 254 IPs got %d", count)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Fatal(err)
	}
	for _, ip := range found.List() {
		err = r.Release(netutils.ParseIPSloppy(ip))
		if err != nil {
			t.Fatalf("unexpected error releasing ip %s", err)
		}
	}
	if r.Used() > 0 {
		t.Fatalf("expected allocator to be empty, got %d", r.Free())
	}
	cidr2 := newServiceCIDR("cidr2", "192.168.0.0/28")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr2, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr2)
	err = r.client.ServiceCIDRs().Delete(context.Background(), cidr.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.deleteServiceCIDR(cidr)

	// wait for the cidr to be processed (delete ServiceCIDR)
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.253"), true)
		if err != nil {
			return true, nil
		}

		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// wait for the cidr to be processed (create ServiceCIDR)
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	count = 0
	for r.Free() > 0 {
		_, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ free: %d count: %d: %v", r.Free(), count, err)
		}
		count++
	}
	if count != 14 {
		t.Fatalf("expected 14 IPs got %d", count)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Fatal(err)
	}

}

func TestCIDRAllocateDualWrite(t *testing.T) {
	featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, false)
	r, err := newTestMetaAllocator()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()

	if f := r.Free(); f != 0 {
		t.Errorf("free: %d", f)
	}

	cidr := newServiceCIDR("test", "192.168.0.0/28")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr)
	// wait for the cidr to be processed and set the informer synced
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			t.Logf("unexpected error %v", err)
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Create a bitmap allocator that will mirror the ip allocator
	_, ipnet, err := netutils.ParseCIDRSloppy(cidr.Spec.CIDRs[0])
	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	bitmapAllocator, err := NewInMemory(ipnet)
	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	r.bitmapAllocator = bitmapAllocator

	found := sets.NewString()
	count := 0
	for r.Free() > 0 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ free: %d count: %d: %v", r.Free(), count, err)
		}
		if r.Free() != bitmapAllocator.Free() {
			t.Fatalf("ip and bitmap allocator out of sync: %d %d", r.Free(), bitmapAllocator.Free())
		}
		count++
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice: %d", ip, count)
		}
		found.Insert(ip.String())
	}
	if count != 14 {
		t.Fatalf("expected 14 IPs got %d", count)
	}
	if _, err := r.AllocateNext(); err == nil {
		t.Fatal(err)
	}
}

func TestCIDRAllocateDualWriteCollision(t *testing.T) {
	featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, false)
	r, err := newTestMetaAllocator()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()

	if f := r.Free(); f != 0 {
		t.Errorf("free: %d", f)
	}

	cidr := newServiceCIDR("test", "192.168.0.0/28")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr)
	// wait for the cidr to be processed and set the informer synced
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			t.Logf("unexpected error %v", err)
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Create a bitmap allocator that will mirror the ip allocator
	_, ipnet, err := netutils.ParseCIDRSloppy(cidr.Spec.CIDRs[0])
	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	bitmapAllocator, err := NewInMemory(ipnet)
	if err != nil {
		t.Fatalf("unexpected failure: %v", err)
	}
	r.bitmapAllocator = bitmapAllocator

	// preallocate one IP in the bitmap allocator
	err = bitmapAllocator.Allocate(netutils.ParseIPSloppy("192.168.0.5"))
	if err != nil {
		t.Fatalf("unexpected error allocating an IP on the bitmap allocator: %v", err)
	}
	// the ipallocator must not be able to allocate
	err = r.Allocate(netutils.ParseIPSloppy("192.168.0.5"))
	if err == nil {
		t.Fatalf("unexpected allocation: %v", err)
	}
}

// TODO: add IPv6 and dual stack test cases
func newServiceCIDR(name, cidr string) *networkingv1.ServiceCIDR {
	return &networkingv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1.ServiceCIDRSpec{
			CIDRs: []string{cidr},
		},
		Status: networkingv1.ServiceCIDRStatus{
			Conditions: []metav1.Condition{
				{
					Type:   string(networkingv1.ServiceCIDRConditionReady),
					Status: metav1.ConditionTrue,
				},
			},
		},
	}
}

func Test_isNotContained(t *testing.T) {
	tests := []struct {
		name     string
		prefix   netip.Prefix
		prefixes []netip.Prefix
		want     bool
	}{
		{
			name:     "ipv4 not contained nor overlapping",
			prefix:   netip.MustParsePrefix("192.168.0.0/24"),
			prefixes: []netip.Prefix{netip.MustParsePrefix("10.0.0.0/24"), netip.MustParsePrefix("10.0.0.0/27")},
			want:     true,
		},
		{
			name:     "ipv4 not contained but contains",
			prefix:   netip.MustParsePrefix("10.0.0.0/8"),
			prefixes: []netip.Prefix{netip.MustParsePrefix("10.0.0.0/24"), netip.MustParsePrefix("10.0.0.0/27")},
			want:     true,
		},
		{
			name:     "ipv4 not contained but matches existing one",
			prefix:   netip.MustParsePrefix("10.0.0.0/24"),
			prefixes: []netip.Prefix{netip.MustParsePrefix("10.0.0.0/24"), netip.MustParsePrefix("10.0.0.0/27")},
			want:     true,
		},
		{
			name:     "ipv4 contained but matches existing one",
			prefix:   netip.MustParsePrefix("10.0.0.0/27"),
			prefixes: []netip.Prefix{netip.MustParsePrefix("10.0.0.0/24"), netip.MustParsePrefix("10.0.0.0/27")},
			want:     false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isNotContained(tt.prefix, tt.prefixes); got != tt.want {
				t.Errorf("isNotContained() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCIDRAllocatorClusterIPAllocatedMetrics(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, true)
	clearMetrics()

	r, err := newTestMetaAllocator()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()

	// Enable metrics for the meta allocator
	r.EnableMetrics()

	// Create first CIDR - small /30 network (only 2 usable IPs)
	cidr1 := newServiceCIDR("test1", "192.168.1.0/30")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr1, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr1)

	// Wait for the first CIDR to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.1.1"), true)
		if err != nil {
			t.Logf("unexpected error %v", err)
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Check initial metrics for first CIDR
	em1 := testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, "192.168.1.0/30", em1)

	// Allocate all IPs from first CIDR (should be 2 usable IPs: .1 and .2)
	found := sets.NewString()
	allocatedFromCIDR1 := 0
	for r.Free() > 0 && allocatedFromCIDR1 < 2 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error allocating from first CIDR @ allocated: %d: %v", allocatedFromCIDR1, err)
		}
		allocatedFromCIDR1++
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice", ip)
		}
		found.Insert(ip.String())
	}

	// Verify we allocated 2 IPs from first CIDR
	if allocatedFromCIDR1 != 2 {
		t.Fatalf("expected 2 IPs from first CIDR, got %d", allocatedFromCIDR1)
	}

	// Check metrics after filling first CIDR
	dynamicAllocatedCidr1, err := testutil.GetCounterMetricValue(clusterIPAllocations.WithLabelValues("192.168.1.0/30", "dynamic"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocations.Name, err)
	}
	if dynamicAllocatedCidr1 != 2 {
		t.Fatalf("Expected 2 dynamic allocations from first CIDR, received %f", dynamicAllocatedCidr1)
	}

	// Create second CIDR - small /29 network (6 usable IPs)
	cidr2 := newServiceCIDR("test2", "10.0.0.0/29")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr2, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr2)

	// Wait for the second CIDR to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("10.0.0.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Check initial metrics for second CIDR
	em2 := testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, "10.0.0.0/29", em2)

	// Allocate all remaining IPs from second CIDR (should be 6 usable IPs)
	allocatedFromCIDR2 := 0
	for r.Free() > 0 && allocatedFromCIDR2 < 6 {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error allocating from second CIDR @ allocated: %d: %v", allocatedFromCIDR2, err)
		}
		allocatedFromCIDR2++
		if found.Has(ip.String()) {
			t.Fatalf("allocated %s twice", ip)
		}
		found.Insert(ip.String())
	}

	// Verify we allocated 6 IPs from second CIDR
	if allocatedFromCIDR2 != 6 {
		t.Fatalf("expected 6 IPs from second CIDR, got %d", allocatedFromCIDR2)
	}

	// Check total allocated IPs
	totalAllocated := allocatedFromCIDR1 + allocatedFromCIDR2
	if totalAllocated != 8 {
		t.Fatalf("expected 8 total IPs allocated, got %d", totalAllocated)
	}

	// Check metrics after filling second CIDR
	dynamicAllocatedCidr2, err := testutil.GetCounterMetricValue(clusterIPAllocations.WithLabelValues("10.0.0.0/29", "dynamic"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocations.Name, err)
	}
	if dynamicAllocatedCidr2 != 6 {
		t.Fatalf("Expected 6 dynamic allocations from second CIDR, received %f", dynamicAllocatedCidr2)
	}

	// Try to allocate more IPs - should fail since both CIDRs are exhausted
	if _, err := r.AllocateNext(); err == nil {
		t.Fatal("expected error when trying to allocate from exhausted CIDRs")
	}

	// Parse the CIDRs for proper IP containment checking
	_, cidr1Net, err := netutils.ParseCIDRSloppy("192.168.1.0/30")
	if err != nil {
		t.Fatalf("failed to parse CIDR1: %v", err)
	}
	_, cidr2Net, err := netutils.ParseCIDRSloppy("10.0.0.0/29")
	if err != nil {
		t.Fatalf("failed to parse CIDR2: %v", err)
	}

	// Try to allocate the same IP addresses to generate static allocation errors
	errorCount1 := 0
	errorCount2 := 0
	for s := range found {
		ip := netutils.ParseIPSloppy(s)
		if err := r.Allocate(ip); !errors.Is(err, ErrAllocated) {
			t.Fatalf("expected ErrAllocated when trying to allocate existing IP %s, got: %v", s, err)
		}
		// Count which CIDR the error belongs to using proper CIDR containment
		if cidr1Net.Contains(ip) {
			errorCount1++
		} else if cidr2Net.Contains(ip) {
			errorCount2++
		} else {
			t.Fatalf("IP %s does not belong to any expected CIDR", ip.String())
		}
	}

	// Check static allocation errors for first CIDR
	staticErrorsCidr1, err := testutil.GetCounterMetricValue(clusterIPAllocationErrors.WithLabelValues("192.168.1.0/30", "static"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocationErrors.Name, err)
	}
	if staticErrorsCidr1 != float64(errorCount1) {
		t.Fatalf("Expected %d static allocation errors from first CIDR, received %f", errorCount1, staticErrorsCidr1)
	}

	// Check static allocation errors for second CIDR
	staticErrorsCidr2, err := testutil.GetCounterMetricValue(clusterIPAllocationErrors.WithLabelValues("10.0.0.0/29", "static"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", clusterIPAllocationErrors.Name, err)
	}
	if staticErrorsCidr2 != float64(errorCount2) {
		t.Fatalf("Expected %d static allocation errors from second CIDR, received %f", errorCount2, staticErrorsCidr2)
	}
}

func TestCIDRAllocateNextStopOnError(t *testing.T) {
	r, err := newTestMetaAllocator()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()

	// add a CIDR
	cidr1 := newServiceCIDR("test1", "192.168.0.0/24")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr1, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr1)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// add a second CIDR
	cidr2 := newServiceCIDR("test2", "192.168.1.0/24")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr2, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr2)

	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.1.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	calls := 0
	// override the client to inject the reactor to fail the allocation
	fakeclient, ok := r.client.(*networkingv1fake.FakeNetworkingV1)
	if !ok {
		t.Fatal("invalid client")
	}
	fakeclient.PrependReactor("create", "ipaddresses", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		calls++
		return true, nil, fmt.Errorf("server error")
	}))

	_, err = r.AllocateNext()
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "server error" {
		t.Fatalf("expected 'server error', got %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected 1 call, got %d", calls)
	}
}

func TestCIDRAllocateNextDontStopOnNot(t *testing.T) {
	r, err := newTestMetaAllocator()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Destroy()

	// add a CIDR
	cidr1 := newServiceCIDR("test1", "192.168.0.0/24")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr1, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr1)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return false }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// add a second CIDR
	cidr2 := newServiceCIDR("test2", "192.168.1.0/24")
	_, err = r.client.ServiceCIDRs().Create(context.Background(), cidr2, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.enqueueServiceCIDR(cidr2)

	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.1.1"), true)
		if err != nil {
			return false, nil
		}
		allocator.ipAddressSynced = func() bool { return true }
		return allocator.ready.Load(), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// make the first allocator not ready
	r.allocators["192.168.0.0/24"].allocator.ready.Store(false)
	_, secondSubnet, err := netutils.ParseCIDRSloppy("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200; i++ {
		ip, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !secondSubnet.Contains(ip) {
			t.Fatalf("expected IP %s to be in %s", ip.String(), secondSubnet.String())
		}
	}
}
