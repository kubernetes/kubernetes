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
	"fmt"
	"net/netip"
	"testing"
	"time"

	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	netutils "k8s.io/utils/net"
)

func newTestMetaAllocator() (*MetaAllocator, error) {
	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, 0*time.Second)
	serviceCIDRInformer := informerFactory.Networking().V1beta1().ServiceCIDRs()
	serviceCIDRStore := serviceCIDRInformer.Informer().GetIndexer()
	ipInformer := informerFactory.Networking().V1beta1().IPAddresses()
	ipStore := ipInformer.Informer().GetIndexer()

	client.PrependReactor("create", "servicecidrs", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		cidr := action.(k8stesting.CreateAction).GetObject().(*networkingv1beta1.ServiceCIDR)
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
		cidr := &networkingv1beta1.ServiceCIDR{}
		if exists && err == nil {
			cidr = obj.(*networkingv1beta1.ServiceCIDR)
			err = serviceCIDRStore.Delete(cidr)
		}
		return false, cidr, err
	}))

	client.PrependReactor("create", "ipaddresses", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		ip := action.(k8stesting.CreateAction).GetObject().(*networkingv1beta1.IPAddress)
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
		ip := &networkingv1beta1.IPAddress{}
		if exists && err == nil {
			ip = obj.(*networkingv1beta1.IPAddress)
			err = ipStore.Delete(ip)
		}
		return false, ip, err
	}))

	c := newMetaAllocator(client.NetworkingV1beta1(), serviceCIDRInformer, ipInformer, false, nil)

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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, false)
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableAllocatorDualWrite, false)
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
func newServiceCIDR(name, cidr string) *networkingv1beta1.ServiceCIDR {
	return &networkingv1beta1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1beta1.ServiceCIDRSpec{
			CIDRs: []string{cidr},
		},
		Status: networkingv1beta1.ServiceCIDRStatus{
			Conditions: []metav1.Condition{
				{
					Type:   string(networkingv1beta1.ServiceCIDRConditionReady),
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
