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
	"testing"
	"time"

	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	netutils "k8s.io/utils/net"
)

func newTestMetaAllocator() (*MetaAllocator, error) {
	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, 0*time.Second)
	serviceCIDRInformer := informerFactory.Networking().V1alpha1().ServiceCIDRs()
	serviceCIDRStore := serviceCIDRInformer.Informer().GetIndexer()
	serviceCIDRInformer.Informer().HasSynced()
	ipInformer := informerFactory.Networking().V1alpha1().IPAddresses()
	ipStore := ipInformer.Informer().GetIndexer()

	client.PrependReactor("create", "servicecidrs", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		cidr := action.(k8stesting.CreateAction).GetObject().(*networkingv1alpha1.ServiceCIDR)
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
		cidr := &networkingv1alpha1.ServiceCIDR{}
		if exists && err == nil {
			cidr = obj.(*networkingv1alpha1.ServiceCIDR)
			err = serviceCIDRStore.Delete(cidr)
		}
		return false, cidr, err
	}))

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

	c, err := NewMetaAllocator(client.NetworkingV1alpha1(), serviceCIDRInformer, ipInformer, false)
	if err != nil {
		return nil, err
	}
	// we can not force the state of the informers to be synced without racing
	// so we run our worker here
	go wait.Until(c.runWorker, time.Second, c.internalStopCh)
	return c, nil
}

func TestCIDRAllocateMultiple(t *testing.T) {
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
	r.addServiceCIDR(cidr)
	// wait for the cidr to be processed and set the informer synced
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"))
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
	r.addServiceCIDR(cidr2)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("10.0.0.11"))
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
	r.addServiceCIDR(cidr)
	// wait for the cidr to be processed and set the informer synced
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.1.0"))
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
	r.addServiceCIDR(cidr2)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.0"))
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
	r.addServiceCIDR(cidr)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"))
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
	r.addServiceCIDR(cidr2)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.253"))
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
	r.addServiceCIDR(cidr)
	// wait for the cidr to be processed
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"))
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
	r.addServiceCIDR(cidr2)
	err = r.client.ServiceCIDRs().Delete(context.Background(), cidr.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r.deleteServiceCIDR(cidr)

	// wait for the cidr to be processed (delete ServiceCIDR)
	err = wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.253"))
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
		allocator, err := r.getAllocator(netutils.ParseIPSloppy("192.168.0.1"))
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

// TODO: add IPv6 and dual stack test cases
func newServiceCIDR(name, cidr string) *networkingv1alpha1.ServiceCIDR {
	return &networkingv1alpha1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1alpha1.ServiceCIDRSpec{
			CIDRs: []string{cidr},
		},
		Status: networkingv1alpha1.ServiceCIDRStatus{
			Conditions: []metav1.Condition{
				{
					Type:   string(networkingv1alpha1.ServiceCIDRConditionReady),
					Status: metav1.ConditionTrue,
				},
			},
		},
	}
}
