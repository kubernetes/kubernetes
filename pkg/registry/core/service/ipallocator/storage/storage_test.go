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

package storage

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
	allocatorstore "k8s.io/kubernetes/pkg/registry/core/service/allocator/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	netutils "k8s.io/utils/net"
)

func newStorage(t *testing.T) (*etcd3testing.EtcdTestServer, ipallocator.Interface, allocator.Interface, storage.Interface, factory.DestroyFunc) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	_, cidr, err := netutils.ParseCIDRSloppy("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}

	var backing allocator.Interface
	configForAllocations := etcdStorage.ForResource(api.Resource("serviceipallocations"))
	storage, err := ipallocator.New(cidr, func(max int, rangeSpec string, offset int) (allocator.Interface, error) {
		mem := allocator.NewAllocationMapWithOffset(max, rangeSpec, offset)
		backing = mem
		etcd, err := allocatorstore.NewEtcd(mem, "/ranges/serviceips", configForAllocations)
		if err != nil {
			return nil, err
		}
		return etcd, nil
	})
	if err != nil {
		t.Fatalf("unexpected error creating etcd: %v", err)
	}
	s, d, err := generic.NewRawStorage(configForAllocations, nil, nil, "")
	if err != nil {
		t.Fatalf("Couldn't create storage: %v", err)
	}
	destroyFunc := func() {
		d()
		server.Terminate(t)
	}
	return server, storage, backing, s, destroyFunc
}

func validNewRangeAllocation() *api.RangeAllocation {
	_, cidr, _ := netutils.ParseCIDRSloppy("192.168.1.0/24")
	return &api.RangeAllocation{
		Range: cidr.String(),
	}
}

func key() string {
	return "/ranges/serviceips"
}

func TestEmpty(t *testing.T) {
	_, storage, _, _, destroyFunc := newStorage(t)
	defer destroyFunc()
	if err := storage.Allocate(netutils.ParseIPSloppy("192.168.1.2")); !strings.Contains(err.Error(), "cannot allocate resources of type serviceipallocations at this time") {
		t.Fatal(err)
	}
}

func TestErrors(t *testing.T) {
	_, storage, _, _, destroyFunc := newStorage(t)
	defer destroyFunc()
	err := storage.Allocate(netutils.ParseIPSloppy("192.168.0.0"))
	if _, ok := err.(*ipallocator.ErrNotInRange); !ok {
		t.Fatal(err)
	}
}

func TestStore(t *testing.T) {
	_, storage, backing, si, destroyFunc := newStorage(t)
	defer destroyFunc()
	if err := si.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := storage.Allocate(netutils.ParseIPSloppy("192.168.1.2")); err != nil {
		t.Fatal(err)
	}
	ok, err := backing.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("Expected allocation to fail")
	}
	if err := storage.Allocate(netutils.ParseIPSloppy("192.168.1.2")); err != ipallocator.ErrAllocated {
		t.Fatal(err)
	}
}

func TestAllocateReserved(t *testing.T) {
	_, storage, _, si, destroyFunc := newStorage(t)
	defer destroyFunc()
	if err := si.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// allocate all addresses on the dynamic block
	// subnet /24 = 256 ; dynamic block size is min(max(16,256/16),256) = 16
	dynamicOffset := 16
	max := 254
	dynamicBlockSize := max - dynamicOffset
	for i := 0; i < dynamicBlockSize; i++ {
		if _, err := storage.AllocateNext(); err != nil {
			t.Errorf("Unexpected error trying to allocate: %v", err)
		}
	}
	for i := dynamicOffset; i < max; i++ {
		ip := fmt.Sprintf("192.168.1.%d", i+1)
		if !storage.Has(netutils.ParseIPSloppy(ip)) {
			t.Errorf("IP %s expected to be allocated", ip)
		}
	}

	// allocate all addresses on the static block
	for i := 0; i < dynamicOffset; i++ {
		ip := fmt.Sprintf("192.168.1.%d", i+1)
		if err := storage.Allocate(netutils.ParseIPSloppy(ip)); err != nil {
			t.Errorf("Unexpected error trying to allocate IP %s: %v", ip, err)
		}
	}
	if _, err := storage.AllocateNext(); err == nil {
		t.Error("Allocator expected to be full")
	}
	// release one address in the allocated block and another a new one randomly
	if err := storage.Release(netutils.ParseIPSloppy("192.168.1.10")); err != nil {
		t.Fatalf("Unexpected error trying to release ip 192.168.1.10: %v", err)
	}
	if _, err := storage.AllocateNext(); err != nil {
		t.Error(err)
	}
	if _, err := storage.AllocateNext(); err == nil {
		t.Error("Allocator expected to be full")
	}
}

func TestAllocateReservedDynamicBlockExhausted(t *testing.T) {
	_, storage, _, si, destroyFunc := newStorage(t)
	defer destroyFunc()
	if err := si.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// allocate all addresses both on the dynamic and reserved blocks
	// once the dynamic block has been exhausted
	// the dynamic allocator will use the reserved block
	max := 254

	for i := 0; i < max; i++ {
		if _, err := storage.AllocateNext(); err != nil {
			t.Errorf("Unexpected error trying to allocate: %v", err)
		}
	}
	for i := 0; i < max; i++ {
		ip := fmt.Sprintf("192.168.1.%d", i+1)
		if !storage.Has(netutils.ParseIPSloppy(ip)) {
			t.Errorf("IP %s expected to be allocated", ip)
		}
	}

	if _, err := storage.AllocateNext(); err == nil {
		t.Error("Allocator expected to be full")
	}
}
