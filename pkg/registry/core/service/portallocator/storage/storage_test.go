/*
Copyright 2020 The Kubernetes Authors.

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

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
	allocatorstore "k8s.io/kubernetes/pkg/registry/core/service/allocator/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

const (
	basePortRange = 30000
	sizePortRange = 2768
)

func newStorage(t *testing.T) (*etcd3testing.EtcdTestServer, portallocator.Interface, allocator.Interface, storage.Interface, factory.DestroyFunc) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")

	serviceNodePortRange := utilnet.PortRange{Base: basePortRange, Size: sizePortRange}
	configForAllocations := etcdStorage.ForResource(api.Resource("servicenodeportallocations"))
	var backing allocator.Interface
	storage, err := portallocator.New(serviceNodePortRange, func(max int, rangeSpec string, offset int) (allocator.Interface, error) {
		mem := allocator.NewAllocationMapWithOffset(max, rangeSpec, offset)
		backing = mem
		etcd, err := allocatorstore.NewEtcd(mem, "/ranges/servicenodeports", configForAllocations)
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
	portRange := fmt.Sprintf("%d-%d", basePortRange, basePortRange+sizePortRange-1)
	return &api.RangeAllocation{
		Range: portRange,
	}
}

func key() string {
	return "/ranges/servicenodeports"
}

// TestEmpty fails to allocate ports if the storage wasn't initialized with a servicenodeport range
func TestEmpty(t *testing.T) {
	_, storage, _, _, destroyFunc := newStorage(t)
	defer destroyFunc()
	if err := storage.Allocate(31000); !strings.Contains(err.Error(), "cannot allocate resources of type servicenodeportallocations at this time") {
		t.Fatal(err)
	}
}

// TestAllocate fails to allocate ports out of the valid port range
func TestAllocate(t *testing.T) {
	_, storage, _, si, destroyFunc := newStorage(t)
	defer destroyFunc()
	if err := si.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tests := []struct {
		name   string
		port   int
		errMsg string
	}{
		{
			name:   "Allocate base port",
			port:   basePortRange,
			errMsg: "",
		},
		{
			name:   "Allocate maximum from the port range",
			port:   basePortRange + sizePortRange - 1,
			errMsg: "",
		},
		{
			name:   "Allocate invalid port: base port minus 1",
			port:   basePortRange - 1,
			errMsg: fmt.Sprintf("provided port is not in the valid range. The range of valid ports is %d-%d", basePortRange, basePortRange+sizePortRange-1),
		},
		{
			name:   "Allocate invalid port: maximum port from the port range plus 1",
			port:   basePortRange + sizePortRange,
			errMsg: fmt.Sprintf("provided port is not in the valid range. The range of valid ports is %d-%d", basePortRange, basePortRange+sizePortRange-1),
		},
		{
			name:   "Allocate invalid port",
			port:   -2,
			errMsg: fmt.Sprintf("provided port is not in the valid range. The range of valid ports is %d-%d", basePortRange, basePortRange+sizePortRange-1),
		},
	}
	for _, tt := range tests {
		tt := tt // NOTE: https://github.com/golang/go/wiki/CommonMistakes#using-goroutines-on-loop-iterator-variables
		t.Run(tt.name, func(t *testing.T) {
			err := storage.Allocate(tt.port)
			if (err == nil) != (tt.errMsg == "") {
				t.Fatalf("Error expected %v, received %v", tt.errMsg, err)
			}
			if err != nil && err.Error() != tt.errMsg {
				t.Fatalf("Error message expected %v, received %v", tt.errMsg, err)
			}
		})
	}

}

// TestReallocate test that we can not allocate a port already allocated until it is released
func TestReallocate(t *testing.T) {
	_, storage, backing, si, destroyFunc := newStorage(t)
	defer destroyFunc()
	if err := si.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Allocate a port inside the valid port range
	if err := storage.Allocate(30100); err != nil {
		t.Fatal(err)
	}

	// Try to allocate the same port in the local bitmap
	// The local bitmap stores the offset of the port
	// offset = port - base (30100 - 30000 = 100)
	ok, err := backing.Allocate(100)
	if err != nil {
		t.Fatal(err)
	}
	// It should not allocate the port because it was already allocated
	if ok {
		t.Fatal("Expected allocation to fail")
	}
	// Try to allocate the port again should fail
	if err := storage.Allocate(30100); err != portallocator.ErrAllocated {
		t.Fatal(err)
	}

	// Release the port
	if err := storage.Release(30100); err != nil {
		t.Fatal(err)
	}

	// Try to allocate the port again should succeed because we've released it
	if err := storage.Allocate(30100); err != nil {
		t.Fatal(err)
	}

}

func TestAllocateReserved(t *testing.T) {
	_, storage, _, si, destroyFunc := newStorage(t)
	defer destroyFunc()
	if err := si.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// allocate all ports on the dynamic block
	// default range size is 2768, and dynamic block size is min(max(16,2768/32),128) = 86
	dynamicOffset := 86
	dynamicBlockSize := sizePortRange - dynamicOffset
	for i := 0; i < dynamicBlockSize; i++ {
		if _, err := storage.AllocateNext(); err != nil {
			t.Errorf("Unexpected error trying to allocate: %v", err)
		}
	}
	for i := dynamicOffset; i < sizePortRange; i++ {
		port := i + basePortRange
		if !storage.Has(port) {
			t.Errorf("Port %d expected to be allocated", port)
		}
	}

	// allocate all ports on the static block
	for i := 0; i < dynamicOffset; i++ {
		port := i + basePortRange
		if err := storage.Allocate(port); err != nil {
			t.Errorf("Unexpected error trying to allocate Port %d: %v", port, err)
		}
	}
	if _, err := storage.AllocateNext(); err == nil {
		t.Error("Allocator expected to be full")
	}
	// release one port in the allocated block and another a new one randomly
	if err := storage.Release(basePortRange + 53); err != nil {
		t.Fatalf("Unexpected error trying to release port 30053: %v", err)
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

	// allocate all ports both on the dynamic and reserved blocks
	// once the dynamic block has been exhausted
	// the dynamic allocator will use the reserved block
	for i := 0; i < sizePortRange; i++ {
		if _, err := storage.AllocateNext(); err != nil {
			t.Errorf("Unexpected error trying to allocate: %v", err)
		}
	}
	for i := 0; i < sizePortRange; i++ {
		port := i + basePortRange
		if !storage.Has(port) {
			t.Errorf("Port %d expected to be allocated", port)
		}
	}

	if _, err := storage.AllocateNext(); err == nil {
		t.Error("Allocator expected to be full")
	}
}
