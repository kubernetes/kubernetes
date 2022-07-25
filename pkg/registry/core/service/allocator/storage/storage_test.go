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
	"strings"
	"testing"

	apiserverstorage "k8s.io/apiserver/pkg/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*Etcd, *etcd3testing.EtcdTestServer, allocator.Interface, *storagebackend.Config) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	mem := allocator.NewAllocationMap(100, "rangeSpecValue")
	etcd, err := NewEtcd(mem, "/ranges/serviceips", etcdStorage.ForResource(api.Resource("serviceipallocations")))
	if err != nil {
		t.Fatalf("unexpected error creating etcd: %v", err)
	}
	return etcd, server, mem, &etcdStorage.Config
}

func validNewRangeAllocation() *api.RangeAllocation {
	return &api.RangeAllocation{
		Range: "rangeSpecValue",
	}
}

func key() string {
	return "/ranges/serviceips"
}

func TestEmpty(t *testing.T) {
	storage, server, _, _ := newStorage(t)
	defer server.Terminate(t)
	if _, err := storage.Allocate(1); !strings.Contains(err.Error(), "cannot allocate resources of type serviceipallocations at this time") {
		t.Fatal(err)
	}
}

func TestStore(t *testing.T) {
	storage, server, backing, config := newStorage(t)
	defer server.Terminate(t)
	if err := storage.storage.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := storage.Allocate(2); err != nil {
		t.Fatal(err)
	}
	ok, err := backing.Allocate(2)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("Expected backing allocation to fail")
	}
	if ok, err := storage.Allocate(2); ok || err != nil {
		t.Fatal("Expected allocation to fail")
	}

	other := allocator.NewAllocationMap(100, "rangeSpecValue")

	allocation := &api.RangeAllocation{}
	if err := storage.storage.Get(context.TODO(), key(), apiserverstorage.GetOptions{}, allocation); err != nil {
		t.Fatal(err)
	}
	if allocation.Range != "rangeSpecValue" {
		t.Errorf("unexpected stored Range: %s", allocation.Range)
	}
	if err := other.Restore("rangeSpecValue", allocation.Data); err != nil {
		t.Fatal(err)
	}
	if !other.Has(2) {
		t.Fatalf("could not restore allocated IP: %#v", other)
	}

	other = allocator.NewAllocationMap(100, "rangeSpecValue")
	otherStorage, err := NewEtcd(other, "/ranges/serviceips", config.ForResource(api.Resource("serviceipallocations")))
	if err != nil {
		t.Fatalf("unexpected error creating etcd: %v", err)
	}
	if ok, err := otherStorage.Allocate(2); ok || err != nil {
		t.Fatal(err)
	}
}

// Test that one item is allocated in storage but is not allocated locally
// When try to allocate it, it should fail despite it's free in the local bitmap
// bot not in the storage
func TestAllocatedStorageButReleasedLocally(t *testing.T) {
	storage, server, backing, _ := newStorage(t)
	defer server.Terminate(t)
	if err := storage.storage.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Allocate an item in the storage
	if _, err := storage.Allocate(2); err != nil {
		t.Fatal(err)
	}

	// Release the item in the local bitmap
	// emulating it's out of sync with the storage
	err := backing.Release(2)
	if err != nil {
		t.Fatal(err)
	}

	// It should fail trying to allocate it deespite it's free
	// in the local bitmap because it's not in the storage
	ok, err := storage.Allocate(2)
	if ok || err != nil {
		t.Fatal(err)
	}

}

// Test that one item is free in storage but is  allocated locally
// When try to allocate it, it should succeed despite it's allocated
// in the local bitmap bot not in the storage
func TestAllocatedLocallyButReleasedStorage(t *testing.T) {
	storage, server, backing, _ := newStorage(t)
	defer server.Terminate(t)
	if err := storage.storage.Create(context.TODO(), key(), validNewRangeAllocation(), nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Allocate an item in the local bitmap only but not in the storage
	// emulating it's out of sync with the storage
	if _, err := backing.Allocate(2); err != nil {
		t.Fatal(err)
	}

	// It should be able to allocate it
	// because it's free in the storage
	ok, err := storage.Allocate(2)
	if !ok || err != nil {
		t.Fatal(err)
	}

}
