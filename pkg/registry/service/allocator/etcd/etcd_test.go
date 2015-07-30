/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"strings"
	"testing"

	"github.com/coreos/go-etcd/etcd"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/allocator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/storage"
	etcdstorage "github.com/GoogleCloudPlatform/kubernetes/pkg/storage/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
)

func newEtcdStorage(t *testing.T) (*tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, testapi.Codec(), etcdtest.PathPrefix())
	return fakeEtcdClient, etcdStorage
}

func newStorage(t *testing.T) (*Etcd, allocator.Interface, *tools.FakeEtcdClient) {
	fakeEtcdClient, s := newEtcdStorage(t)

	mem := allocator.NewAllocationMap(100, "rangeSpecValue")
	etcd := NewEtcd(mem, "/ranges/serviceips", "serviceipallocation", s)

	return etcd, mem, fakeEtcdClient
}

func key() string {
	s := "/ranges/serviceips"
	return etcdtest.AddPrefix(s)
}

func TestEmpty(t *testing.T) {
	storage, _, ecli := newStorage(t)
	ecli.ExpectNotFoundGet(key())
	if _, err := storage.Allocate(1); !strings.Contains(err.Error(), "cannot allocate resources of type serviceipallocation at this time") {
		t.Fatal(err)
	}
}

func initialObject(ecli *tools.FakeEtcdClient) {
	ecli.Data[key()] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				CreatedIndex:  1,
				ModifiedIndex: 2,
				Value: runtime.EncodeOrDie(testapi.Codec(), &api.RangeAllocation{
					Range: "rangeSpecValue",
				}),
			},
		},
		E: nil,
	}
}

func TestStore(t *testing.T) {
	storage, backing, ecli := newStorage(t)
	initialObject(ecli)

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

	obj := ecli.Data[key()]
	if obj.R == nil || obj.R.Node == nil {
		t.Fatalf("%s is empty: %#v", key(), obj)
	}
	t.Logf("data: %#v", obj.R.Node)

	other := allocator.NewAllocationMap(100, "rangeSpecValue")

	allocation := &api.RangeAllocation{}
	if err := storage.storage.Get(key(), allocation, false); err != nil {
		t.Fatal(err)
	}
	if allocation.ResourceVersion != "1" {
		t.Fatalf("%#v", allocation)
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
	otherStorage := NewEtcd(other, "/ranges/serviceips", "serviceipallocation", storage.storage)
	if ok, err := otherStorage.Allocate(2); ok || err != nil {
		t.Fatal(err)
	}
}
