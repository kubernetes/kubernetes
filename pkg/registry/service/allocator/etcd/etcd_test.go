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
	"testing"

	"github.com/coreos/go-etcd/etcd"

	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/allocator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
)

func newHelper(t *testing.T) (*tools.FakeEtcdClient, tools.EtcdHelper) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	helper := tools.NewEtcdHelper(fakeEtcdClient, testapi.Codec(), etcdtest.PathPrefix())
	return fakeEtcdClient, helper
}

func newStorage(t *testing.T) (allocator.Interface, allocator.Interface, *tools.FakeEtcdClient) {
	fakeEtcdClient, h := newHelper(t)

	mem := allocator.NewAllocationMap(100, "rangeSpecValue")
	etcd := NewEtcd(mem, "/ranges/serviceips", "serviceipallocation", h)

	return etcd, mem, fakeEtcdClient
}

func key() string {
	s := "/ranges/serviceips"
	return etcdtest.AddPrefix(s)
}

func TestEmpty(t *testing.T) {
	storage, _, ecli := newStorage(t)
	ecli.ExpectNotFoundGet(key())
	if _, err := storage.Allocate(1); fmt.Sprintf("%v", err) != "Cannot allocate resources of type serviceipallocation at this time" {
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

	// TODO: Reintroduce this test?
	//	other := ipallocator.NewCIDRRange(cidr)
	//
	//	allocation := &api.RangeAllocation{}
	//	if err := storage.(*allocator_etcd.Etcd).helper.ExtractObj(key(), allocation, false); err != nil {
	//		t.Fatal(err)
	//	}
	//	if allocation.ResourceVersion != "1" {
	//		t.Fatalf("%#v", allocation)
	//	}
	//	if allocation.Range != "192.168.1.0/24" {
	//		t.Errorf("unexpected stored Range: %s", allocation.Range)
	//	}
	//	if err := other.Restore(cidr, allocation.Data); err != nil {
	//		t.Fatal(err)
	//	}
	//	if !other.Has(net.ParseIP("192.168.1.2")) {
	//		t.Fatalf("could not restore allocated IP: %#v", other)
	//	}
	//
	//	other = ipallocator.NewCIDRRange(cidr)
	//	otherStorage := allocator_etcd.NewEtcd(other, storage.(*allocator_etcd.Etcd).helper)
	//	if err := otherStorage.Allocate(net.ParseIP("192.168.1.2")); err != ipallocator.ErrAllocated {
	//		t.Fatal(err)
	//	}
}
