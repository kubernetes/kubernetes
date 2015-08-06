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
	"net"
	"strings"
	"testing"

	"github.com/coreos/go-etcd/etcd"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/registry/service/allocator"
	allocator_etcd "k8s.io/kubernetes/pkg/registry/service/allocator/etcd"
	"k8s.io/kubernetes/pkg/registry/service/ipallocator"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

func newEtcdStorage(t *testing.T) (*tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, testapi.Codec(), etcdtest.PathPrefix())
	return fakeEtcdClient, etcdStorage
}

func newStorage(t *testing.T) (ipallocator.Interface, allocator.Interface, *tools.FakeEtcdClient) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	_, cidr, err := net.ParseCIDR("192.168.1.0/24")
	if err != nil {
		t.Fatal(err)
	}

	var backing allocator.Interface
	storage := ipallocator.NewAllocatorCIDRRange(cidr, func(max int, rangeSpec string) allocator.Interface {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		backing = mem
		etcd := allocator_etcd.NewEtcd(mem, "/ranges/serviceips", "serviceipallocation", etcdStorage)
		return etcd
	})

	return storage, backing, fakeEtcdClient
}

func key() string {
	s := "/ranges/serviceips"
	return etcdtest.AddPrefix(s)
}

func TestEmpty(t *testing.T) {
	storage, _, ecli := newStorage(t)
	ecli.ExpectNotFoundGet(key())
	if err := storage.Allocate(net.ParseIP("192.168.1.2")); !strings.Contains(err.Error(), "cannot allocate resources of type serviceipallocation at this time") {
		t.Fatal(err)
	}
}

func TestErrors(t *testing.T) {
	storage, _, _ := newStorage(t)
	if err := storage.Allocate(net.ParseIP("192.168.0.0")); err != ipallocator.ErrNotInRange {
		t.Fatal(err)
	}
}

func initialObject(ecli *tools.FakeEtcdClient) {
	_, cidr, _ := net.ParseCIDR("192.168.1.0/24")
	ecli.Data[key()] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				CreatedIndex:  1,
				ModifiedIndex: 2,
				Value: runtime.EncodeOrDie(testapi.Codec(), &api.RangeAllocation{
					Range: cidr.String(),
				}),
			},
		},
		E: nil,
	}
}

func TestStore(t *testing.T) {
	storage, r, ecli := newStorage(t)
	initialObject(ecli)

	if err := storage.Allocate(net.ParseIP("192.168.1.2")); err != nil {
		t.Fatal(err)
	}
	ok, err := r.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("Expected allocation to fail")
	}
	if err := storage.Allocate(net.ParseIP("192.168.1.2")); err != ipallocator.ErrAllocated {
		t.Fatal(err)
	}

	obj := ecli.Data[key()]
	if obj.R == nil || obj.R.Node == nil {
		t.Fatalf("%s is empty: %#v", key(), obj)
	}
	t.Logf("data: %#v", obj.R.Node)
}
