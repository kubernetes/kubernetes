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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pool"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
)

func newHelper(t *testing.T) (*tools.FakeEtcdClient, tools.EtcdHelper) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	helper := tools.NewEtcdHelper(fakeEtcdClient, testapi.Codec(), etcdtest.PathPrefix())
	return fakeEtcdClient, helper
}

func Test_EtcdPoolAllocator_Allocate(t *testing.T) {
	driver := &pool.TestPoolDriver{Items: []string{"a", "b", "c"}}
	pa := &EtcdPoolAllocator{}
	_, etcd := newHelper(t)
	pa.Init(driver, &etcd, "/base/")

	pool.TestPoolAllocatorAllocate(t, pa)
}

func Test_EtcdPoolAllocator_AllocateNext(t *testing.T) {
	driver := &pool.TestPoolDriver{Items: []string{"a", "b", "c"}}
	pa := &EtcdPoolAllocator{}
	_, etcd := newHelper(t)
	pa.Init(driver, &etcd, "/base/")

	// Turn off random allocation attempts, so we just allocate in sequence
	pa.DisableRandomAllocation()

	pool.TestPoolAllocatorAllocateNext(t, pa)
}

func Test_EtcdPoolAllocator_Release(t *testing.T) {
	driver := &pool.TestPoolDriver{Items: []string{"a", "b", "c"}}
	pa := &EtcdPoolAllocator{}
	_, etcd := newHelper(t)
	pa.Init(driver, &etcd, "/base/")

	// Turn off random allocation attempts, so we just allocate in sequence
	pa.DisableRandomAllocation()

	pool.TestPoolAllocatorRelease(t, pa)
}
