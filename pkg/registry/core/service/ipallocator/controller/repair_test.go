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

package controller

import (
	"fmt"
	"net"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

type mockRangeRegistry struct {
	getCalled bool
	item      *api.RangeAllocation
	err       error

	updateCalled bool
	updated      *api.RangeAllocation
	updateErr    error
}

func (r *mockRangeRegistry) Get() (*api.RangeAllocation, error) {
	r.getCalled = true
	return r.item, r.err
}

func (r *mockRangeRegistry) CreateOrUpdate(alloc *api.RangeAllocation) error {
	r.updateCalled = true
	r.updated = alloc
	return r.updateErr
}

func TestRepair(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	_, cidr, _ := net.ParseCIDR("192.168.1.0/24")
	ipregistry := &mockRangeRegistry{
		item: &api.RangeAllocation{},
	}
	r := NewRepair(0, fakeClient.Core(), cidr, ipregistry)

	if err := r.RunOnce(); err != nil {
		t.Fatal(err)
	}
	if !ipregistry.updateCalled || ipregistry.updated == nil || ipregistry.updated.Range != cidr.String() || ipregistry.updated != ipregistry.item {
		t.Errorf("unexpected ipregistry: %#v", ipregistry)
	}

	ipregistry = &mockRangeRegistry{
		item:      &api.RangeAllocation{},
		updateErr: fmt.Errorf("test error"),
	}
	r = NewRepair(0, fakeClient.Core(), cidr, ipregistry)
	if err := r.RunOnce(); !strings.Contains(err.Error(), ": test error") {
		t.Fatal(err)
	}
}

func TestRepairEmpty(t *testing.T) {
	_, cidr, _ := net.ParseCIDR("192.168.1.0/24")
	previous := ipallocator.NewCIDRRange(cidr)
	previous.Allocate(net.ParseIP("192.168.1.10"))

	var dst api.RangeAllocation
	err := previous.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	fakeClient := fake.NewSimpleClientset()
	ipregistry := &mockRangeRegistry{
		item: &api.RangeAllocation{
			ObjectMeta: api.ObjectMeta{
				ResourceVersion: "1",
			},
			Range: dst.Range,
			Data:  dst.Data,
		},
	}
	r := NewRepair(0, fakeClient.Core(), cidr, ipregistry)
	if err := r.RunOnce(); err != nil {
		t.Fatal(err)
	}
	after := ipallocator.NewCIDRRange(cidr)
	if err := after.Restore(cidr, ipregistry.updated.Data); err != nil {
		t.Fatal(err)
	}
	if after.Has(net.ParseIP("192.168.1.10")) {
		t.Errorf("unexpected ipallocator state: %#v", after)
	}
}

func TestRepairWithExisting(t *testing.T) {
	_, cidr, _ := net.ParseCIDR("192.168.1.0/24")
	previous := ipallocator.NewCIDRRange(cidr)

	var dst api.RangeAllocation
	err := previous.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	fakeClient := fake.NewSimpleClientset(
		&api.Service{
			ObjectMeta: api.ObjectMeta{Namespace: "one", Name: "one"},
			Spec:       api.ServiceSpec{ClusterIP: "192.168.1.1"},
		},
		&api.Service{
			ObjectMeta: api.ObjectMeta{Namespace: "two", Name: "two"},
			Spec:       api.ServiceSpec{ClusterIP: "192.168.1.100"},
		},
		&api.Service{ // outside CIDR, will be dropped
			ObjectMeta: api.ObjectMeta{Namespace: "three", Name: "three"},
			Spec:       api.ServiceSpec{ClusterIP: "192.168.0.1"},
		},
		&api.Service{ // empty, ignored
			ObjectMeta: api.ObjectMeta{Namespace: "four", Name: "four"},
			Spec:       api.ServiceSpec{ClusterIP: ""},
		},
		&api.Service{ // duplicate, dropped
			ObjectMeta: api.ObjectMeta{Namespace: "five", Name: "five"},
			Spec:       api.ServiceSpec{ClusterIP: "192.168.1.1"},
		},
		&api.Service{ // headless
			ObjectMeta: api.ObjectMeta{Namespace: "six", Name: "six"},
			Spec:       api.ServiceSpec{ClusterIP: "None"},
		},
	)

	ipregistry := &mockRangeRegistry{
		item: &api.RangeAllocation{
			ObjectMeta: api.ObjectMeta{
				ResourceVersion: "1",
			},
			Range: dst.Range,
			Data:  dst.Data,
		},
	}
	r := NewRepair(0, fakeClient.Core(), cidr, ipregistry)
	if err := r.RunOnce(); err != nil {
		t.Fatal(err)
	}
	after := ipallocator.NewCIDRRange(cidr)
	if err := after.Restore(cidr, ipregistry.updated.Data); err != nil {
		t.Fatal(err)
	}
	if !after.Has(net.ParseIP("192.168.1.1")) || !after.Has(net.ParseIP("192.168.1.100")) {
		t.Errorf("unexpected ipallocator state: %#v", after)
	}
	if after.Free() != 252 {
		t.Errorf("unexpected ipallocator state: %#v", after)
	}
}
