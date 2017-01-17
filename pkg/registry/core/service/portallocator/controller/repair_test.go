/*
Copyright 2016 The Kubernetes Authors.

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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
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
	registry := &mockRangeRegistry{
		item: &api.RangeAllocation{Range: "100-200"},
	}
	pr, _ := net.ParsePortRange(registry.item.Range)
	r := NewRepair(0, fakeClient.Core(), *pr, registry)

	if err := r.RunOnce(); err != nil {
		t.Fatal(err)
	}
	if !registry.updateCalled || registry.updated == nil || registry.updated.Range != pr.String() || registry.updated != registry.item {
		t.Errorf("unexpected registry: %#v", registry)
	}

	registry = &mockRangeRegistry{
		item:      &api.RangeAllocation{Range: "100-200"},
		updateErr: fmt.Errorf("test error"),
	}
	r = NewRepair(0, fakeClient.Core(), *pr, registry)
	if err := r.RunOnce(); !strings.Contains(err.Error(), ": test error") {
		t.Fatal(err)
	}
}

func TestRepairLeak(t *testing.T) {
	pr, _ := net.ParsePortRange("100-200")
	previous := portallocator.NewPortAllocator(*pr)
	previous.Allocate(111)

	var dst api.RangeAllocation
	err := previous.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	fakeClient := fake.NewSimpleClientset()
	registry := &mockRangeRegistry{
		item: &api.RangeAllocation{
			ObjectMeta: metav1.ObjectMeta{
				ResourceVersion: "1",
			},
			Range: dst.Range,
			Data:  dst.Data,
		},
	}

	r := NewRepair(0, fakeClient.Core(), *pr, registry)
	// Run through the "leak detection holdoff" loops.
	for i := 0; i < (numRepairsBeforeLeakCleanup - 1); i++ {
		if err := r.RunOnce(); err != nil {
			t.Fatal(err)
		}
		after, err := portallocator.NewFromSnapshot(registry.updated)
		if err != nil {
			t.Fatal(err)
		}
		if !after.Has(111) {
			t.Errorf("expected portallocator to still have leaked port")
		}
	}
	// Run one more time to actually remove the leak.
	if err := r.RunOnce(); err != nil {
		t.Fatal(err)
	}
	after, err := portallocator.NewFromSnapshot(registry.updated)
	if err != nil {
		t.Fatal(err)
	}
	if after.Has(111) {
		t.Errorf("expected portallocator to not have leaked port")
	}
}

func TestRepairWithExisting(t *testing.T) {
	pr, _ := net.ParsePortRange("100-200")
	previous := portallocator.NewPortAllocator(*pr)

	var dst api.RangeAllocation
	err := previous.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	fakeClient := fake.NewSimpleClientset(
		&api.Service{
			ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "one"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{NodePort: 111}},
			},
		},
		&api.Service{
			ObjectMeta: metav1.ObjectMeta{Namespace: "two", Name: "two"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{NodePort: 122}, {NodePort: 133}},
			},
		},
		&api.Service{ // outside range, will be dropped
			ObjectMeta: metav1.ObjectMeta{Namespace: "three", Name: "three"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{NodePort: 201}},
			},
		},
		&api.Service{ // empty, ignored
			ObjectMeta: metav1.ObjectMeta{Namespace: "four", Name: "four"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{}},
			},
		},
		&api.Service{ // duplicate, dropped
			ObjectMeta: metav1.ObjectMeta{Namespace: "five", Name: "five"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{NodePort: 111}},
			},
		},
	)

	registry := &mockRangeRegistry{
		item: &api.RangeAllocation{
			ObjectMeta: metav1.ObjectMeta{
				ResourceVersion: "1",
			},
			Range: dst.Range,
			Data:  dst.Data,
		},
	}
	r := NewRepair(0, fakeClient.Core(), *pr, registry)
	if err := r.RunOnce(); err != nil {
		t.Fatal(err)
	}
	after, err := portallocator.NewFromSnapshot(registry.updated)
	if err != nil {
		t.Fatal(err)
	}
	if !after.Has(111) || !after.Has(122) || !after.Has(133) {
		t.Errorf("unexpected portallocator state: %#v", after)
	}
	if free := after.Free(); free != 98 {
		t.Errorf("unexpected portallocator state: %d free", free)
	}
}
