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

package lock

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"
)

type testRegistry struct {
	*registrytest.GenericRegistry
}

func NewTestREST() (testRegistry, *REST) {
	reg := testRegistry{registrytest.NewGeneric(nil)}
	return reg, NewStorage(reg)
}

func testLock(name string, heldby string, ttl uint64, ns string) *api.Lock {
	return &api.Lock{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: api.LockSpec{
			HeldBy: heldby,
			LeaseTime: ttl,
		},
	}
}

func TestRESTCreate(t *testing.T) {
	table := []struct {
		ctx   api.Context
		lock *api.Lock
		valid bool
	}{
		{
			ctx:   api.NewDefaultContext(),
			lock: testLock("lock1", "app1", 30, api.NamespaceDefault),
			valid: true,
		}, {
			ctx:   api.NewDefaultContext(),
			lock: testLock("lock2", "app1", 45, api.NamespaceDefault),
			valid: true,
		}, {
			ctx:   api.NewDefaultContext(),
			lock: testLock("lock1", "app2", 15, api.NamespaceDefault),
			valid: true,
		}, {
			ctx:   api.WithNamespace(api.NewContext(), "nondefault"),
			lock: testLock("lock1", "app1", 30, "nondefault"),
			valid: true,
		}, {
			ctx:   api.WithNamespace(api.NewContext(), "nondefault"),
			lock: testLock("lock1", "app1", 30, api.NamespaceDefault),
			valid: false,
		}, {
			ctx:   api.NewDefaultContext(),
			lock: testLock("", "app1", 30, api.NamespaceDefault),
			valid: false,
		}, {
			ctx:   api.NewDefaultContext(),
			lock: testLock("lock1", "", 30, api.NamespaceDefault),
			valid: false,
		}, {
			ctx:   api.NewDefaultContext(),
			lock: testLock("lock1", "app1", 0, api.NamespaceDefault),
			valid: false,
		},
	}

	for _, item := range table {
		_, storage := NewTestREST()
		c, err := storage.Create(item.ctx, item.lock)
		if !item.valid {
			if err == nil {
				ctxNS := api.NamespaceValue(item.ctx)
				t.Errorf("unexpected non-error for %v (%v, %v)", item.lock.Name, ctxNS, item.lock.Namespace)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: Unexpected error %v", item.lock.Name, err)
			continue
		}
		if !api.HasObjectMetaSystemFieldValues(&item.lock.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		if e, a := item.lock, c; !reflect.DeepEqual(e, a) {
			t.Errorf("diff: %s", util.ObjectDiff(e, a))
		}
		if len(c.(*api.Lock).Spec.AcquiredTime) == 0 {
			t.Errorf("AcquiredTime for lock is empty")
		}
		if c.(*api.Lock).Spec.AcquiredTime != c.(*api.Lock).Spec.RenewTime {
			t.Errorf("AcquiredTime for lock does not match RenewTime")
		}
		// Ensure we implement the interface
		_ = rest.Watcher(storage)
	}
}

func TestRESTUpdate(t *testing.T) {
	_, rest := NewTestREST()
	lockA := testLock("lock1", "app1", 30, api.NamespaceDefault)
	_, err := rest.Create(api.NewDefaultContext(), lockA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), lockA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := lockA, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
	atime := got.(*api.Lock).Spec.AcquiredTime
	rtime := got.(*api.Lock).Spec.RenewTime
	if len(atime) == 0 {
		t.Errorf("AcquiredTime for lock is empty")
	}
	if atime != rtime {
		t.Errorf("AcquiredTime for lock does not match RenewTime")
	}
	lockB := testLock("lock1", "app2", 30, api.NamespaceDefault)
	_, _, err = rest.Update(api.NewDefaultContext(), lockB)
	if err == nil {
		t.Fatalf("Unexpected success updating lock with %v", lockB)
	}
	_, _, err = rest.Update(api.NewDefaultContext(), lockA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got2, err := rest.Get(api.NewDefaultContext(), lockA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := lockA, got2; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
	if atime != got2.(*api.Lock).Spec.AcquiredTime {
		t.Fatalf("Lock acquired time changed from %s to %s", atime, got2.(*api.Lock).Spec.AcquiredTime)
	}
	if rtime == got2.(*api.Lock).Spec.RenewTime {
		t.Fatalf("Lock renew time did not change from %s", rtime)
	}
}

func TestRESTDelete(t *testing.T) {
	_, rest := NewTestREST()
	lock := testLock("lock1", "app1", 30, api.NamespaceDefault)
	_, err := rest.Create(api.NewDefaultContext(), lock)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	c, err := rest.Delete(api.NewDefaultContext(), lock.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if stat := c.(*api.Status); stat.Status != api.StatusSuccess {
		t.Errorf("unexpected status: %v", stat)
	}
}

func TestRESTGet(t *testing.T) {
	_, rest := NewTestREST()
	lock := testLock("lock1", "app1", 30, api.NamespaceDefault)
	_, err := rest.Create(api.NewDefaultContext(), lock)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), lock.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := lock, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTgetAttrs(t *testing.T) {
	name := "lock1"
	hb := "app1"
	lt := 30
	at := "time1"
	rt := "time2"
	_, rest := NewTestREST()
	lock := &api.Lock{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec: api.LockSpec{
			HeldBy:          hb,
			LeaseTime:       uint64(lt),
			AcquiredTime:    at,
			RenewTime:       rt,
		},
	}
	label, field, err := rest.getAttrs(lock)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := label, (labels.Set{}); !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
	expect := fields.Set{
		"metadata.name":  name,
		"spec.heldby":    hb,
		"spec.duration":  string(lt),
		"spec.atime":     at,
		"spec.rtime":     rt,
	}
	if e, a := expect, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTList(t *testing.T) {
	reg, rest := NewTestREST()
	lockA := &api.Lock{
		ObjectMeta: api.ObjectMeta{Name: "lock1"},
		Spec: api.LockSpec{
			HeldBy:          "app1",
			LeaseTime:       uint64(30),
			AcquiredTime:    "lock1 app1 at",
			RenewTime:       "lock1 app1 rt",
		},
	}
	lockB := &api.Lock{
		ObjectMeta: api.ObjectMeta{Name: "lock2"},
		Spec: api.LockSpec{
			HeldBy:          "app1",
			LeaseTime:       uint64(15),
			AcquiredTime:    "lock2 app1 at",
			RenewTime:       "lock2 app1 rt",
		},
	}
	lockC := &api.Lock{
		ObjectMeta: api.ObjectMeta{Name: "lock3"},
		Spec: api.LockSpec{
			HeldBy:          "app2",
			LeaseTime:       uint64(45),
			AcquiredTime:    "lock3 app2 at",
			RenewTime:       "lock3 app2 rt",
		},
	}
	reg.ObjectList = &api.LockList{
		Items: []api.Lock{*lockA, *lockB, *lockC},
	}
	got, err := rest.List(api.NewContext(), labels.Everything(), fields.Set{"spec.heldby": "app1"}.AsSelector())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expect := &api.LockList{
		Items: []api.Lock{*lockA, *lockB},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTWatch(t *testing.T) {
	lockA := &api.Lock{
		ObjectMeta: api.ObjectMeta{Name: "lock1"},
		Spec: api.LockSpec{
			HeldBy:          "app1",
			LeaseTime:       uint64(30),
			AcquiredTime:    "lock1 app1 at",
			RenewTime:       "lock1 app1 rt",
		},
	}
	reg, rest := NewTestREST()
	wi, err := rest.Watch(api.NewContext(), labels.Everything(), fields.Everything(), "0")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	go func() {
		reg.Broadcaster.Action(watch.Added, lockA)
	}()
	got := <-wi.ResultChan()
	if e, a := lockA, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}
