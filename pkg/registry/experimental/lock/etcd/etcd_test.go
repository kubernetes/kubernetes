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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

// TODO: Refactor to use standard test drivers

func newEtcdStorage(t *testing.T) (*tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, latest.Codec, etcdtest.PathPrefix())
	return fakeEtcdClient, etcdStorage
}

func testLock(name string, heldby string, ttl int64, ns string) *expapi.Lock {
	return &expapi.Lock{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: expapi.LockSpec{
			HeldBy:       heldby,
			LeaseSeconds: ttl,
		},
	}
}

func TestRESTCreate(t *testing.T) {
	table := []struct {
		ctx   api.Context
		lock  *expapi.Lock
		valid bool
	}{
		{
			ctx:   api.NewDefaultContext(),
			lock:  testLock("lock1", "app1", 30, api.NamespaceDefault),
			valid: true,
		}, {
			ctx:   api.WithNamespace(api.NewContext(), "nondefault"),
			lock:  testLock("lock1", "app1", 30, "nondefault"),
			valid: true,
		}, {
			ctx:   api.WithNamespace(api.NewContext(), "nondefault"),
			lock:  testLock("lock1", "app1", 30, api.NamespaceDefault),
			valid: false,
		}, {
			ctx:   api.NewDefaultContext(),
			lock:  testLock("", "app1", 30, api.NamespaceDefault),
			valid: false,
		}, {
			ctx:   api.NewDefaultContext(),
			lock:  testLock("lock1", "", 30, api.NamespaceDefault),
			valid: false,
		}, {
			ctx:   api.NewDefaultContext(),
			lock:  testLock("lock1", "app1", 0, api.NamespaceDefault),
			valid: false,
		},
	}

	for _, item := range table {
		_, etcdStorage := newEtcdStorage(t)
		storage := NewStorage(etcdStorage)
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
		if c.(*expapi.Lock).Status.AcquiredTime.IsZero() == true {
			t.Fatalf("Lock doesn't have an acquired time set")
		}
		if c.(*expapi.Lock).Status.LastRenewalTime.IsZero() == true {
			t.Fatalf("Lock doesn't have a renew time set")
		}
		if len(c.(*expapi.Lock).Status.AcquiredTime.String()) == 0 {
			t.Errorf("AcquiredTime for lock is empty")
		}
		if c.(*expapi.Lock).Status.AcquiredTime != c.(*expapi.Lock).Status.LastRenewalTime {
			t.Errorf("AcquiredTime for lock does not match LastRenewalTime")
		}
	}
}

func TestRESTUpdate(t *testing.T) {
	_, etcdStorage := newEtcdStorage(t)
	rest := NewStorage(etcdStorage)
	lockSpec := testLock("lock1", "app1", 30, api.NamespaceDefault)
	_, err := rest.Create(api.NewDefaultContext(), lockSpec)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), lockSpec.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	lock := got.(*expapi.Lock)
	lock.Spec.HeldBy = "app2"
	_, _, err = rest.Update(api.NewDefaultContext(), lock)
	if err == nil {
		t.Fatalf("Unexpected success updating lock with %v", lock)
	}

	atime := lock.Status.AcquiredTime
	rtime := lock.Status.LastRenewalTime
	lockSpec.ResourceVersion = lock.ResourceVersion
	update, _, err := rest.Update(api.NewDefaultContext(), lockSpec)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if update.(*expapi.Lock).Status.AcquiredTime.IsZero() == true {
		t.Fatalf("Updated lock doesn't have an acquired time set")
	}
	if update.(*expapi.Lock).Status.LastRenewalTime.IsZero() == true {
		t.Fatalf("Updated lock doesn't have a renew time set")
	}
	got2, err := rest.Get(api.NewDefaultContext(), lockSpec.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if got2.(*expapi.Lock).Status.AcquiredTime.IsZero() == true {
		t.Fatalf("lock gotten after update doesn't have an acquired time set")
	}
	if atime != got2.(*expapi.Lock).Status.AcquiredTime {
		t.Fatalf("Lock acquired time changed from %s to %s", atime, got2.(*expapi.Lock).Status.AcquiredTime)
	}
	if got2.(*expapi.Lock).Status.LastRenewalTime.IsZero() == true {
		t.Fatalf("lock gotten after update doesn't have a renew time set")
	}
	if rtime == got2.(*expapi.Lock).Status.LastRenewalTime {
		t.Fatalf("Lock renew time did not change from %s", rtime)
	}
}

func TestRESTDelete(t *testing.T) {
	_, etcdStorage := newEtcdStorage(t)
	rest := NewStorage(etcdStorage)
	lock := testLock("lock1", "app1", 30, api.NamespaceDefault)
	_, err := rest.Create(api.NewDefaultContext(), lock)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	c, err := rest.Delete(api.NewDefaultContext(), lock.Name, nil)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if stat := c.(*api.Status); stat.Status != api.StatusSuccess {
		t.Errorf("unexpected status: %v", stat)
	}
}

func TestRESTGet(t *testing.T) {
	_, etcdStorage := newEtcdStorage(t)
	rest := NewStorage(etcdStorage)
	lock := testLock("lock1", "app1", 30, api.NamespaceDefault)
	_, err := rest.Create(api.NewDefaultContext(), lock)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), lock.Name)
	gotLock := got.(*expapi.Lock)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if lock.Name != gotLock.Name {
		t.Fatalf("Lock name %s does not match expected %s", lock.Name, gotLock.Name)
	}
	if lock.Namespace != gotLock.Namespace {
		t.Fatalf("Lock namespace %s does not match expected %s", lock.Namespace, gotLock.Namespace)
	}
	if lock.Spec.HeldBy != gotLock.Spec.HeldBy {
		t.Fatalf("Lock heldby %s does not match expected %s", lock.Spec.HeldBy, gotLock.Spec.HeldBy)
	}
	if lock.Spec.LeaseSeconds != gotLock.Spec.LeaseSeconds {
		t.Fatalf("Lock leasttime %d does not match expected %d", lock.Spec.LeaseSeconds, gotLock.Spec.LeaseSeconds)
	}
}

func TestRESTList(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	rest := NewStorage(etcdStorage)
	fakeEtcdClient.ChangeIndex = 1
	ctx := api.NewDefaultContext()
	key := rest.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	lockA := &expapi.Lock{
		ObjectMeta: api.ObjectMeta{Name: "lock1"},
		Spec: expapi.LockSpec{
			HeldBy:       "app1",
			LeaseSeconds: 30,
		},
	}
	lockB := &expapi.Lock{
		ObjectMeta: api.ObjectMeta{Name: "lock2"},
		Spec: expapi.LockSpec{
			HeldBy:       "app1",
			LeaseSeconds: 15,
		},
	}
	lockC := &expapi.Lock{
		ObjectMeta: api.ObjectMeta{Name: "lock3"},
		Spec: expapi.LockSpec{
			HeldBy:       "app2",
			LeaseSeconds: 45,
		},
	}
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, lockA),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, lockB),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, lockC),
					},
				},
			},
		},
	}

	gotObj, err := rest.List(ctx, labels.Everything(), fields.Set{"spec.heldBy": "app1"}.AsSelector())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got := gotObj.(*expapi.LockList)
	if len(got.Items) != 2 {
		t.Errorf("Expected 2 list objects, got %d", len(got.Items))
	}
	if got.Items[0].Name != "lock1" {
		t.Errorf("Unexpected lock: %#v", got.Items[0])
	}
	if got.Items[1].Name != "lock2" {
		t.Errorf("Unexpected lock: %#v", got.Items[0])
	}
}

func TestRESTWatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	rest := NewStorage(etcdStorage)
	watching, err := rest.Watch(ctx,
		labels.SelectorFromSet(labels.Set{"name": "lock1"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeEtcdClient.WaitForWatchCompletion()

	lockA := &expapi.Lock{
		ObjectMeta: api.ObjectMeta{
			Name: "lock1",
			Labels: map[string]string{
				"name": "lock1",
			},
		},
		Spec: expapi.LockSpec{
			HeldBy:       "app1",
			LeaseSeconds: 30,
		},
		Status: expapi.LockStatus{
			AcquiredTime:    util.NewTime(time.Now()),
			LastRenewalTime: util.NewTime(time.Now()),
		},
	}
	lockBytes, _ := latest.Codec.Encode(lockA)
	fakeEtcdClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(lockBytes),
		},
	}
	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	case <-time.After(time.Millisecond * 100):
		t.Error("unexpected timeout from result channel")
	}
	watching.Stop()
}
