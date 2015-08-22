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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestLockEtcdRegistry(t *testing.T) (*tools.FakeEtcdClient, generic.Registry) {
	f := tools.NewFakeEtcdClient(t)
	f.TestIndex = true
	s := etcdstorage.NewEtcdStorage(f, testapi.Codec(), etcdtest.PathPrefix())
	return f, NewEtcdRegistry(s)
}

var testTTL uint64 = 30
var lockName string = "aprocess"
var appName string = "aprocess"

func TestLockCreate(t *testing.T) {
	lock := &api.Lock{
		ObjectMeta: api.ObjectMeta{
			Name: lockName,
			Namespace: api.NamespaceDefault,
		},
		Spec: api.LockSpec{
			HeldBy: appName,
			LeaseTime: testTTL,
		},
	}

	existingLock := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), lock),
				ModifiedIndex: 1,
				CreatedIndex:  1,
				TTL:           int64(testTTL),
			},
		},
		E: nil,
	}

	empty := tools.EtcdResponseWithError{
                R: &etcd.Response{},
                E: tools.EtcdErrorNotFound,
	}

	ctx := api.NewDefaultContext()
	path, err := etcdgeneric.NamespaceKeyFunc(ctx, "/locks", lockName)
	path = etcdtest.AddPrefix(path)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   tools.EtcdResponseWithError
		toCreate runtime.Object
		errOK    func(error) bool
	}{
		"normal": {
			existing: empty,
			expect:   existingLock,
			toCreate: lock,
			errOK:    func(err error) bool { return err == nil },
		},
		"preExisting": {
			existing: existingLock,
			expect:   existingLock,
			toCreate: lock,
			errOK:    errors.IsAlreadyExists,
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestLockEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.CreateWithName(ctx, lockName, item.toCreate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

func TestLockUpdate(t *testing.T) {
	atime := time.Now().String()
	utime := time.Now().Add(time.Duration(testTTL) * time.Second).String()

	lock := &api.Lock{
		ObjectMeta: api.ObjectMeta{
			Name: lockName,
			Namespace: api.NamespaceDefault,
		},
		Spec: api.LockSpec{
			HeldBy: appName,
			LeaseTime: testTTL,
			AcquiredTime: atime,
			RenewTime: atime,
		},
	}

	updatedLock := &api.Lock{
		ObjectMeta: api.ObjectMeta{
			Name: lockName,
			Namespace: api.NamespaceDefault,
			ResourceVersion: "1",
		},
		Spec: api.LockSpec{
			HeldBy: appName,
			LeaseTime: testTTL,
			AcquiredTime: atime,
			RenewTime: utime,
		},
	}


	nodeWithLock := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), lock),
				ModifiedIndex: 1,
				CreatedIndex:  1,
				TTL:           int64(testTTL),
			},
		},
		E: nil,
	}

	nodeWithUpdatedLock := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), updatedLock),
				ModifiedIndex: 1,
				CreatedIndex:  1,
				TTL:           int64(testTTL),
			},
		},
		E: nil,
	}

	empty := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
}

	ctx := api.NewDefaultContext()
	path, err := etcdgeneric.NamespaceKeyFunc(ctx, "/locks", lockName)
	path = etcdtest.AddPrefix(path)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   tools.EtcdResponseWithError
		toUpdate runtime.Object
		errOK    func(error) bool
	}{
		"doesNotExist": {
			existing: empty,
			expect:   nodeWithLock,
			toUpdate: lock,
			errOK:    func(err error) bool { return err == nil },
		},
		"replaceExisting": {
			existing: nodeWithLock,
			expect:   nodeWithUpdatedLock,
			toUpdate: updatedLock,
			errOK:    func(err error) bool { return err == nil },
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestLockEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.UpdateWithName(ctx, lockName, item.toUpdate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectGoPrintDiff(e, a))
		}
	}
}
