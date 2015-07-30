/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package event

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	etcdstorage "github.com/GoogleCloudPlatform/kubernetes/pkg/storage/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

var testTTL uint64 = 60

func NewTestEventEtcdRegistry(t *testing.T) (*tools.FakeEtcdClient, generic.Registry) {
	f := tools.NewFakeEtcdClient(t)
	f.TestIndex = true

	s := etcdstorage.NewEtcdStorage(f, testapi.Codec(), etcdtest.PathPrefix())
	return f, NewEtcdRegistry(s, testTTL)
}

func TestEventCreate(t *testing.T) {
	eventA := &api.Event{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Reason:     "forTesting",
	}
	eventB := &api.Event{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Reason:     "forTesting",
	}

	nodeWithEventA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), eventA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
				TTL:           int64(testTTL),
			},
		},
		E: nil,
	}

	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	ctx := api.NewDefaultContext()
	key := "foo"
	path, err := etcdgeneric.NamespaceKeyFunc(ctx, "/events", key)
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
			existing: emptyNode,
			expect:   nodeWithEventA,
			toCreate: eventA,
			errOK:    func(err error) bool { return err == nil },
		},
		"preExisting": {
			existing: nodeWithEventA,
			expect:   nodeWithEventA,
			toCreate: eventB,
			errOK:    errors.IsAlreadyExists,
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestEventEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.CreateWithName(ctx, key, item.toCreate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

func TestEventUpdate(t *testing.T) {
	eventA := &api.Event{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Reason:     "forTesting",
	}
	eventB := &api.Event{
		ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: api.NamespaceDefault},
		Reason:     "for testing again",
	}
	eventC := &api.Event{
		ObjectMeta: api.ObjectMeta{Name: "pan", Namespace: api.NamespaceDefault, ResourceVersion: "1"},
		Reason:     "for testing again something else",
	}

	nodeWithEventA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), eventA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
				TTL:           int64(testTTL),
			},
		},
		E: nil,
	}

	nodeWithEventB := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), eventB),
				ModifiedIndex: 1,
				CreatedIndex:  1,
				TTL:           int64(testTTL),
			},
		},
		E: nil,
	}

	nodeWithEventC := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), eventC),
				ModifiedIndex: 1,
				CreatedIndex:  1,
				TTL:           int64(testTTL),
			},
		},
		E: nil,
	}

	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	ctx := api.NewDefaultContext()
	key := "foo"
	path, err := etcdgeneric.NamespaceKeyFunc(ctx, "/events", key)
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
			existing: emptyNode,
			expect:   nodeWithEventA,
			toUpdate: eventA,
			errOK:    func(err error) bool { return err == nil },
		},
		"doesNotExist2": {
			existing: emptyNode,
			expect:   nodeWithEventB,
			toUpdate: eventB,
			errOK:    func(err error) bool { return err == nil },
		},
		"replaceExisting": {
			existing: nodeWithEventA,
			expect:   nodeWithEventC,
			toUpdate: eventC,
			errOK:    func(err error) bool { return err == nil },
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestEventEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.UpdateWithName(ctx, key, item.toUpdate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectGoPrintDiff(e, a))
		}
	}
}
