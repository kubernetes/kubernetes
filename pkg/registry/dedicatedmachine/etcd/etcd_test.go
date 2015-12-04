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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "extensions")
	dedicatedStorage := NewREST(etcdStorage, generic.UndecoratedStorage)
	return dedicatedStorage, server
}

func newValidDedicatedMachine() *extensions.DedicatedMachine {
	return &extensions.DedicatedMachine{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.DedicatedMachineSpec{
			LabelValue: "bar",
		},
	}
}

var validDedicated = newValidDedicatedMachine()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	ds := newValidDedicatedMachine()
	ds.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		ds,
		// invalid (invalid labelvalue)
		&extensions.DedicatedMachine{
			Spec: extensions.DedicatedMachineSpec{
				LabelValue: "bar_",
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestUpdate(
		// valid
		newValidDedicatedMachine(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.DedicatedMachine)
			object.Spec.LabelValue = "a"
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.DedicatedMachine)
			object.UID = "newUID"
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.DedicatedMachine)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.DedicatedMachine)
			object.Spec.LabelValue = "bar*"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestDelete(newValidDedicatedMachine())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestGet(newValidDedicatedMachine())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestList(newValidDedicatedMachine())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestWatch(
		validDedicated,
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// notmatching fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": "foo"},
		},
	)
}
