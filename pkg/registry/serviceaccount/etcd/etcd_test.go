/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1}
	return NewREST(restOptions), server
}

func validNewServiceAccount(name string) *api.ServiceAccount {
	return &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Secrets: []api.ObjectReference{},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	serviceAccount := validNewServiceAccount("foo")
	serviceAccount.ObjectMeta = api.ObjectMeta{GenerateName: "foo-"}
	test.TestCreate(
		// valid
		serviceAccount,
		// invalid
		&api.ServiceAccount{},
		&api.ServiceAccount{
			ObjectMeta: api.ObjectMeta{Name: "name with spaces"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		validNewServiceAccount("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.ServiceAccount)
			object.Secrets = []api.ObjectReference{{}}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ReturnDeletedObject()
	test.TestDelete(validNewServiceAccount("foo"))
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestGet(validNewServiceAccount("foo"))
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestList(validNewServiceAccount("foo"))
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestWatch(
		validNewServiceAccount("foo"),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": "foo"},
		},
	)
}
