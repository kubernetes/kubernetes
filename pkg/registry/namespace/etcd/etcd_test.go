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
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t, "")
	storage, _, _ := NewREST(etcdStorage)
	return storage, fakeClient
}

func validNewNamespace() *api.Namespace {
	return &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	}
}

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	namespace := validNewNamespace()
	namespace.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		namespace,
		// invalid
		&api.Namespace{
			ObjectMeta: api.ObjectMeta{Name: "bad value"},
		},
	)
}

func TestCreateSetsFields(t *testing.T) {
	storage, _ := newStorage(t)
	namespace := validNewNamespace()
	ctx := api.NewContext()
	_, err := storage.Create(ctx, namespace)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	object, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actual := object.(*api.Namespace)
	if actual.Name != namespace.Name {
		t.Errorf("unexpected namespace: %#v", actual)
	}
	if len(actual.UID) == 0 {
		t.Errorf("expected namespace UID to be set: %#v", actual)
	}
	if actual.Status.Phase != api.NamespaceActive {
		t.Errorf("expected namespace phase to be set to active, but %v", actual.Status.Phase)
	}
}

func TestDelete(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewNamespace())
}

func TestGet(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestGet(validNewNamespace())
}

func TestList(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestList(validNewNamespace())
}

func TestWatch(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestWatch(
		validNewNamespace(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
			{"name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

func TestDeleteNamespaceWithIncompleteFinalizers(t *testing.T) {
	storage, fakeClient := newStorage(t)
	key := etcdtest.AddPrefix("namespaces/foo")
	ctx := api.NewContext()
	now := unversioned.Now()
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:              "foo",
			DeletionTimestamp: &now,
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{api.FinalizerKubernetes},
		},
		Status: api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	if _, err := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), namespace), 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := storage.Delete(ctx, "foo", nil); err == nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestDeleteNamespaceWithCompleteFinalizers(t *testing.T) {
	storage, fakeClient := newStorage(t)
	key := etcdtest.AddPrefix("namespaces/foo")
	ctx := api.NewContext()
	now := unversioned.Now()
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:              "foo",
			DeletionTimestamp: &now,
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{},
		},
		Status: api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	if _, err := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), namespace), 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := storage.Delete(ctx, "foo", nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
