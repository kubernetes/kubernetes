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

package storage

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "namespaces"}
	namespaceStorage, _, _ := NewREST(restOptions)
	return namespaceStorage, server
}

func validNewNamespace() *api.Namespace {
	return &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	test := genericregistrytest.New(t, storage.store).ClusterScope()
	namespace := validNewNamespace()
	namespace.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		namespace,
		// invalid
		&api.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: "bad value"},
		},
	)
}

func TestCreateSetsFields(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	namespace := validNewNamespace()
	ctx := genericapirequest.NewContext()
	_, err := storage.Create(ctx, namespace, rest.ValidateAllObjectFunc, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	object, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
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
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	test := genericregistrytest.New(t, storage.store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewNamespace())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	test := genericregistrytest.New(t, storage.store).ClusterScope()
	test.TestGet(validNewNamespace())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	test := genericregistrytest.New(t, storage.store).ClusterScope()
	test.TestList(validNewNamespace())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	test := genericregistrytest.New(t, storage.store).ClusterScope()
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
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	key := "namespaces/foo"
	ctx := genericapirequest.NewContext()
	now := metav1.Now()
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "foo",
			DeletionTimestamp: &now,
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{api.FinalizerKubernetes},
		},
		Status: api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, _, err := storage.Delete(ctx, "foo", nil); err == nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestDeleteNamespaceWithCompleteFinalizers(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	key := "namespaces/foo"
	ctx := genericapirequest.NewContext()
	now := metav1.Now()
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "foo",
			DeletionTimestamp: &now,
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{},
		},
		Status: api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, _, err := storage.Delete(ctx, "foo", nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	expected := []string{"ns"}
	registrytest.AssertShortNames(t, storage, expected)
}
