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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"

	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "namespaces"}
	namespaceStorage, _, _, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
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
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
	_, err := storage.Create(ctx, namespace, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
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

	// note that this ultimately may call validation
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
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
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
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx = genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace.Name)
	obj, immediate, err := storage.Delete(ctx, "foo", rest.ValidateAllObjectFunc, nil)
	if err != nil {
		t.Fatalf("unexpected error")
	}
	if immediate {
		t.Fatalf("unexpected immediate flag")
	}
	if ns, ok := obj.(*api.Namespace); !ok || obj == nil || ns == nil || ns.Name != namespace.Name {
		t.Fatalf("object not returned by delete")
	}
	// should still exist
	if _, err := storage.Get(ctx, "foo", &metav1.GetOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestUpdateDeletingNamespaceWithIncompleteMetadataFinalizers(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	key := "namespaces/foo"
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
	now := metav1.Now()
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "foo",
			DeletionTimestamp: &now,
			Finalizers:        []string{"example.com/foo"},
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{},
		},
		Status: api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx = genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace.Name)
	ns, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, _, err = storage.Update(ctx, "foo", rest.DefaultUpdatedObjectInfo(ns), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// should still exist
	_, err = storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestUpdateDeletingNamespaceWithIncompleteSpecFinalizers(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	key := "namespaces/foo"
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
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
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ns, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx = genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace.Name)
	if _, _, err = storage.Update(ctx, "foo", rest.DefaultUpdatedObjectInfo(ns), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// should still exist
	_, err = storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestUpdateDeletingNamespaceWithCompleteFinalizers(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	key := "namespaces/foo"
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
	now := metav1.Now()
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "foo",
			DeletionTimestamp: &now,
			Finalizers:        []string{"example.com/foo"},
		},
		Status: api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ns, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ns.(*api.Namespace).Finalizers = nil
	ctx = genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace.Name)
	if _, _, err = storage.Update(ctx, "foo", rest.DefaultUpdatedObjectInfo(ns), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// should not exist
	_, err = storage.Get(ctx, "foo", &metav1.GetOptions{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("expected NotFound, got %v", err)
	}
}

func TestFinalizeDeletingNamespaceWithCompleteFinalizers(t *testing.T) {
	// get finalize storage
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "namespaces"}
	storage, _, finalizeStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}

	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	defer finalizeStorage.store.DestroyFunc()
	key := "namespaces/foo"
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
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
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ns, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ns.(*api.Namespace).Spec.Finalizers = nil
	ctx = genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace.Name)
	if _, _, err = finalizeStorage.Update(ctx, "foo", rest.DefaultUpdatedObjectInfo(ns), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// should not exist
	_, err = storage.Get(ctx, "foo", &metav1.GetOptions{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("expected NotFound, got %v", err)
	}
}

func TestFinalizeDeletingNamespaceWithIncompleteMetadataFinalizers(t *testing.T) {
	// get finalize storage
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "namespaces"}
	storage, _, finalizeStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}

	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	defer finalizeStorage.store.DestroyFunc()
	key := "namespaces/foo"
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
	now := metav1.Now()
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "foo",
			DeletionTimestamp: &now,
			Finalizers:        []string{"example.com/foo"},
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{api.FinalizerKubernetes},
		},
		Status: api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ns, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ns.(*api.Namespace).Spec.Finalizers = nil
	ctx = genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace.Name)
	if _, _, err = finalizeStorage.Update(ctx, "foo", rest.DefaultUpdatedObjectInfo(ns), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// should still exist
	_, err = storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDeleteNamespaceWithCompleteFinalizers(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	key := "namespaces/foo"
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
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
	if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ctx = genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace.Name)
	if _, _, err := storage.Delete(ctx, "foo", rest.ValidateAllObjectFunc, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// should not exist
	_, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("expected NotFound, got %v", err)
	}
}

func TestDeleteWithGCFinalizers(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()

	propagationBackground := metav1.DeletePropagationBackground
	propagationForeground := metav1.DeletePropagationForeground
	propagationOrphan := metav1.DeletePropagationOrphan
	trueVar := true

	var tests = []struct {
		name          string
		deleteOptions *metav1.DeleteOptions

		existingFinalizers  []string
		remainingFinalizers map[string]bool
	}{
		{
			name:          "nil-with-orphan",
			deleteOptions: nil,
			existingFinalizers: []string{
				metav1.FinalizerOrphanDependents,
			},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerOrphanDependents: true,
			},
		},
		{
			name:          "nil-with-delete",
			deleteOptions: nil,
			existingFinalizers: []string{
				metav1.FinalizerDeleteDependents,
			},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerDeleteDependents: true,
			},
		},
		{
			name:                "nil-without-finalizers",
			deleteOptions:       nil,
			existingFinalizers:  []string{},
			remainingFinalizers: map[string]bool{},
		},
		{
			name: "propagation-background-with-orphan",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationBackground,
			},
			existingFinalizers: []string{
				metav1.FinalizerOrphanDependents,
			},
			remainingFinalizers: map[string]bool{},
		},
		{
			name: "propagation-background-with-delete",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationBackground,
			},
			existingFinalizers: []string{
				metav1.FinalizerDeleteDependents,
			},
			remainingFinalizers: map[string]bool{},
		},
		{
			name: "propagation-background-without-finalizers",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationBackground,
			},
			existingFinalizers:  []string{},
			remainingFinalizers: map[string]bool{},
		},
		{
			name: "propagation-foreground-with-orphan",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationForeground,
			},
			existingFinalizers: []string{
				metav1.FinalizerOrphanDependents,
			},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerDeleteDependents: true,
			},
		},
		{
			name: "propagation-foreground-with-delete",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationForeground,
			},
			existingFinalizers: []string{
				metav1.FinalizerDeleteDependents,
			},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerDeleteDependents: true,
			},
		},
		{
			name: "propagation-foreground-without-finalizers",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationForeground,
			},
			existingFinalizers: []string{},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerDeleteDependents: true,
			},
		},
		{
			name: "propagation-orphan-with-orphan",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationOrphan,
			},
			existingFinalizers: []string{
				metav1.FinalizerOrphanDependents,
			},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerOrphanDependents: true,
			},
		},
		{
			name: "propagation-orphan-with-delete",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationOrphan,
			},
			existingFinalizers: []string{
				metav1.FinalizerDeleteDependents,
			},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerOrphanDependents: true,
			},
		},
		{
			name: "propagation-orphan-without-finalizers",
			deleteOptions: &metav1.DeleteOptions{
				PropagationPolicy: &propagationOrphan,
			},
			existingFinalizers: []string{},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerOrphanDependents: true,
			},
		},
		{
			name: "orphan-dependents-with-orphan",
			deleteOptions: &metav1.DeleteOptions{
				OrphanDependents: &trueVar,
			},
			existingFinalizers: []string{
				metav1.FinalizerOrphanDependents,
			},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerOrphanDependents: true,
			},
		},
		{
			name: "orphan-dependents-with-delete",
			deleteOptions: &metav1.DeleteOptions{
				OrphanDependents: &trueVar,
			},
			existingFinalizers: []string{
				metav1.FinalizerDeleteDependents,
			},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerOrphanDependents: true,
			},
		},
		{
			name: "orphan-dependents-without-finalizers",
			deleteOptions: &metav1.DeleteOptions{
				OrphanDependents: &trueVar,
			},
			existingFinalizers: []string{},
			remainingFinalizers: map[string]bool{
				metav1.FinalizerOrphanDependents: true,
			},
		},
	}

	for _, test := range tests {
		key := "namespaces/" + test.name
		ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
		namespace := &api.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:       test.name,
				Finalizers: test.existingFinalizers,
			},
			Spec: api.NamespaceSpec{
				Finalizers: []api.FinalizerName{},
			},
			Status: api.NamespaceStatus{Phase: api.NamespaceActive},
		}
		if err := storage.store.Storage.Create(ctx, key, namespace, nil, 0, false); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		var obj runtime.Object
		var err error
		ctx = genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace.Name)
		if obj, _, err = storage.Delete(ctx, test.name, rest.ValidateAllObjectFunc, test.deleteOptions); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		ns, ok := obj.(*api.Namespace)
		if !ok {
			t.Errorf("unexpected object kind: %+v", obj)
		}
		if len(ns.Finalizers) != len(test.remainingFinalizers) {
			t.Errorf("%s: unexpected remaining finalizers: %v", test.name, ns.Finalizers)
		}
		for _, f := range ns.Finalizers {
			if test.remainingFinalizers[f] != true {
				t.Errorf("%s: unexpected finalizer %s", test.name, f)
			}
		}
	}
}

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.store.DestroyFunc()
	expected := []string{"ns"}
	registrytest.AssertShortNames(t, storage, expected)
}
