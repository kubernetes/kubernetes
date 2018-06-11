/*
Copyright 2018 The Kubernetes Authors.

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

	storageapiv1alpha1 "k8s.io/api/storage/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/api/testapi"
	api "k8s.io/kubernetes/pkg/apis/core"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, storageapi.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "volumesnapshots",
	}
	volumeSnapshotStorage, statusStorage := NewREST(restOptions)
	return volumeSnapshotStorage, statusStorage, server
}

func validNewVolumeSnapshot(name, ns string) *storageapi.VolumeSnapshot {
	vs := &storageapi.VolumeSnapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: storageapi.VolumeSnapshotSpec{
			PersistentVolumeClaimName: "bar",
		},
	}
	return vs
}

func TestCreate(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	vs := validNewVolumeSnapshot("foo", metav1.NamespaceDefault)
	vs.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		vs,
		// invalid
		&storageapi.VolumeSnapshot{
			ObjectMeta: metav1.ObjectMeta{Name: "*BadName!"},
		},
	)
}

func TestUpdate(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		validNewVolumeSnapshot("foo", metav1.NamespaceDefault),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*storageapi.VolumeSnapshot)
			object.Spec.SnapshotDataName = "test"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ReturnDeletedObject()
	test.TestDelete(validNewVolumeSnapshot("foo", metav1.NamespaceDefault))
}

func TestGet(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(validNewVolumeSnapshot("foo", metav1.NamespaceDefault))
}

func TestList(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(validNewVolumeSnapshot("foo", metav1.NamespaceDefault))
}

func TestWatch(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validNewVolumeSnapshot("foo", metav1.NamespaceDefault),
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

func TestUpdateStatus(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericapirequest.NewDefaultContext()

	key, _ := storage.KeyFunc(ctx, "foo")
	pvcStart := validNewVolumeSnapshot("foo", metav1.NamespaceDefault)
	err := storage.Storage.Create(ctx, key, pvcStart, nil, 0)

	vs := &storageapi.VolumeSnapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: storageapi.VolumeSnapshotSpec{
			PersistentVolumeClaimName: "bar",
		},
		Status: storageapi.VolumeSnapshotStatus{
			Conditions: []storageapi.VolumeSnapshotCondition{
				{
					Status:  api.ConditionTrue,
					Message: "Failed to create the snapshot date",
					Type:    storageapi.VolumeSnapshotConditionError,
				},
			},
		},
	}
	_, _, err = statusStorage.Update(ctx, vs.Name, rest.DefaultUpdatedObjectInfo(vs), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	vsOut := obj.(*storageapi.VolumeSnapshot)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(vs.Status, vsOut.Status) {
		t.Errorf("unexpected object: %s", diff.ObjectDiff(vs.Status, vsOut.Status))
	}
}

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"vs"}
	registrytest.AssertShortNames(t, storage, expected)
}
