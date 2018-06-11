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
	"fmt"
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
		ResourcePrefix:          "volumesnapshotdatas",
	}
	volumeSnapshotDataStorage, statusStorage := NewREST(restOptions)
	return volumeSnapshotDataStorage, statusStorage, server
}

func validNewVolumeSnapshotData(name string) *storageapi.VolumeSnapshotData {
	vsd := &storageapi.VolumeSnapshotData{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storageapi.VolumeSnapshotDataSpec{
			VolumeSnapshotDataSource: storageapi.VolumeSnapshotDataSource{
				HostPath: &storageapi.HostPathVolumeSnapshotSource{Path: "/foo"},
			},
		},
	}
	return vsd
}

func TestCreate(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	vsd := validNewVolumeSnapshotData("foo")
	vsd.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		vsd,
		// invalid
		&storageapi.VolumeSnapshotData{
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
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		validNewVolumeSnapshotData("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*storageapi.VolumeSnapshotData)
			object.Spec.VolumeSnapshotRef = &api.ObjectReference{
				Kind: "VolumeSnapshot",
				Name: "snapshotname",
			}
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
	test := genericregistrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewVolumeSnapshotData("foo"))
}

func TestGet(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewVolumeSnapshotData("foo"))
}

func TestList(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewVolumeSnapshotData("foo"))
}

func TestWatch(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewVolumeSnapshotData("foo"),
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
	ctx := genericapirequest.NewContext()
	key, _ := storage.KeyFunc(ctx, "foo")
	vsdStart := validNewVolumeSnapshotData("foo")
	err := storage.Storage.Create(ctx, key, vsdStart, nil, 0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	vsdIn := &storageapi.VolumeSnapshotData{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Status: storageapi.VolumeSnapshotDataStatus{
			Conditions: []storageapi.VolumeSnapshotDataCondition{
				{
					Status:  api.ConditionTrue,
					Message: fmt.Sprintf("Failed to create the snapshot date"),
					Type:    storageapi.VolumeSnapshotDataConditionError,
				},
			},
		},
	}

	_, _, err = statusStorage.Update(ctx, vsdIn.Name, rest.DefaultUpdatedObjectInfo(vsdIn), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	vsdOut := obj.(*storageapi.VolumeSnapshotData)
	// only compare the relevant change b/c metadata will differ
	if !apiequality.Semantic.DeepEqual(vsdIn.Status, vsdOut.Status) {
		t.Errorf("unexpected object: %s", diff.ObjectDiff(vsdIn.Status, vsdOut.Status))
	}
}

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"vsd"}
	registrytest.AssertShortNames(t, storage, expected)
}
