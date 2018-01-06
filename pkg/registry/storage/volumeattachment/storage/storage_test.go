/*
Copyright 2017 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/api/testapi"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, storageapi.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "volumeattachments",
	}
	volumeAttachmentStorage := NewREST(restOptions)
	return volumeAttachmentStorage, server
}

func validNewVolumeAttachment(name string) *storageapi.VolumeAttachment {
	pvName := "foo"
	return &storageapi.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storageapi.VolumeAttachmentSpec{
			Attacher: "valid-attacher",
			Source: storageapi.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
			NodeName: "valid-node",
		},
	}
}

func validChangedVolumeAttachment() *storageapi.VolumeAttachment {
	return validNewVolumeAttachment("foo")
}

func TestCreate(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions exception v1alpha1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	volumeAttachment := validNewVolumeAttachment("foo")
	volumeAttachment.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo"}
	pvName := "foo"
	test.TestCreate(
		// valid
		volumeAttachment,
		// invalid
		&storageapi.VolumeAttachment{
			ObjectMeta: metav1.ObjectMeta{Name: "*BadName!"},
			Spec: storageapi.VolumeAttachmentSpec{
				Attacher: "invalid-attacher-!@#$%^&*()",
				Source: storageapi.VolumeAttachmentSource{
					PersistentVolumeName: &pvName,
				},
				NodeName: "invalid-node-!@#$%^&*()",
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions except v1alpha1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		validNewVolumeAttachment("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*storageapi.VolumeAttachment)
			object.Status.Attached = true
			return object
		},
		//invalid update
		func(obj runtime.Object) runtime.Object {
			object := obj.(*storageapi.VolumeAttachment)
			object.Spec.Attacher = "invalid-attacher-!@#$%^&*()"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions except v1alpha1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewVolumeAttachment("foo"))
}

func TestGet(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions except v1alpha1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewVolumeAttachment("foo"))
}

func TestList(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions except v1alpha1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewVolumeAttachment("foo"))
}

func TestWatch(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1alpha1.SchemeGroupVersion {
		// skip the test for all versions except v1alpha1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewVolumeAttachment("foo"),
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
		},
	)
}
