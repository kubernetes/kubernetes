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
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, storageapi.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "volumeattachments",
	}
	volumeAttachmentStorage, err := NewStorage(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return volumeAttachmentStorage.VolumeAttachment, volumeAttachmentStorage.Status, server
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

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
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
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()

	test.TestUpdate(
		// valid
		validNewVolumeAttachment("foo"),
		// we still allow status field to be set in both v1 and v1beta1
		// it is just that in v1 the new value does not take effect.
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
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewVolumeAttachment("foo"))
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewVolumeAttachment("foo"))
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewVolumeAttachment("foo"))
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
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

func TestEtcdStatusUpdate(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericapirequest.NewDefaultContext()

	attachment := validNewVolumeAttachment("foo")
	if _, err := storage.Create(ctx, attachment, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, attachment.ObjectMeta.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// update status
	attachmentIn := obj.(*storageapi.VolumeAttachment).DeepCopy()
	attachmentIn.Status.Attached = true

	_, _, err = statusStorage.Update(ctx, attachmentIn.Name, rest.DefaultUpdatedObjectInfo(attachmentIn), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update status: %v", err)
	}

	// validate object got updated
	obj, err = storage.Get(ctx, attachmentIn.ObjectMeta.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	attachmentOut := obj.(*storageapi.VolumeAttachment)
	if !apiequality.Semantic.DeepEqual(attachmentIn.Spec, attachmentOut.Spec) {
		t.Errorf("objects differ: %v", diff.ObjectDiff(attachmentOut.Spec, attachmentIn.Spec))
	}
	if !apiequality.Semantic.DeepEqual(attachmentIn.Status, attachmentOut.Status) {
		t.Errorf("objects differ: %v", diff.ObjectDiff(attachmentOut.Status, attachmentIn.Status))
	}
}
