/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, storageapi.SchemeGroupVersion.WithResource("csidrivers").GroupResource())
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "csidrivers",
	}
	csiDriverStorage, err := NewStorage(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return csiDriverStorage.CSIDriver, server
}

func validNewCSIDriver(name string) *storageapi.CSIDriver {
	attachRequired := true
	podInfoOnMount := true
	requiresRepublish := true
	storageCapacity := true
	seLinuxMount := true
	return &storageapi.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storageapi.CSIDriverSpec{
			AttachRequired:    &attachRequired,
			PodInfoOnMount:    &podInfoOnMount,
			RequiresRepublish: &requiresRepublish,
			StorageCapacity:   &storageCapacity,
			SELinuxMount:      &seLinuxMount,
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	csiDriver := validNewCSIDriver("foo")
	csiDriver.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo"}
	attachNotRequired := false
	notPodInfoOnMount := false
	notRequiresRepublish := false
	notStorageCapacity := false
	notSELinuxMount := false
	test.TestCreate(
		// valid
		csiDriver,
		// invalid
		&storageapi.CSIDriver{
			ObjectMeta: metav1.ObjectMeta{Name: "*BadName!"},
			Spec: storageapi.CSIDriverSpec{
				AttachRequired:    &attachNotRequired,
				PodInfoOnMount:    &notPodInfoOnMount,
				RequiresRepublish: &notRequiresRepublish,
				StorageCapacity:   &notStorageCapacity,
				SELinuxMount:      &notSELinuxMount,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	notPodInfoOnMount := false

	test.TestUpdate(
		// valid
		validNewCSIDriver("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*storageapi.CSIDriver)
			object.Labels = map[string]string{"a": "b"}
			return object
		},
		//invalid update
		func(obj runtime.Object) runtime.Object {
			object := obj.(*storageapi.CSIDriver)
			object.Spec.PodInfoOnMount = &notPodInfoOnMount
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewCSIDriver("foo"))
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewCSIDriver("foo"))
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewCSIDriver("foo"))
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewCSIDriver("foo"),
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
