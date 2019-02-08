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

	storageapiv1beta1 "k8s.io/api/storage/v1beta1"
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
		ResourcePrefix:          "csinodes",
	}
	csiNodeStorage := NewStorage(restOptions)
	return csiNodeStorage.CSINode, server
}

func validNewCSINode(name string) *storageapi.CSINode {
	return &storageapi.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storageapi.CSINodeSpec{
			Drivers: []storageapi.CSINodeDriver{
				{
					Name:         "valid-driver-name",
					NodeID:       "valid-node",
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1beta1.SchemeGroupVersion {
		// skip the test for all versions exception v1beta1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	csiNode := validNewCSINode("foo")
	csiNode.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		csiNode,
		// invalid
		&storageapi.CSINode{
			ObjectMeta: metav1.ObjectMeta{Name: "*BadName!"},
			Spec: storageapi.CSINodeSpec{
				Drivers: []storageapi.CSINodeDriver{
					{
						Name:         "invalid-name-!@#$%^&*()",
						NodeID:       "invalid-node-!@#$%^&*()",
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1beta1.SchemeGroupVersion {
		// skip the test for all versions exception v1beta1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()

	test.TestUpdate(
		// valid
		validNewCSINode("foo"),
		// we allow status field to be set in v1beta1
		func(obj runtime.Object) runtime.Object {
			object := obj.(*storageapi.CSINode)
			//object.Status = *getCSINodeStatus()
			return object
		},
		//invalid update
		func(obj runtime.Object) runtime.Object {
			object := obj.(*storageapi.CSINode)
			object.Spec.Drivers[0].Name = "invalid-name-!@#$%^&*()"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1beta1.SchemeGroupVersion {
		// skip the test for all versions exception v1beta1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewCSINode("foo"))
}

func TestGet(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1beta1.SchemeGroupVersion {
		// skip the test for all versions exception v1beta1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewCSINode("foo"))
}

func TestList(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1beta1.SchemeGroupVersion {
		// skip the test for all versions exception v1beta1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewCSINode("foo"))
}

func TestWatch(t *testing.T) {
	if *testapi.Storage.GroupVersion() != storageapiv1beta1.SchemeGroupVersion {
		// skip the test for all versions exception v1beta1
		return
	}

	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewCSINode("foo"),
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
