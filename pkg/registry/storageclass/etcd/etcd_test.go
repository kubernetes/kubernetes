/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, extensions.GroupName)
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1}
	storageClassStorage := NewREST(restOptions)
	return storageClassStorage, server
}

func validNewStorageClass(name string) *extensions.StorageClass {
	return &extensions.StorageClass{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Provisioner: "kubernetes.io/aws-ebs",
		Parameters: map[string]string{
			"foo": "bar",
		},
	}
}

func validChangedStorageClass() *extensions.StorageClass {
	return validNewStorageClass("foo")
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	storageClass := validNewStorageClass("foo")
	storageClass.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		storageClass,
		// invalid
		&extensions.StorageClass{
			ObjectMeta: api.ObjectMeta{Name: "*BadName!"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		validNewStorageClass("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.StorageClass)
			object.Parameters = map[string]string{"foo": "bar"}
			return object
		},
		//invalid update
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.StorageClass)
			object.Parameters = map[string]string{"faz": "bar"}
			return object
		},
	)

}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewStorageClass("foo"))
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewStorageClass("foo"))
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewStorageClass("foo"))
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewStorageClass("foo"),
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
