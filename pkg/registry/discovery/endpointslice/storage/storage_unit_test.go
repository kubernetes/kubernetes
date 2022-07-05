/*
Copyright 2022 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/discovery"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, discovery.Resource("endpointslices"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "endpointslices",
	}
	endpointsliceStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return endpointsliceStorage, server
}

var (
	namespace = metav1.NamespaceNone
	name      = "foo-endpointslice"
)

func validEndpointslice() *discovery.EndpointSlice {
	return &discovery.EndpointSlice{
		TypeMeta:    metav1.TypeMeta{},
		ObjectMeta:  metav1.ObjectMeta{Name: name, Namespace: namespace},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints:   []discovery.Endpoint{},
		Ports:       []discovery.EndpointPort{},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	endpointslice := validEndpointslice()
	invalidEndpointslice := validEndpointslice()
	invalidEndpointslice.Endpoints = append(invalidEndpointslice.Endpoints, discovery.Endpoint{})
	test.TestCreate(
		// valid
		endpointslice,
		invalidEndpointslice,
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		validEndpointslice(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*discovery.EndpointSlice)
			object.Endpoints = append(object.Endpoints, discovery.Endpoint{Addresses: []string{"10.3.245.137"}})
			return object
		},
		// invalid updateFunc: Empty Endpoint Address.
		func(obj runtime.Object) runtime.Object {
			object := obj.(*discovery.EndpointSlice)
			object.Endpoints = append(object.Endpoints, discovery.Endpoint{})
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestDelete(validEndpointslice())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(validEndpointslice())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(validEndpointslice())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validEndpointslice(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": name},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": name},
		},
	)
}
