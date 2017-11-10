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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, admissionregistration.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "initializerconfigurations",
	}
	return NewREST(restOptions), server
}

func validInitializerConfiguration() *admissionregistration.InitializerConfiguration {
	return &admissionregistration.InitializerConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Initializers: []admissionregistration.Initializer{{
			Name: "initializer.k8s.io",
			Rules: []admissionregistration.Rule{{
				APIGroups:   []string{"a"},
				APIVersions: []string{"a"},
				Resources:   []string{"a"},
			}},
		}},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	initializerConfiguration := validInitializerConfiguration()
	initializerConfiguration.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo-"}
	test.TestCreate(
		// valid
		initializerConfiguration,
		// invalid
		&admissionregistration.InitializerConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: "_-a123-a_"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		validInitializerConfiguration(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*admissionregistration.InitializerConfiguration)
			object.Initializers = []admissionregistration.Initializer{{
				Name: "initializer.k8s.io",
				Rules: []admissionregistration.Rule{{
					APIGroups:   []string{"a", "b"},
					APIVersions: []string{"a", "b"},
					Resources:   []string{"a", "b"},
				}},
			}}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*admissionregistration.InitializerConfiguration)
			object.Initializers = []admissionregistration.Initializer{{
				Name: "initializer.k8s.io",
				Rules: []admissionregistration.Rule{{
					APIGroups:   []string{"a"},
					APIVersions: []string{"a"},
					Resources:   []string{"a/b"},
				}},
			}}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestDelete(validInitializerConfiguration())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validInitializerConfiguration())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validInitializerConfiguration())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validInitializerConfiguration(),
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
			{"name": "foo"},
		},
	)
}
