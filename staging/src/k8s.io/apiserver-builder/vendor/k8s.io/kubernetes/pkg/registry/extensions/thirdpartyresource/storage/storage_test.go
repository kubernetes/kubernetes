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
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	// Ensure that extensions/v1beta1 package is initialized.
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	_ "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, extensions.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "thirdpartyresources",
	}
	return NewREST(restOptions), server
}

func validNewThirdPartyResource(name string) *extensions.ThirdPartyResource {
	return &extensions.ThirdPartyResource{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Versions: []extensions.APIVersion{
			{
				Name: "v1",
			},
		},
	}
}

func namer(i int) string {
	return fmt.Sprintf("kind%d.example.com", i)
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope().Namer(namer).GeneratesName()
	rsrc := validNewThirdPartyResource("kind.domain.tld")
	test.TestCreate(
		// valid
		rsrc,
		// invalid
		&extensions.ThirdPartyResource{},
		&extensions.ThirdPartyResource{ObjectMeta: metav1.ObjectMeta{Name: "kind"}, Versions: []extensions.APIVersion{{Name: "v1"}}},
		&extensions.ThirdPartyResource{ObjectMeta: metav1.ObjectMeta{Name: "kind.tld"}, Versions: []extensions.APIVersion{{Name: "v1"}}},
		&extensions.ThirdPartyResource{ObjectMeta: metav1.ObjectMeta{Name: "kind.domain.tld"}, Versions: []extensions.APIVersion{{Name: "v.1"}}},
		&extensions.ThirdPartyResource{ObjectMeta: metav1.ObjectMeta{Name: "kind.domain.tld"}, Versions: []extensions.APIVersion{{Name: "stable/v1"}}},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope().Namer(namer)
	test.TestUpdate(
		// valid
		validNewThirdPartyResource("kind.domain.tld"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.ThirdPartyResource)
			object.Description = "new description"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope().Namer(namer)
	test.TestDelete(validNewThirdPartyResource("kind.domain.tld"))
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope().Namer(namer)
	test.TestGet(validNewThirdPartyResource("kind.domain.tld"))
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope().Namer(namer)
	test.TestList(validNewThirdPartyResource("kind.domain.tld"))
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope().Namer(namer)
	test.TestWatch(
		validNewThirdPartyResource("kind.domain.tld"),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": "foo"},
		},
	)
}
