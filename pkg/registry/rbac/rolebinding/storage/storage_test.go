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
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, rbac.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "rolebindings",
	}
	return NewREST(restOptions), server
}

func validRoleBinding() *rbac.RoleBinding {
	return &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		RoleRef: rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "valid"},
		Subjects: []rbac.Subject{
			{Name: "validsaname", APIGroup: "", Kind: rbac.ServiceAccountKind},
			{Name: "valid@username", APIGroup: rbac.GroupName, Kind: rbac.UserKind},
			{Name: "valid@groupname", APIGroup: rbac.GroupName, Kind: rbac.GroupKind},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	roleBinding := validRoleBinding()
	roleBinding.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		roleBinding,
		// invalid
		&rbac.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "/Invalid"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestUpdate(
		// valid
		validRoleBinding(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*rbac.RoleBinding)
			object.Subjects = []rbac.Subject{
				{Name: "validsaname", APIGroup: "", Kind: rbac.ServiceAccountKind},
			}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*rbac.RoleBinding)
			object.RoleRef = rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Type", Name: "invalid"}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestDelete(validRoleBinding())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestGet(validRoleBinding())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestList(validRoleBinding())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestWatch(
		validRoleBinding(),
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
