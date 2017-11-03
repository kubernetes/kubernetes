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
		ResourcePrefix:          "clusterrolebindings",
	}
	return NewREST(restOptions), server
}

func validNewClusterRoleBinding() *rbac.ClusterRoleBinding {
	return &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "master",
			Labels: map[string]string{"roleKey": "roleBind"},
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "ClusterRole",
			Name:     "valid",
		},
		Subjects: []rbac.Subject{
			{Name: "validsaname", APIGroup: "", Namespace: "foo", Kind: rbac.ServiceAccountKind},
			{Name: "valid@username", APIGroup: rbac.GroupName, Kind: rbac.UserKind},
			{Name: "valid@groupname", APIGroup: rbac.GroupName, Kind: rbac.GroupKind},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	invalid := validNewClusterRoleBinding()
	invalid.RoleRef.Name = "invalid"
	invalid.Labels = map[string]string{"a": "b"}
	invalid.Subjects = []rbac.Subject{{Name: "subject"}}
	test.TestCreate(
		validNewClusterRoleBinding(),
		// invalid cases
		invalid,
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope().AllowCreateOnUpdate()
	test.TestUpdate(
		// valid
		validNewClusterRoleBinding(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			clb := obj.(*rbac.ClusterRoleBinding)
			clb.Labels = map[string]string{"a": "b"}
			clb.Subjects = []rbac.Subject{{Namespace: "foo", Name: "subject", Kind: rbac.ServiceAccountKind}}
			return clb
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			clb := obj.(*rbac.ClusterRoleBinding)
			clb.Subjects = []rbac.Subject{{Namespace: "foo", Name: "subject:bad", Kind: rbac.ServiceAccountKind}}
			return clb
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestDelete(validNewClusterRoleBinding())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewClusterRoleBinding())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewClusterRoleBinding())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewClusterRoleBinding(),
		// matching labels
		[]labels.Set{
			{"roleKey": "roleBind"},
		},

		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "master"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}
