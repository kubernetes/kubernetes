/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	apiclusters "k8s.io/kubernetes/pkg/apis/clusters"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, apiclusters.GroupName)
	restOptions := generic.RESTOptions{etcdStorage, generic.UndecoratedStorage}
	storage, _ := NewREST(restOptions)
	return storage, server
}

func validNewCluster() *apiclusters.Cluster {
	return &apiclusters.Cluster{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
		Spec: apiclusters.ClusterSpec{
			Address: apiclusters.ClusterAddress{
				Url: "http://localhost:8888",
			},
		},
		Status: apiclusters.ClusterStatus{
			Phase: apiclusters.ClusterPending,
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd).ClusterScope()
	cluster := validNewCluster()
	cluster.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		cluster,
		&apiclusters.Cluster{
			ObjectMeta: api.ObjectMeta{Name: "-a123-a_"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd).ClusterScope()
	test.TestUpdate(
		// valid
		validNewCluster(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apiclusters.Cluster)
			object.Spec.Credential = "bar"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewCluster())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd).ClusterScope()
	test.TestGet(validNewCluster())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd).ClusterScope()
	test.TestList(validNewCluster())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd).ClusterScope()
	test.TestWatch(
		validNewCluster(),
		// matching labels
		[]labels.Set{
			{"name": "foo"},
		},
		// not matching labels
		[]labels.Set{
			{"name": "bar"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matchin fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}
