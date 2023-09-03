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
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	_ "k8s.io/kubernetes/pkg/apis/networking/install"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, networking.Resource("clustercidrs"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "clustercidrs",
	}
	clusterCIDRStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return clusterCIDRStorage, server
}

var (
	namespace = metav1.NamespaceNone
	name      = "foo-clustercidr"
)

func newClusterCIDR() *networking.ClusterCIDR {
	return &networking.ClusterCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: networking.ClusterCIDRSpec{
			PerNodeHostBits: int32(8),
			IPv4:            "10.1.0.0/16",
			IPv6:            "fd00:1:1::/64",
			NodeSelector: &api.NodeSelector{
				NodeSelectorTerms: []api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{
							{
								Key:      "foo",
								Operator: api.NodeSelectorOpIn,
								Values:   []string{"bar"},
							},
						},
					},
				},
			},
		},
	}
}

func validClusterCIDR() *networking.ClusterCIDR {
	return newClusterCIDR()
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	validCC := validClusterCIDR()
	noCIDRCC := validClusterCIDR()
	noCIDRCC.Spec.IPv4 = ""
	noCIDRCC.Spec.IPv6 = ""
	invalidCCPerNodeHostBits := validClusterCIDR()
	invalidCCPerNodeHostBits.Spec.PerNodeHostBits = 100
	invalidCCCIDR := validClusterCIDR()
	invalidCCCIDR.Spec.IPv6 = "10.1.0.0/16"

	test.TestCreate(
		// valid
		validCC,
		//invalid
		noCIDRCC,
		invalidCCPerNodeHostBits,
		invalidCCCIDR,
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	test.TestUpdate(
		// valid
		validClusterCIDR(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.ClusterCIDR)
			object.Finalizers = []string{"test.k8s.io/test-finalizer"}
			return object
		},
		// invalid updateFunc: ObjectMeta is not to be tampered with.
		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.ClusterCIDR)
			object.Name = ""
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	test.TestDelete(validClusterCIDR())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	test.TestGet(validClusterCIDR())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	test.TestList(validClusterCIDR())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	test.TestWatch(
		validClusterCIDR(),
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

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"cc"}
	registrytest.AssertShortNames(t, storage, expected)
}
