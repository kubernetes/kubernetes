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
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	"k8s.io/kubernetes/pkg/apis/networking"
	_ "k8s.io/kubernetes/pkg/apis/networking/install"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, networking.Resource("clustercidrconfigs"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "clustercidrconfigs",
	}
	clusterCIDRConfigStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return clusterCIDRConfigStorage, server
}

var (
	namespace = metav1.NamespaceNone
	name      = "foo-clustercidrconfig"
)

func newClusterCIDRConfig() *networking.ClusterCIDRConfig {
	return &networking.ClusterCIDRConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: networking.ClusterCIDRConfigSpec{
			PerNodeHostBits: int32(8),
			IPv4CIDR:        "10.1.0.0/16",
			IPv6CIDR:        "fd00:1:1::/64",
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

func validClusterCIDRConfig() *networking.ClusterCIDRConfig {
	return newClusterCIDRConfig()
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	validCCC := validClusterCIDRConfig()
	noCIDRCCC := validClusterCIDRConfig()
	noCIDRCCC.Spec.IPv4CIDR = ""
	noCIDRCCC.Spec.IPv6CIDR = ""
	invalidCCCPerNodeHostBits := validClusterCIDRConfig()
	invalidCCCPerNodeHostBits.Spec.PerNodeHostBits = 100
	invalidCCCCIDR := validClusterCIDRConfig()
	invalidCCCCIDR.Spec.IPv6CIDR = "10.1.0.0/16"

	test.TestCreate(
		// valid
		validCCC,
		//invalid
		noCIDRCCC,
		invalidCCCPerNodeHostBits,
		invalidCCCCIDR,
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
		validClusterCIDRConfig(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.ClusterCIDRConfig)
			object.Finalizers = []string{"test.k8s.io/test-finalizer"}
			return object
		},
		// invalid updateFunc: ObjectMeta is not to be tampered with.
		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.ClusterCIDRConfig)
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
	test.TestDelete(validClusterCIDRConfig())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	test.TestGet(validClusterCIDRConfig())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	test.TestList(validClusterCIDRConfig())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()
	test.TestWatch(
		validClusterCIDRConfig(),
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
	expected := []string{"ccc"}
	registrytest.AssertShortNames(t, storage, expected)
}
