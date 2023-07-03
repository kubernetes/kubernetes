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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "nodes",
	}
	storage, err := NewStorage(restOptions, kubeletclient.KubeletClientConfig{
		Port:                  10250,
		PreferredAddressTypes: []string{string(api.NodeInternalIP)},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return storage.Node, server
}

type tweak func(*api.Node)

func newNode(name string, tweaks ...tweak) *api.Node {
	node := &api.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": name,
			},
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("10"),
				api.ResourceMemory: resource.MustParse("0"),
			},
		},
	}

	for _, tweak := range tweaks {
		tweak(node)
	}

	return node
}

func setNodeIPAddress(addr string) tweak {
	return func(node *api.Node) {
		node.Status.Addresses = []api.NodeAddress{
			{Type: api.NodeInternalIP, Address: addr},
		}
	}
}

func setNodeDaemonEndpoint(port int32) tweak {
	return func(node *api.Node) {
		node.Status.DaemonEndpoints = api.NodeDaemonEndpoints{
			KubeletEndpoint: api.DaemonEndpoint{
				Port: port,
			},
		}
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestCreate(
		// valid
		newNode("foo"),
		// invalid
		newNode("_-a123-a_"),
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		newNode("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.Node)
			object.Spec.Unschedulable = !object.Spec.Unschedulable
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestDelete(newNode("foo"))
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(newNode("foo"))
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(newNode("foo"))
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		newNode("foo"),
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
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"no"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestResourceLocation(t *testing.T) {
	type testCase struct {
		name  string
		node  *api.Node
		query string
		host  string
		err   bool
	}

	testCases := []testCase{{
		name:  "proxyable hostname with default port",
		node:  newNode("node0", setNodeIPAddress("10.0.0.1")),
		query: "node0",
		host:  "10.0.0.1:10250",
	}, {
		name:  "proxyable hostname with kubelet port in query",
		node:  newNode("node0", setNodeIPAddress("10.0.0.1")),
		query: "node0:5000",
		host:  "10.0.0.1:5000",
	}, {
		name:  "proxyable hostname with kubelet port in status",
		node:  newNode("node0", setNodeIPAddress("10.0.0.1"), setNodeDaemonEndpoint(5000)),
		query: "node0",
		host:  "10.0.0.1:5000",
	}, {
		name:  "non-proxyable hostname with default port",
		node:  newNode("node0", setNodeIPAddress("127.0.0.1")),
		query: "node0",
		host:  "",
		err:   true,
	}, {
		name:  "non-proxyable hostname with kubelet port in query",
		node:  newNode("node0", setNodeIPAddress("127.0.0.1")),
		query: "node0:5000",
		host:  "",
		err:   true,
	}, {
		name:  "non-proxyable hostname with kubelet port in status",
		node:  newNode("node0", setNodeIPAddress("127.0.0.1"), setNodeDaemonEndpoint(443)),
		query: "node0",
		host:  "",
		err:   true,
	}}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			storage, server := newStorage(t)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.WithNamespace(genericapirequest.NewDefaultContext(), fmt.Sprintf("namespace-%s", testCase.name))
			key, _ := storage.KeyFunc(ctx, testCase.node.Name)
			if err := storage.Storage.Create(ctx, key, testCase.node, nil, 0, false); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			redirector := rest.Redirector(storage)
			location, _, err := redirector.ResourceLocation(ctx, testCase.query)

			if err != nil {
				if !testCase.err {
					t.Fatalf("Unexpected error: %v", err)
				}
				return
			} else if testCase.err {
				t.Fatalf("Expected error but got none")
			}

			if location == nil {
				t.Errorf("Unexpected nil resource location: %v", location)
			}

			if location.Host != testCase.host {
				t.Errorf("Unexpected host: expected %v, but got %v", testCase.host, location.Host)
			}
		})
	}
}
