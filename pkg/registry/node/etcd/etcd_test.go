/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"net/http"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
)

type fakeConnectionInfoGetter struct {
}

func (fakeConnectionInfoGetter) GetConnectionInfo(host string) (string, uint, http.RoundTripper, error) {
	return "http", 12345, nil, nil
}

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t, "")
	storage, _ := NewREST(etcdStorage, false, fakeConnectionInfoGetter{}, nil)
	return storage, fakeClient
}

func validNewNode() *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
		Spec: api.NodeSpec{
			ExternalID: "external",
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
				api.ResourceName(api.ResourceMemory): resource.MustParse("0"),
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	node := validNewNode()
	node.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		node,
		// invalid
		&api.Node{
			ObjectMeta: api.ObjectMeta{Name: "_-a123-a_"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestUpdate(
		// valid
		validNewNode(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.Node)
			object.Spec.Unschedulable = !object.Spec.Unschedulable
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestDelete(validNewNode())
}

func TestGet(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestGet(validNewNode())
}

func TestList(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestList(validNewNode())
}

func TestWatch(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestWatch(
		validNewNode(),
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
