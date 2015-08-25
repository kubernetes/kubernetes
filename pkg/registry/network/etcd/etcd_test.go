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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/registry/network"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t)
	storage, _ := NewREST(etcdStorage)
	return storage, fakeClient
}

func validNewNetwork() *api.Network {
	return &api.Network{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
		Spec: api.NetworkSpec{
			Subnets: []api.Subnet {
				api.Subnet{
					CIDR: "192.168.0.0/24",
					Gateway: "192.168.0.1",
				},
			},
		} ,
	}
}

func validChangedNetwork() *api.Network {
	Network := validNewNetwork()
	Network.ResourceVersion = "1"
	Network.Labels = map[string]string{
		"foo": "bar",
	}
	return Network
}

func TestStorage(t *testing.T) {
	storage, _ := newStorage(t)
	network.NewRegistry(storage)
}

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError).ClusterScope()
	Network := validNewNetwork()
	Network.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		Network,
		// invalid
		&api.Network{
			ObjectMeta: api.ObjectMeta{Name: "bad value"},
		},
	)
}

func expectNetwork(t *testing.T, out runtime.Object) (*api.Network, bool) {
	Network, ok := out.(*api.Network)
	if !ok || Network == nil {
		t.Errorf("Expected an api.Network object, was %#v", out)
		return nil, false
	}
	return Network, true
}

func TestCreateSetsFields(t *testing.T) {
	storage, fakeClient := newStorage(t)
	Network := validNewNetwork()
	ctx := api.NewContext()
	_, err := storage.Create(ctx, Network)
	if err != fakeClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}

	object, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actual := object.(*api.Network)
	if actual.Name != Network.Name {
		t.Errorf("unexpected Network: %#v", actual)
	}
	if len(actual.UID) == 0 {
		t.Errorf("expected Network UID to be set: %#v", actual)
	}
	if actual.Status.Phase != api.NetworkInitializing {
		t.Errorf("expected Network phase to be set to initializing, but %v", actual.Status.Phase)
	}
}

func TestNetworkDecode(t *testing.T) {
	storage, _ := newStorage(t)
	expected := validNewNetwork()
	expected.Status.Phase = api.NetworkInitializing
	body, err := testapi.Codec().Encode(expected)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := storage.New()
	if err := testapi.Codec().DecodeInto(body, actual); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("mismatch: %s", util.ObjectDiff(expected, actual))
	}
}

func TestGet(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError).ClusterScope()
	Network := validNewNetwork()
	test.TestGet(Network)
}

func TestList(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError).ClusterScope()
	key := etcdtest.AddPrefix(storage.KeyRootFunc(test.TestContext()))
	Network := validNewNetwork()
	test.TestList(
		Network,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeClient, resourceVersion)
		})
}

func TestDeleteNetwork(t *testing.T) {
	storage, fakeClient := newStorage(t)
	fakeClient.ChangeIndex = 1
	ctx := api.NewContext()
	key, err := storage.Etcd.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(testapi.Codec(), &api.Network{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
					},
					Status: api.NetworkStatus{Phase: api.NetworkInitializing},
				}),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	_, err = storage.Delete(api.NewContext(), "foo", nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
