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
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/v1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"

	"github.com/coreos/go-etcd/etcd"
)

func init() {
	// Ensure that expapi/v1 packege is used, so that it will get initialized and register HorizontalPodAutoscaler object.
	_ = v1.ThirdPartyResource{}
}

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t)
	return NewREST(etcdStorage), fakeClient
}

func validNewThirdPartyResource(name string) *expapi.ThirdPartyResource {
	return &expapi.ThirdPartyResource{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Versions: []expapi.APIVersion{
			{
				Name: "stable/v1",
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	rsrc := validNewThirdPartyResource("foo")
	rsrc.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		rsrc,
		func(ctx api.Context, obj runtime.Object) error {
			return registrytest.SetObject(fakeClient, storage.KeyFunc, ctx, obj)
		},
		func(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
			return registrytest.GetObject(fakeClient, storage.KeyFunc, storage.NewFunc, ctx, obj)
		},
		// invalid
		&expapi.ThirdPartyResource{},
	)
}

func TestUpdate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	key, err := storage.KeyFunc(test.TestContext(), "foo")
	if err != nil {
		t.Fatal(err)
	}
	key = etcdtest.AddPrefix(key)
	fakeClient.ExpectNotFoundGet(key)
	fakeClient.ChangeIndex = 2
	rsrc := validNewThirdPartyResource("foo")
	existing := validNewThirdPartyResource("exists")
	existing.Namespace = test.TestNamespace()
	obj, err := storage.Create(test.TestContext(), existing)
	if err != nil {
		t.Fatalf("unable to create object: %v", err)
	}
	older := obj.(*expapi.ThirdPartyResource)
	older.ResourceVersion = "1"
	test.TestUpdate(
		rsrc,
		existing,
		older,
	)
}

func TestDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	rsrc := validNewThirdPartyResource("foo2")
	key, _ := storage.KeyFunc(ctx, "foo2")
	key = etcdtest.AddPrefix(key)
	createFn := func() runtime.Object {
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(testapi.Codec(), rsrc),
					ModifiedIndex: 1,
				},
			},
		}
		return rsrc
	}
	gracefulSetFn := func() bool {
		if fakeClient.Data[key].R.Node == nil {
			return false
		}
		return fakeClient.Data[key].R.Node.TTL == 30
	}
	test.TestDeleteNoGraceful(createFn, gracefulSetFn)
}

func TestGet(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	rsrc := validNewThirdPartyResource("foo")
	test.TestGet(rsrc)
}

func TestEmptyList(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, fakeClient := newStorage(t)
	fakeClient.ChangeIndex = 1
	key := registry.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: fakeClient.NewError(tools.EtcdErrorCodeNotFound),
	}
	rsrcList, err := registry.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rsrcList.(*expapi.ThirdPartyResourceList).Items) != 0 {
		t.Errorf("Unexpected non-zero autoscaler list: %#v", rsrcList)
	}
	if rsrcList.(*expapi.ThirdPartyResourceList).ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", rsrcList)
	}
}

func TestList(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, fakeClient := newStorage(t)
	fakeClient.ChangeIndex = 1
	key := registry.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(testapi.Codec(), &expapi.ThirdPartyResource{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(testapi.Codec(), &expapi.ThirdPartyResource{
							ObjectMeta: api.ObjectMeta{Name: "bar"},
						}),
					},
				},
			},
		},
	}
	obj, err := registry.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	rsrcList := obj.(*expapi.ThirdPartyResourceList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(rsrcList.Items) != 2 {
		t.Errorf("Unexpected ThirdPartyResource list: %#v", rsrcList)
	}
	if rsrcList.Items[0].Name != "foo" {
		t.Errorf("Unexpected ThirdPartyResource: %#v", rsrcList.Items[0])
	}
	if rsrcList.Items[1].Name != "bar" {
		t.Errorf("Unexpected ThirdPartyResource: %#v", rsrcList.Items[1])
	}
}
