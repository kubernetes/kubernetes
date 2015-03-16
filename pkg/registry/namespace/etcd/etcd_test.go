/*
Copyright 2015 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest/resttest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/namespace"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func newHelper(t *testing.T) (*tools.FakeEtcdClient, tools.EtcdHelper) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	helper := tools.EtcdHelper{Client: fakeEtcdClient, Codec: latest.Codec, ResourceVersioner: tools.RuntimeVersionAdapter{latest.ResourceVersioner}}
	return fakeEtcdClient, helper
}

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient, tools.EtcdHelper) {
	fakeEtcdClient, h := newHelper(t)
	storage := NewREST(h)
	return storage, fakeEtcdClient, h
}

func validNewNamespace() *api.Namespace {
	return &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	}
}

func validChangedNamespace() *api.Namespace {
	namespace := validNewNamespace()
	namespace.ResourceVersion = "1"
	namespace.Labels = map[string]string{
		"foo": "bar",
	}
	return namespace
}

func TestStorage(t *testing.T) {
	storage, _, _ := newStorage(t)
	namespace.NewRegistry(storage)
}

func TestCreate(t *testing.T) {
	fakeEtcdClient, helper := newHelper(t)
	storage := NewREST(helper)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	namespace := validNewNamespace()
	namespace.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		namespace,
		// invalid
		&api.Namespace{
			ObjectMeta: api.ObjectMeta{Namespace: "nope"},
		},
	)
}

func expectNamespace(t *testing.T, out runtime.Object) (*api.Namespace, bool) {
	namespace, ok := out.(*api.Namespace)
	if !ok || namespace == nil {
		t.Errorf("Expected an api.Namespace object, was %#v", out)
		return nil, false
	}
	return namespace, true
}

func TestCreateSetsFields(t *testing.T) {
	fakeEtcdClient, helper := newHelper(t)
	storage := NewREST(helper)
	namespace := validNewNamespace()
	_, err := storage.Create(api.NewDefaultContext(), namespace)
	if err != fakeEtcdClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := &api.Namespace{}
	if err := helper.ExtractObj("/registry/namespaces/foo", actual, false); err != nil {
		t.Fatalf("unexpected extraction error: %v", err)
	}
	if actual.Name != namespace.Name {
		t.Errorf("unexpected namespace: %#v", actual)
	}
	if len(actual.UID) == 0 {
		t.Errorf("expected namespace UID to be set: %#v", actual)
	}
	if actual.Status.Phase != api.NamespaceActive {
		t.Errorf("expected namespace phase to be set to active, but %v", actual.Status.Phase)
	}
}

func TestListEmptyNamespaceList(t *testing.T) {
	fakeEtcdClient, helper := newHelper(t)
	fakeEtcdClient.ChangeIndex = 1
	fakeEtcdClient.Data["/registry/namespaces"] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: fakeEtcdClient.NewError(tools.EtcdErrorCodeNotFound),
	}

	storage := NewREST(helper)
	namespaces, err := storage.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(namespaces.(*api.NamespaceList).Items) != 0 {
		t.Errorf("Unexpected non-zero namespace list: %#v", namespaces)
	}
	if namespaces.(*api.NamespaceList).ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", namespaces)
	}
}

func TestListNamespaceList(t *testing.T) {
	fakeEtcdClient, helper := newHelper(t)
	fakeEtcdClient.Data["/registry/namespaces"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
							ObjectMeta: api.ObjectMeta{Name: "bar"},
						}),
					},
				},
			},
		},
	}
	storage := NewREST(helper)
	namespacesObj, err := storage.List(api.NewDefaultContext(), labels.Everything(), fields.Everything())
	namespaces := namespacesObj.(*api.NamespaceList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(namespaces.Items) != 2 {
		t.Errorf("Unexpected namespaces list: %#v", namespaces)
	}
	if namespaces.Items[0].Name != "foo" || namespaces.Items[0].Status.Phase != api.NamespaceActive {
		t.Errorf("Unexpected namespace: %#v", namespaces.Items[0])
	}
	if namespaces.Items[1].Name != "bar" {
		t.Errorf("Unexpected namespace: %#v", namespaces.Items[1])
	}
}

func TestListNamespaceListSelection(t *testing.T) {
	fakeEtcdClient, helper := newHelper(t)
	fakeEtcdClient.Data["/registry/namespaces"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
						ObjectMeta: api.ObjectMeta{Name: "foo"},
					})},
					{Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
						ObjectMeta: api.ObjectMeta{Name: "bar"},
					})},
					{Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
						ObjectMeta: api.ObjectMeta{Name: "baz"},
						Status:     api.NamespaceStatus{Phase: api.NamespaceTerminating},
					})},
					{Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
						ObjectMeta: api.ObjectMeta{
							Name:   "qux",
							Labels: map[string]string{"label": "qux"},
						},
					})},
					{Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
						ObjectMeta: api.ObjectMeta{Name: "zot"},
					})},
				},
			},
		},
	}
	storage := NewREST(helper)
	ctx := api.NewDefaultContext()
	table := []struct {
		label, field string
		expectedIDs  util.StringSet
	}{
		{
			expectedIDs: util.NewStringSet("foo", "bar", "baz", "qux", "zot"),
		}, {
			field:       "name=zot",
			expectedIDs: util.NewStringSet("zot"),
		}, {
			label:       "label=qux",
			expectedIDs: util.NewStringSet("qux"),
		}, {
			field:       "status.phase=Terminating",
			expectedIDs: util.NewStringSet("baz"),
		},
	}

	for index, item := range table {
		label, err := labels.Parse(item.label)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		field, err := fields.ParseSelector(item.field)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		namespacesObj, err := storage.List(ctx, label, field)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		namespaces := namespacesObj.(*api.NamespaceList)

		set := util.NewStringSet()
		for i := range namespaces.Items {
			set.Insert(namespaces.Items[i].Name)
		}
		if e, a := len(item.expectedIDs), len(set); e != a {
			t.Errorf("%v: Expected %v, got %v", index, item.expectedIDs, set)
		}
	}
}

func TestNamespaceDecode(t *testing.T) {
	storage := NewREST(tools.EtcdHelper{})
	expected := validNewNamespace()
	expected.Status.Phase = api.NamespaceActive
	body, err := latest.Codec.Encode(expected)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := storage.New()
	if err := latest.Codec.DecodeInto(body, actual); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("mismatch: %s", util.ObjectDiff(expected, actual))
	}
}

func TestGet(t *testing.T) {
	expect := validNewNamespace()
	expect.Status.Phase = api.NamespaceActive
	fakeEtcdClient, helper := newHelper(t)
	fakeEtcdClient.Data["/registry/namespaces/foo"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, expect),
			},
		},
	}
	storage := NewREST(helper)
	obj, err := storage.Get(api.NewContext(), "foo")
	namespace := obj.(*api.Namespace)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expect.Status.Phase = api.NamespaceActive
	if e, a := expect, namespace; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("Unexpected namespace: %s", util.ObjectDiff(e, a))
	}
}

func TestDeleteNamespace(t *testing.T) {
	fakeEtcdClient, helper := newHelper(t)
	fakeEtcdClient.ChangeIndex = 1
	fakeEtcdClient.Data["/registry/namespaces/foo"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
					},
					Status: api.NamespaceStatus{Phase: api.NamespaceActive},
				}),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	storage := NewREST(helper)
	_, err := storage.Delete(api.NewDefaultContext(), "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// TODO: when we add life-cycle, this will go to Terminating, and then we need to test Terminating to gone
}
