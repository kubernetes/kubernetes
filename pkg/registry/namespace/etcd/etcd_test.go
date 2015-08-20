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
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/registry/namespace"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func newEtcdStorage(t *testing.T) (*tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, latest.Codec, etcdtest.PathPrefix())
	return fakeEtcdClient, etcdStorage
}

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient, s := newEtcdStorage(t)
	storage, _, _ := NewStorage(s)
	return storage, fakeEtcdClient, s
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
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _, _ := NewStorage(etcdStorage)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()
	namespace := validNewNamespace()
	namespace.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		namespace,
		// invalid
		&api.Namespace{
			ObjectMeta: api.ObjectMeta{Name: "bad value"},
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
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _, _ := NewStorage(etcdStorage)
	namespace := validNewNamespace()
	_, err := storage.Create(api.NewContext(), namespace)
	if err != fakeEtcdClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := &api.Namespace{}
	ctx := api.NewContext()
	key, err := storage.Etcd.KeyFunc(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected key error: %v", err)
	}
	if err := etcdStorage.Get(key, actual, false); err != nil {
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

func TestNamespaceDecode(t *testing.T) {
	_, etcdStorage := newEtcdStorage(t)
	storage, _, _ := NewStorage(etcdStorage)
	expected := validNewNamespace()
	expected.Status.Phase = api.NamespaceActive
	expected.Spec.Finalizers = []api.FinalizerName{api.FinalizerKubernetes}
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
	storage, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()
	namespace := validNewNamespace()
	test.TestGet(namespace)
}

func TestList(t *testing.T) {
	storage, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()
	key := etcdtest.AddPrefix(storage.KeyRootFunc(test.TestContext()))
	namespace := validNewNamespace()
	test.TestList(
		namespace,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeEtcdClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeEtcdClient, resourceVersion)
		})
}

func TestDeleteNamespace(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.ChangeIndex = 1
	storage, _, _ := NewStorage(etcdStorage)
	ctx := api.NewContext()
	key, err := storage.Etcd.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
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
	_, err = storage.Delete(api.NewContext(), "foo", nil)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDeleteNamespaceWithIncompleteFinalizers(t *testing.T) {
	now := util.Now()
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.ChangeIndex = 1
	key := etcdtest.AddPrefix("/namespaces/foo")
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
					ObjectMeta: api.ObjectMeta{
						Name:              "foo",
						DeletionTimestamp: &now,
					},
					Spec: api.NamespaceSpec{
						Finalizers: []api.FinalizerName{api.FinalizerKubernetes},
					},
					Status: api.NamespaceStatus{Phase: api.NamespaceActive},
				}),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	storage, _, _ := NewStorage(etcdStorage)
	_, err := storage.Delete(api.NewContext(), "foo", nil)
	if err == nil {
		t.Fatalf("expected error: %v", err)
	}
}

func TestDeleteNamespaceWithCompleteFinalizers(t *testing.T) {
	now := util.Now()
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.ChangeIndex = 1
	key := etcdtest.AddPrefix("/namespaces/foo")
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.Namespace{
					ObjectMeta: api.ObjectMeta{
						Name:              "foo",
						DeletionTimestamp: &now,
					},
					Spec: api.NamespaceSpec{
						Finalizers: []api.FinalizerName{},
					},
					Status: api.NamespaceStatus{Phase: api.NamespaceActive},
				}),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	storage, _, _ := NewStorage(etcdStorage)
	_, err := storage.Delete(api.NewContext(), "foo", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
