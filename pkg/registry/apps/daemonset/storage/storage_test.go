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
	"testing"

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, apps.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "daemonsets",
	}
	daemonSetStorage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return daemonSetStorage, statusStorage, server
}

func newValidDaemonSet() *apps.DaemonSet {
	return &apps.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
			Labels:    map[string]string{"a": "b"},
		},
		Spec: apps.DaemonSetSpec{
			UpdateStrategy: apps.DaemonSetUpdateStrategy{
				Type: apps.OnDeleteDaemonSetStrategyType,
			},
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: podtest.MakePodSpec(),
			},
		},
	}
}

var validDaemonSet = newValidDaemonSet()

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	ds := newValidDaemonSet()
	ds.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		ds,
		// invalid (invalid selector)
		&apps.DaemonSet{
			Spec: apps.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{}},
				Template: validDaemonSet.Spec.Template,
			},
		},
		// invalid update strategy
		&apps.DaemonSet{
			Spec: apps.DaemonSetSpec{
				Selector: validDaemonSet.Spec.Selector,
				Template: validDaemonSet.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		newValidDaemonSet(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.DaemonSet)
			object.Spec.Template.Spec.NodeSelector = map[string]string{"c": "d"}
			object.Spec.Template.Spec.DNSPolicy = api.DNSDefault
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.DaemonSet)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.DaemonSet)
			object.Spec.Template.Spec.RestartPolicy = api.RestartPolicyOnFailure
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestDelete(newValidDaemonSet())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(newValidDaemonSet())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(newValidDaemonSet())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validDaemonSet,
		// matching labels
		[]labels.Set{
			{"a": "b"},
		},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// notmatching fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": "foo"},
		},
	)
}

func TestUpdateStatus(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	dsStart := newValidDaemonSet()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewDefaultContext(), dsStart.Namespace)
	key, _ := storage.KeyFunc(ctx, dsStart.Name)
	err := storage.Storage.Create(ctx, key, dsStart, nil, 0, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	ds := dsStart.DeepCopy()
	ds.Status.ObservedGeneration = 1
	_, _, err = statusStorage.Update(ctx, ds.Name, rest.DefaultUpdatedObjectInfo(ds), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, ds.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	dsOut := obj.(*apps.DaemonSet)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(ds.Status, dsOut.Status) {
		t.Errorf("unexpected object: %s", cmp.Diff(ds.Status, dsOut.Status))
	}
}

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"ds"}
	registrytest.AssertShortNames(t, storage, expected)
}
