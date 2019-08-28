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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, scheduling.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "priorityclasses",
	}
	rest, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unable to create REST %v", err)
	}
	return rest, server
}

func validNewPriorityClass() *scheduling.PriorityClass {
	return &scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Value:         100,
		GlobalDefault: false,
		Description:   "This is created only for testing.",
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestCreate(
		validNewPriorityClass(),
		// invalid cases
		&scheduling.PriorityClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: "*badName",
			},
			Value:         100,
			GlobalDefault: true,
			Description:   "This is created only for testing.",
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		validNewPriorityClass(),
		// There is no valid update function
		func(obj runtime.Object) runtime.Object {
			pc := obj.(*scheduling.PriorityClass)
			pc.Value = 100
			pc.GlobalDefault = false
			return pc
		},
		// invalid updates
		// Change Value
		func(obj runtime.Object) runtime.Object {
			pc := obj.(*scheduling.PriorityClass)
			pc.Value = 200
			pc.GlobalDefault = false
			return pc
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestDelete(validNewPriorityClass())
}

// TestDeleteSystemPriorityClass checks that system priority classes cannot be deleted.
func TestDeleteSystemPriorityClass(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	key := "test/system-node-critical"
	ctx := genericapirequest.NewContext()
	pc := scheduling.SystemPriorityClasses()[0]
	if err := storage.Store.Storage.Create(ctx, key, pc, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, _, err := storage.Delete(ctx, pc.Name, rest.ValidateAllObjectFunc, nil); err == nil {
		t.Error("expected to receive an error")
	}
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewPriorityClass())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewPriorityClass())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewPriorityClass(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
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
	expected := []string{"pc"}
	registrytest.AssertShortNames(t, storage, expected)
}
