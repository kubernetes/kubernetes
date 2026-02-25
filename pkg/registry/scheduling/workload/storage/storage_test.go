/*
Copyright 2025 The Kubernetes Authors.

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
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, scheduling.Resource("workloads"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "workloads",
	}
	rest, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("Unable to create REST %v", err)
	}
	return rest, server
}

func validNewWorkload() *scheduling.Workload {
	return &scheduling.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: scheduling.WorkloadSpec{
			PodGroups: []scheduling.PodGroup{
				{
					Name: "bar",
					Policy: scheduling.PodGroupPolicy{
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 5,
						},
					},
				},
			},
		},
	}
}

func newTester(t *testing.T, storage *genericregistry.Store) *genericregistrytest.Tester {
	return genericregistrytest.New(t, storage).SetRequestInfo(&genericapirequest.RequestInfo{
		APIGroup:   "scheduling.k8s.io",
		APIVersion: "v1alpha1",
		Resource:   "workloads",
	})
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := newTester(t, storage.Store)
	test.TestCreate(
		validNewWorkload(),
		// invalid cases
		&scheduling.Workload{
			ObjectMeta: metav1.ObjectMeta{
				Name: "*badName",
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := newTester(t, storage.Store)
	test.TestUpdate(
		// valid
		validNewWorkload(),
		// valid update
		// Set ControllerRef
		func(obj runtime.Object) runtime.Object {
			w := obj.(*scheduling.Workload)
			w.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
				Kind: "foo",
				Name: "baz",
			}
			return w
		},
		// invalid update
		// Update MinCount
		func(obj runtime.Object) runtime.Object {
			w := obj.(*scheduling.Workload)
			w.Spec.PodGroups[0].Policy.Gang.MinCount = 4
			return w
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := newTester(t, storage.Store)
	test.TestDelete(validNewWorkload())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := newTester(t, storage.Store)
	test.TestGet(validNewWorkload())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := newTester(t, storage.Store)
	test.TestList(validNewWorkload())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := newTester(t, storage.Store)
	test.TestWatch(
		validNewWorkload(),
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
