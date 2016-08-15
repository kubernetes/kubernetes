/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, policy.GroupName)
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "poddisruptionbudgets"}
	podDisruptionBudgetStorage, statusStorage := NewREST(restOptions)
	return podDisruptionBudgetStorage, statusStorage, server
}

// createPodDisruptionBudget is a helper function that returns a PodDisruptionBudget with the updated resource version.
func createPodDisruptionBudget(storage *REST, pdb policy.PodDisruptionBudget, t *testing.T) (policy.PodDisruptionBudget, error) {
	ctx := api.WithNamespace(api.NewContext(), pdb.Namespace)
	obj, err := storage.Create(ctx, &pdb)
	if err != nil {
		t.Errorf("Failed to create PodDisruptionBudget, %v", err)
	}
	newPS := obj.(*policy.PodDisruptionBudget)
	return *newPS, nil
}

func validNewPodDisruptionBudget() *policy.PodDisruptionBudget {
	return &policy.PodDisruptionBudget{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
			Labels:    map[string]string{"a": "b"},
		},
		Spec: policy.PodDisruptionBudgetSpec{
			Selector:     &unversioned.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			MinAvailable: intstr.FromInt(7),
		},
		Status: policy.PodDisruptionBudgetStatus{},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	pdb := validNewPodDisruptionBudget()
	pdb.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		pdb,
		// TODO: Add an invalid case when we have validation.
	)
}

// TODO: Test updates to spec when we allow them.

func TestStatusUpdate(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)

	ctx := api.WithNamespace(api.NewContext(), api.NamespaceDefault)
	key := etcdtest.AddPrefix("/poddisruptionbudgets/" + api.NamespaceDefault + "/foo")
	validPodDisruptionBudget := validNewPodDisruptionBudget()
	if err := storage.Storage.Create(ctx, key, validPodDisruptionBudget, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := policy.PodDisruptionBudget{
		ObjectMeta: validPodDisruptionBudget.ObjectMeta,
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: intstr.FromInt(8),
		},
		Status: policy.PodDisruptionBudgetStatus{
			ExpectedPods: 8,
		},
	}

	if _, _, err := statusStorage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pdb := obj.(*policy.PodDisruptionBudget)
	if pdb.Spec.MinAvailable.IntValue() != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", pdb.Spec.MinAvailable)
	}
	if pdb.Status.ExpectedPods != 8 {
		t.Errorf("we expected .status.replicas to be updated to %d but it was %v", 7, pdb.Status.ExpectedPods)
	}
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestGet(validNewPodDisruptionBudget())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestList(validNewPodDisruptionBudget())
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestDelete(validNewPodDisruptionBudget())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestWatch(
		validNewPodDisruptionBudget(),
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
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

// TODO: Test generation number.
