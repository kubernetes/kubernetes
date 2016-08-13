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
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, apps.GroupName)
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "petsets"}
	petSetStorage, statusStorage := NewREST(restOptions)
	return petSetStorage, statusStorage, server
}

// createPetSet is a helper function that returns a PetSet with the updated resource version.
func createPetSet(storage *REST, ps apps.PetSet, t *testing.T) (apps.PetSet, error) {
	ctx := api.WithNamespace(api.NewContext(), ps.Namespace)
	obj, err := storage.Create(ctx, &ps)
	if err != nil {
		t.Errorf("Failed to create PetSet, %v", err)
	}
	newPS := obj.(*apps.PetSet)
	return *newPS, nil
}

func validNewPetSet() *apps.PetSet {
	return &apps.PetSet{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
			Labels:    map[string]string{"a": "b"},
		},
		Spec: apps.PetSetSpec{
			Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "test",
							Image:           "test_image",
							ImagePullPolicy: api.PullIfNotPresent,
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			Replicas: 7,
		},
		Status: apps.PetSetStatus{},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	ps := validNewPetSet()
	ps.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		ps,
		// TODO: Add an invalid case when we have validation.
	)
}

// TODO: Test updates to spec when we allow them.

func TestStatusUpdate(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)

	ctx := api.WithNamespace(api.NewContext(), api.NamespaceDefault)
	key := etcdtest.AddPrefix("/petsets/" + api.NamespaceDefault + "/foo")
	validPetSet := validNewPetSet()
	if err := storage.Storage.Create(ctx, key, validPetSet, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := apps.PetSet{
		ObjectMeta: validPetSet.ObjectMeta,
		Spec: apps.PetSetSpec{
			Replicas: 7,
		},
		Status: apps.PetSetStatus{
			Replicas: 7,
		},
	}

	if _, _, err := statusStorage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ps := obj.(*apps.PetSet)
	if ps.Spec.Replicas != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", ps.Spec.Replicas)
	}
	if ps.Status.Replicas != 7 {
		t.Errorf("we expected .status.replicas to be updated to %d but it was %v", 7, ps.Status.Replicas)
	}
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestGet(validNewPetSet())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestList(validNewPetSet())
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestDelete(validNewPetSet())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestWatch(
		validNewPetSet(),
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
