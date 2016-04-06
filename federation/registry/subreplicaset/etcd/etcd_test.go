/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

const defaultReplicas = 100

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, federation.GroupName)
	restOptions := generic.RESTOptions{Storage: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1}
	storage, statusStorage := NewREST(restOptions)
	return storage, statusStorage, server
}

// createSubReplicaSet is a helper function that returns a SubReplicaSet with the updated resource version.
func createSubReplicaSet(storage *REST, rs federation.SubReplicaSet, t *testing.T) (federation.SubReplicaSet, error) {
	ctx := api.WithNamespace(api.NewContext(), rs.Namespace)
	obj, err := storage.Create(ctx, &rs)
	if err != nil {
		t.Errorf("Failed to create SubReplicaSet, %v", err)
	}
	newRS := obj.(*federation.SubReplicaSet)
	return *newRS, nil
}

func validNewSubReplicaSet() *federation.SubReplicaSet {
	return &federation.SubReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.ReplicaSetSpec{
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
		Status: extensions.ReplicaSetStatus{
			Replicas: 5,
		},
	}
}

var validSubReplicaSet = *validNewSubReplicaSet()

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	rs := validNewSubReplicaSet()
	rs.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		rs,
		// invalid (invalid selector)
		&federation.SubReplicaSet{
			Spec: extensions.ReplicaSetSpec{
				Replicas: 2,
				Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{}},
				Template: validSubReplicaSet.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestUpdate(
		// valid
		validNewSubReplicaSet(),
		// valid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*federation.SubReplicaSet)
			object.Spec.Replicas = object.Spec.Replicas + 1
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*federation.SubReplicaSet)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*federation.SubReplicaSet)
			object.Spec.Selector = &unversioned.LabelSelector{MatchLabels: map[string]string{}}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestDelete(validNewSubReplicaSet())
}

func TestGenerationNumber(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	modifiedSno := *validNewSubReplicaSet()
	modifiedSno.Generation = 100
	modifiedSno.Status.ObservedGeneration = 10
	ctx := api.NewDefaultContext()
	rs, err := createSubReplicaSet(storage, modifiedSno, t)
	etcdRS, err := storage.Get(ctx, rs.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ := etcdRS.(*federation.SubReplicaSet)

	// Generation initialization
	if storedRS.Generation != 1 && storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number %v, status generation %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}

	// Updates to spec should increment the generation number
	storedRS.Spec.Replicas += 1
	storage.Update(ctx, storedRS)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdRS, err = storage.Get(ctx, rs.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ = etcdRS.(*federation.SubReplicaSet)
	if storedRS.Generation != 2 || storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation, spec: %v, status: %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}

	// Updates to status should not increment either spec or status generation numbers
	storedRS.Status.Replicas += 1
	storage.Update(ctx, storedRS)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdRS, err = storage.Get(ctx, rs.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ = etcdRS.(*federation.SubReplicaSet)
	if storedRS.Generation != 2 || storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number, spec: %v, status: %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestGet(validNewSubReplicaSet())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestList(validNewSubReplicaSet())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestWatch(
		validNewSubReplicaSet(),
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
			{"status.replicas": "5"},
			{"metadata.name": "foo"},
			{"status.replicas": "5", "metadata.name": "foo"},
		},
		// not matchin fields
		[]fields.Set{
			{"status.replicas": "10"},
			{"metadata.name": "bar"},
			{"name": "foo"},
			{"status.replicas": "10", "metadata.name": "foo"},
			{"status.replicas": "0", "metadata.name": "bar"},
		},
	)
}

func TestStatusUpdate(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)

	ctx := api.WithNamespace(api.NewContext(), api.NamespaceDefault)
	key := etcdtest.AddPrefix("/subreplicasets/" + api.NamespaceDefault + "/foo")
	if err := storage.Storage.Create(ctx, key, &validSubReplicaSet, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := federation.SubReplicaSet{
		ObjectMeta: validSubReplicaSet.ObjectMeta,
		Spec: extensions.ReplicaSetSpec{
			Replicas: defaultReplicas,
		},
		Status: extensions.ReplicaSetStatus{
			Replicas: defaultReplicas,
		},
	}

	if _, _, err := statusStorage.Update(ctx, &update); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rs := obj.(*federation.SubReplicaSet)
	if rs.Spec.Replicas != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", rs.Spec.Replicas)
	}
	if rs.Status.Replicas != defaultReplicas {
		t.Errorf("we expected .status.replicas to be updated to %d but it was %v", defaultReplicas, rs.Status.Replicas)
	}
}
