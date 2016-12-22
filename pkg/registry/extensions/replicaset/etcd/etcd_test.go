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
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/util/diff"
)

const defaultReplicas = 100

func newStorage(t *testing.T) (*ReplicaSetStorage, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "extensions")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "replicasets"}
	replicaSetStorage := NewStorage(restOptions)
	return &replicaSetStorage, server
}

// createReplicaSet is a helper function that returns a ReplicaSet with the updated resource version.
func createReplicaSet(storage *REST, rs extensions.ReplicaSet, t *testing.T) (extensions.ReplicaSet, error) {
	ctx := api.WithNamespace(api.NewContext(), rs.Namespace)
	obj, err := storage.Create(ctx, &rs)
	if err != nil {
		t.Errorf("Failed to create ReplicaSet, %v", err)
	}
	newRS := obj.(*extensions.ReplicaSet)
	return *newRS, nil
}

func validNewReplicaSet() *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
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

var validReplicaSet = *validNewReplicaSet()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.ReplicaSet.Store)
	rs := validNewReplicaSet()
	rs.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		rs,
		// invalid (invalid selector)
		&extensions.ReplicaSet{
			Spec: extensions.ReplicaSetSpec{
				Replicas: 2,
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{}},
				Template: validReplicaSet.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.ReplicaSet.Store)
	test.TestUpdate(
		// valid
		validNewReplicaSet(),
		// valid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.ReplicaSet)
			object.Spec.Replicas = object.Spec.Replicas + 1
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.ReplicaSet)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.ReplicaSet)
			object.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{}}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.ReplicaSet.Store)
	test.TestDelete(validNewReplicaSet())
}

func TestGenerationNumber(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	modifiedSno := *validNewReplicaSet()
	modifiedSno.Generation = 100
	modifiedSno.Status.ObservedGeneration = 10
	ctx := api.NewDefaultContext()
	rs, err := createReplicaSet(storage.ReplicaSet, modifiedSno, t)
	etcdRS, err := storage.ReplicaSet.Get(ctx, rs.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ := etcdRS.(*extensions.ReplicaSet)

	// Generation initialization
	if storedRS.Generation != 1 && storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number %v, status generation %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}

	// Updates to spec should increment the generation number
	storedRS.Spec.Replicas += 1
	storage.ReplicaSet.Update(ctx, storedRS.Name, rest.DefaultUpdatedObjectInfo(storedRS, api.Scheme))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdRS, err = storage.ReplicaSet.Get(ctx, rs.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ = etcdRS.(*extensions.ReplicaSet)
	if storedRS.Generation != 2 || storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation, spec: %v, status: %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}

	// Updates to status should not increment either spec or status generation numbers
	storedRS.Status.Replicas += 1
	storage.ReplicaSet.Update(ctx, storedRS.Name, rest.DefaultUpdatedObjectInfo(storedRS, api.Scheme))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdRS, err = storage.ReplicaSet.Get(ctx, rs.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ = etcdRS.(*extensions.ReplicaSet)
	if storedRS.Generation != 2 || storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number, spec: %v, status: %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.ReplicaSet.Store)
	test.TestGet(validNewReplicaSet())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.ReplicaSet.Store)
	test.TestList(validNewReplicaSet())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := registrytest.New(t, storage.ReplicaSet.Store)
	test.TestWatch(
		validNewReplicaSet(),
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

func TestScaleGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()

	name := "foo"

	var rs extensions.ReplicaSet
	ctx := api.WithNamespace(api.NewContext(), api.NamespaceDefault)
	key := "/replicasets/" + api.NamespaceDefault + "/" + name
	if err := storage.ReplicaSet.Storage.Create(ctx, key, &validReplicaSet, &rs, 0); err != nil {
		t.Fatalf("error setting new replica set (key: %s) %v: %v", key, validReplicaSet, err)
	}

	want := &extensions.Scale{
		ObjectMeta: api.ObjectMeta{
			Name:              name,
			Namespace:         api.NamespaceDefault,
			UID:               rs.UID,
			ResourceVersion:   rs.ResourceVersion,
			CreationTimestamp: rs.CreationTimestamp,
		},
		Spec: extensions.ScaleSpec{
			Replicas: validReplicaSet.Spec.Replicas,
		},
		Status: extensions.ScaleStatus{
			Replicas: validReplicaSet.Status.Replicas,
			Selector: validReplicaSet.Spec.Selector,
		},
	}
	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	got := obj.(*extensions.Scale)
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	if !api.Semantic.DeepEqual(got, want) {
		t.Errorf("unexpected scale: %s", diff.ObjectDiff(got, want))
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()

	name := "foo"

	var rs extensions.ReplicaSet
	ctx := api.WithNamespace(api.NewContext(), api.NamespaceDefault)
	key := "/replicasets/" + api.NamespaceDefault + "/" + name
	if err := storage.ReplicaSet.Storage.Create(ctx, key, &validReplicaSet, &rs, 0); err != nil {
		t.Fatalf("error setting new replica set (key: %s) %v: %v", key, validReplicaSet, err)
	}
	replicas := 12
	update := extensions.Scale{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.ScaleSpec{
			Replicas: int32(replicas),
		},
	}

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil {
		t.Fatalf("error updating scale %v: %v", update, err)
	}

	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale := obj.(*extensions.Scale)
	if scale.Spec.Replicas != int32(replicas) {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, scale.Spec.Replicas)
	}

	update.ResourceVersion = rs.ResourceVersion
	update.Spec.Replicas = 15

	if _, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil && !errors.IsConflict(err) {
		t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
	}
}

func TestStatusUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()

	ctx := api.WithNamespace(api.NewContext(), api.NamespaceDefault)
	key := "/replicasets/" + api.NamespaceDefault + "/foo"
	if err := storage.ReplicaSet.Storage.Create(ctx, key, &validReplicaSet, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := extensions.ReplicaSet{
		ObjectMeta: validReplicaSet.ObjectMeta,
		Spec: extensions.ReplicaSetSpec{
			Replicas: defaultReplicas,
		},
		Status: extensions.ReplicaSetStatus{
			Replicas: defaultReplicas,
		},
	}

	if _, _, err := storage.Status.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.ReplicaSet.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rs := obj.(*extensions.ReplicaSet)
	if rs.Spec.Replicas != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", rs.Spec.Replicas)
	}
	if rs.Status.Replicas != defaultReplicas {
		t.Errorf("we expected .status.replicas to be updated to %d but it was %v", defaultReplicas, rs.Status.Replicas)
	}
}
