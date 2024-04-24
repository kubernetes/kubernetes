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

package storage

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

const defaultReplicas = 100

func newStorage(t *testing.T) (*ReplicaSetStorage, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "extensions")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "replicasets"}
	replicaSetStorage, err := NewStorage(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return &replicaSetStorage, server
}

// createReplicaSet is a helper function that returns a ReplicaSet with the updated resource version.
func createReplicaSet(storage *REST, rs apps.ReplicaSet, t *testing.T) (apps.ReplicaSet, error) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), rs.Namespace)
	obj, err := storage.Create(ctx, &rs, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Failed to create ReplicaSet, %v", err)
	}
	newRS := obj.(*apps.ReplicaSet)
	return *newRS, nil
}

func validNewReplicaSet() *apps.ReplicaSet {
	return &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: apps.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: podtest.MakePod("").Spec,
			},
			Replicas: 7,
		},
		Status: apps.ReplicaSetStatus{
			Replicas: 5,
		},
	}
}

var validReplicaSet = *validNewReplicaSet()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.ReplicaSet.Store)
	rs := validNewReplicaSet()
	rs.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		rs,
		// invalid (invalid selector)
		&apps.ReplicaSet{
			Spec: apps.ReplicaSetSpec{
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
	test := genericregistrytest.New(t, storage.ReplicaSet.Store)
	test.TestUpdate(
		// valid
		validNewReplicaSet(),
		// valid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.ReplicaSet)
			object.Spec.Replicas = object.Spec.Replicas + 1
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.ReplicaSet)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*apps.ReplicaSet)
			object.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{}}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.ReplicaSet.Store)
	test.TestDelete(validNewReplicaSet())
}

func TestGenerationNumber(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	modifiedSno := *validNewReplicaSet()
	modifiedSno.Generation = 100
	modifiedSno.Status.ObservedGeneration = 10
	ctx := genericapirequest.NewDefaultContext()
	rs, err := createReplicaSet(storage.ReplicaSet, modifiedSno, t)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdRS, err := storage.ReplicaSet.Get(ctx, rs.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ := etcdRS.(*apps.ReplicaSet)

	// Generation initialization
	if storedRS.Generation != 1 || storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number %v, status generation %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}

	// Updates to spec should increment the generation number
	storedRS.Spec.Replicas++
	if _, _, err := storage.ReplicaSet.Update(ctx, storedRS.Name, rest.DefaultUpdatedObjectInfo(storedRS), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdRS, err = storage.ReplicaSet.Get(ctx, rs.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ = etcdRS.(*apps.ReplicaSet)
	if storedRS.Generation != 2 || storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation, spec: %v, status: %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}

	// Updates to status should not increment either spec or status generation numbers
	storedRS.Status.Replicas++
	if _, _, err := storage.ReplicaSet.Update(ctx, storedRS.Name, rest.DefaultUpdatedObjectInfo(storedRS), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdRS, err = storage.ReplicaSet.Get(ctx, rs.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedRS, _ = etcdRS.(*apps.ReplicaSet)
	if storedRS.Generation != 2 || storedRS.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number, spec: %v, status: %v", storedRS.Generation, storedRS.Status.ObservedGeneration)
	}
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.ReplicaSet.Store)
	test.TestGet(validNewReplicaSet())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.ReplicaSet.Store)
	test.TestList(validNewReplicaSet())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.ReplicaSet.Store)
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
		// not matching fields
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

	var rs apps.ReplicaSet
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/replicasets/" + metav1.NamespaceDefault + "/" + name
	if err := storage.ReplicaSet.Storage.Create(ctx, key, &validReplicaSet, &rs, 0, false); err != nil {
		t.Fatalf("error setting new replica set (key: %s) %v: %v", key, validReplicaSet, err)
	}

	selector, err := metav1.LabelSelectorAsSelector(validReplicaSet.Spec.Selector)
	if err != nil {
		t.Fatal(err)
	}

	want := &autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         metav1.NamespaceDefault,
			UID:               rs.UID,
			ResourceVersion:   rs.ResourceVersion,
			CreationTimestamp: rs.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: validReplicaSet.Spec.Replicas,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: validReplicaSet.Status.Replicas,
			Selector: selector.String(),
		},
	}
	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	got := obj.(*autoscaling.Scale)
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	if !apiequality.Semantic.DeepEqual(got, want) {
		t.Errorf("unexpected scale: %s", cmp.Diff(got, want))
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()

	name := "foo"

	var rs apps.ReplicaSet
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/replicasets/" + metav1.NamespaceDefault + "/" + name
	if err := storage.ReplicaSet.Storage.Create(ctx, key, &validReplicaSet, &rs, 0, false); err != nil {
		t.Fatalf("error setting new replica set (key: %s) %v: %v", key, validReplicaSet, err)
	}
	replicas := 12
	update := autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: int32(replicas),
		},
	}

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("error updating scale %v: %v", update, err)
	}

	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale := obj.(*autoscaling.Scale)
	if scale.Spec.Replicas != int32(replicas) {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, scale.Spec.Replicas)
	}

	update.ResourceVersion = rs.ResourceVersion
	update.Spec.Replicas = 15

	if _, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil && !apierrors.IsConflict(err) {
		t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
	}
}

func TestStatusUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/replicasets/" + metav1.NamespaceDefault + "/foo"
	if err := storage.ReplicaSet.Storage.Create(ctx, key, &validReplicaSet, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := apps.ReplicaSet{
		ObjectMeta: validReplicaSet.ObjectMeta,
		Spec: apps.ReplicaSetSpec{
			Replicas: defaultReplicas,
		},
		Status: apps.ReplicaSetStatus{
			Replicas: defaultReplicas,
		},
	}

	if _, _, err := storage.Status.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.ReplicaSet.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rs := obj.(*apps.ReplicaSet)
	if rs.Spec.Replicas != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", rs.Spec.Replicas)
	}
	if rs.Status.Replicas != defaultReplicas {
		t.Errorf("we expected .status.replicas to be updated to %d but it was %v", defaultReplicas, rs.Status.Replicas)
	}
}

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.DestroyFunc()
	expected := []string{"rs"}
	registrytest.AssertShortNames(t, storage.ReplicaSet, expected)
}

func TestCategories(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.ReplicaSet.Store.DestroyFunc()
	expected := []string{"all"}
	registrytest.AssertCategories(t, storage.ReplicaSet, expected)
}

func TestScalePatchErrors(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	validObj := &validReplicaSet
	namespace := validObj.Namespace
	name := validObj.Name
	resourceStore := storage.ReplicaSet.Store
	scaleStore := storage.Scale

	defer resourceStore.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)

	{
		applyNotFoundPatch := func() rest.TransformFunc {
			return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
				t.Errorf("notfound patch called")
				return currentObject, nil
			}
		}
		_, _, err := scaleStore.Update(ctx, "bad-name", rest.DefaultUpdatedObjectInfo(nil, applyNotFoundPatch()), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if !apierrors.IsNotFound(err) {
			t.Errorf("expected notfound, got %v", err)
		}
	}

	if _, err := resourceStore.Create(ctx, validObj, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	{
		applyBadUIDPatch := func() rest.TransformFunc {
			return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
				currentObject.(*autoscaling.Scale).UID = "123"
				return currentObject, nil
			}
		}
		_, _, err := scaleStore.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyBadUIDPatch()), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if !apierrors.IsConflict(err) {
			t.Errorf("expected conflict, got %v", err)
		}
	}

	{
		applyBadResourceVersionPatch := func() rest.TransformFunc {
			return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
				currentObject.(*autoscaling.Scale).ResourceVersion = "123"
				return currentObject, nil
			}
		}
		_, _, err := scaleStore.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyBadResourceVersionPatch()), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if !apierrors.IsConflict(err) {
			t.Errorf("expected conflict, got %v", err)
		}
	}
}

func TestScalePatchConflicts(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	validObj := &validReplicaSet
	namespace := validObj.Namespace
	name := validObj.Name
	resourceStore := storage.ReplicaSet.Store
	scaleStore := storage.Scale

	defer resourceStore.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	if _, err := resourceStore.Create(ctx, validObj, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	applyLabelPatch := func(labelName, labelValue string) rest.TransformFunc {
		return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
			currentObject.(metav1.Object).SetLabels(map[string]string{labelName: labelValue})
			return currentObject, nil
		}
	}
	stopCh := make(chan struct{})
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		// continuously submits a patch that updates a label and verifies the label update was effective
		labelName := "timestamp"
		for i := 0; ; i++ {
			select {
			case <-stopCh:
				return
			default:
				expectedLabelValue := fmt.Sprint(i)
				updated, _, err := resourceStore.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyLabelPatch(labelName, fmt.Sprint(i))), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
				if err != nil {
					t.Errorf("error patching main resource: %v", err)
					return
				}
				gotLabelValue := updated.(metav1.Object).GetLabels()[labelName]
				if gotLabelValue != expectedLabelValue {
					t.Errorf("wrong label value: expected: %s, got: %s", expectedLabelValue, gotLabelValue)
					return
				}
			}
		}
	}()

	// continuously submits a scale patch of replicas for a monotonically increasing replica value
	applyReplicaPatch := func(replicas int) rest.TransformFunc {
		return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
			currentObject.(*autoscaling.Scale).Spec.Replicas = int32(replicas)
			return currentObject, nil
		}
	}
	for i := 0; i < 100; i++ {
		result, _, err := scaleStore.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyReplicaPatch(i)), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("error patching scale: %v", err)
		}
		scale := result.(*autoscaling.Scale)
		if scale.Spec.Replicas != int32(i) {
			t.Errorf("wrong replicas count: expected: %d got: %d", i, scale.Spec.Replicas)
		}
	}
	close(stopCh)
	wg.Wait()
}
