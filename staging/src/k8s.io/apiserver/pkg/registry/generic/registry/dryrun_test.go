/*
Copyright 2018 The Kubernetes Authors.

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

package registry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/api/apitesting"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/ptr"
)

func NewDryRunnableTestStorage(t *testing.T) (DryRunnableStorage, func()) {
	server, sc := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	sc.Codec = apitesting.TestStorageCodec(codecs, examplev1.SchemeGroupVersion)
	s, destroy, err := factory.Create(*sc.ForResource(schema.GroupResource{Resource: "pods"}), nil, nil, nil, "/pods")
	if err != nil {
		t.Fatalf("Error creating storage: %v", err)
	}
	return DryRunnableStorage{Storage: s, Codec: sc.Codec}, func() {
		destroy()
		server.Terminate(t)
	}
}

func UnstructuredOrDie(j string) *unstructured.Unstructured {
	m := map[string]interface{}{}
	err := json.Unmarshal([]byte(j), &m)
	if err != nil {
		panic(fmt.Errorf("Failed to unmarshal into Unstructured: %v", err))
	}
	return &unstructured.Unstructured{Object: m}
}

func TestDryRunCreateDoesntCreate(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "/pods/key", obj, out, 0, true)
	if err != nil {
		t.Fatalf("Failed to create new dry-run object: %v", err)
	}

	err = s.Get(context.Background(), "/pods/key", storage.GetOptions{}, out)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeKeyNotFound {
		t.Errorf("Expected key to be not found, error: %v", err)
	}
}

func TestDryRunCreateReturnsObject(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "/pods/key", obj, out, 0, true)
	if err != nil {
		t.Fatalf("Failed to create new dry-run object: %v", err)
	}

	if !reflect.DeepEqual(obj, out) {
		t.Errorf("Returned object different from input object:\nExpected: %v\nGot: %v", obj, out)
	}
}

func TestDryRunCreateExistingObjectFails(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "/pods/key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	err = s.Create(context.Background(), "/pods/key", obj, out, 0, true)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeKeyExists {
		t.Errorf("Expected KeyExists error: %v", err)
	}

}

func TestDryRunUpdateMissingObjectFails(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		return input, nil, errors.New("UpdateFunction shouldn't be called")
	}

	err := s.GuaranteedUpdate(context.Background(), "/pods/key", obj, false, nil, updateFunc, true, nil)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeKeyNotFound {
		t.Errorf("Expected key to be not found, error: %v", err)
	}
}

func TestDryRunUpdatePreconditions(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod", "metadata": {"uid": "my-uid"}}`)
	out := UnstructuredOrDie(`{}`)
	err := s.Create(context.Background(), "/pods/key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		u, ok := input.(*unstructured.Unstructured)
		if !ok {
			return input, nil, errors.New("Input object is not unstructured")
		}
		unstructured.SetNestedField(u.Object, "value", "field")
		return u, nil, nil
	}
	wrongID := types.UID("wrong-uid")
	myID := types.UID("my-uid")
	err = s.GuaranteedUpdate(context.Background(), "/pods/key", obj, false, &storage.Preconditions{UID: &wrongID}, updateFunc, true, nil)
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeInvalidObj {
		t.Errorf("Expected invalid object, error: %v", err)
	}

	err = s.GuaranteedUpdate(context.Background(), "/pods/key", obj, false, &storage.Preconditions{UID: &myID}, updateFunc, true, nil)
	if err != nil {
		t.Fatalf("Failed to update with valid precondition: %v", err)
	}
}

func TestDryRunUpdateDoesntUpdate(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	created := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "/pods/key", obj, created, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		u, ok := input.(*unstructured.Unstructured)
		if !ok {
			return input, nil, errors.New("Input object is not unstructured")
		}
		unstructured.SetNestedField(u.Object, "value", "field")
		return u, nil, nil
	}

	err = s.GuaranteedUpdate(context.Background(), "/pods/key", obj, false, nil, updateFunc, true, nil)
	if err != nil {
		t.Fatalf("Failed to dry-run update: %v", err)
	}
	out := UnstructuredOrDie(`{}`)
	err = s.Get(context.Background(), "/pods/key", storage.GetOptions{}, out)
	if err != nil {
		t.Fatalf("Failed to get storage: %v", err)
	}
	if !reflect.DeepEqual(created, out) {
		t.Fatalf("Returned object %q different from expected %q", created, out)
	}
}

func TestDryRunUpdateReturnsObject(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "/pods/key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		u, ok := input.(*unstructured.Unstructured)
		if !ok {
			return input, nil, errors.New("Input object is not unstructured")
		}
		unstructured.SetNestedField(u.Object, "value", "field")
		return u, nil, nil
	}

	err = s.GuaranteedUpdate(context.Background(), "/pods/key", obj, false, nil, updateFunc, true, nil)
	if err != nil {
		t.Fatalf("Failed to dry-run update: %v", err)
	}
	out = UnstructuredOrDie(`{"field": "value", "kind": "Pod", "metadata": {"resourceVersion": "2"}}`)
	if !reflect.DeepEqual(obj, out) {
		t.Fatalf("Returned object %#v different from expected %#v", obj, out)
	}
}

func TestDryRunDeleteDoesntDelete(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "/pods/key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	err = s.Delete(context.Background(), "/pods/key", out, nil, rest.ValidateAllObjectFunc, true, nil, storage.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to dry-run delete the object: %v", err)
	}

	err = s.Get(context.Background(), "/pods/key", storage.GetOptions{}, out)
	if err != nil {
		t.Fatalf("Failed to retrieve dry-run deleted object: %v", err)
	}
}

func TestDryRunDeleteMissingObjectFails(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	out := UnstructuredOrDie(`{}`)
	err := s.Delete(context.Background(), "/pods/key", out, nil, rest.ValidateAllObjectFunc, true, nil, storage.DeleteOptions{})
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeKeyNotFound {
		t.Errorf("Expected key to be not found, error: %v", err)
	}
}

func TestDryRunDeleteReturnsObject(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod"}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "/pods/key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	out = UnstructuredOrDie(`{}`)
	expected := UnstructuredOrDie(`{"kind": "Pod", "metadata": {"resourceVersion": "2"}}`)
	err = s.Delete(context.Background(), "/pods/key", out, nil, rest.ValidateAllObjectFunc, true, nil, storage.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete with valid precondition: %v", err)
	}
	if !reflect.DeepEqual(expected, out) {
		t.Fatalf("Returned object %q doesn't match expected: %q", out, expected)
	}
}

func TestDryRunDeletePreconditions(t *testing.T) {
	s, destroy := NewDryRunnableTestStorage(t)
	defer destroy()

	obj := UnstructuredOrDie(`{"kind": "Pod", "metadata": {"uid": "my-uid"}}`)
	out := UnstructuredOrDie(`{}`)

	err := s.Create(context.Background(), "/pods/key", obj, out, 0, false)
	if err != nil {
		t.Fatalf("Failed to create new object: %v", err)
	}

	wrongID := types.UID("wrong-uid")
	myID := types.UID("my-uid")
	err = s.Delete(context.Background(), "/pods/key", out, &storage.Preconditions{UID: &wrongID}, rest.ValidateAllObjectFunc, true, nil, storage.DeleteOptions{})
	if e, ok := err.(*storage.StorageError); !ok || e.Code != storage.ErrCodeInvalidObj {
		t.Errorf("Expected invalid object, error: %v", err)
	}

	err = s.Delete(context.Background(), "/pods/key", out, &storage.Preconditions{UID: &myID}, rest.ValidateAllObjectFunc, true, nil, storage.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete with valid precondition: %v", err)
	}
}

func TestStoreDoesntMutateCachedObject(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ConsistentGetFromCache, true)

	gvk := examplev1.SchemeGroupVersion.WithKind("Pod")
	listGVK := examplev1.SchemeGroupVersion.WithKind("PodList")

	storageModes := []struct {
		name   string
		dryRun bool
	}{
		{name: "normal", dryRun: false},
		{name: "dry-run", dryRun: true},
	}
	ops := []storeMutationOp{
		opUpdate,
		opDelete,
		opGracefulDelete,
	}
	objectTypes := []struct {
		name        string
		codec       runtime.Codec
		newFunc     func() runtime.Object
		newListFunc func() runtime.Object
		newTestObj  func() runtime.Object
	}{
		{
			name:        "structured",
			codec:       apitesting.TestStorageCodec(codecs, examplev1.SchemeGroupVersion),
			newFunc:     func() runtime.Object { return &example.Pod{} },
			newListFunc: func() runtime.Object { return &example.PodList{} },
			newTestObj:  newStructuredTestPod,
		},
		{
			name:  "unstructured",
			codec: unstructured.UnstructuredJSONScheme,
			newFunc: func() runtime.Object {
				u := &unstructured.Unstructured{}
				u.SetGroupVersionKind(gvk)
				return u
			},
			newListFunc: func() runtime.Object {
				u := &unstructured.UnstructuredList{}
				u.SetGroupVersionKind(listGVK)
				return u
			},
			newTestObj: newUnstructuredTestPod,
		},
	}

	for _, mode := range storageModes {
		for _, op := range ops {
			for _, objType := range objectTypes {
				name := fmt.Sprintf("%s/%s/%s", mode.name, op, objType.name)
				t.Run(name, func(t *testing.T) {
					destroy, store := newTestCachedStore(t, objType.codec, objType.newFunc, objType.newListFunc)
					t.Cleanup(destroy)
					testStoreDoesntMutateCachedObject(t, store, objType.newTestObj, mode.dryRun, op)
				})
			}
		}
	}
}

type storeMutationOp string

const (
	opUpdate         storeMutationOp = "update"
	opDelete         storeMutationOp = "delete"
	opGracefulDelete storeMutationOp = "graceful-delete"
)

func testStoreDoesntMutateCachedObject(t *testing.T, store *Store, newTestObj func() runtime.Object, dryRun bool, op storeMutationOp) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "default")
	key, err := store.KeyFunc(ctx, "pod-a")
	if err != nil {
		t.Fatalf("KeyFunc failed: %v", err)
	}
	if err := store.Storage.Create(ctx, key, newTestObj(), store.NewFunc(), 0, false); err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}

	before, err := store.Get(ctx, "pod-a", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get pod before %s: %v", op, err)
	}
	wantBefore := before.DeepCopyObject()

	var dryRunOpts []string
	if dryRun {
		dryRunOpts = []string{metav1.DryRunAll}
	}

	var returned runtime.Object
	switch op {
	case opUpdate:
		returned, _, err = store.Update(
			ctx,
			"pod-a",
			rest.DefaultUpdatedObjectInfo(nil, func(ctx context.Context, _, oldObj runtime.Object) (runtime.Object, error) {
				mutateAnnotations(t, oldObj, "updated", "true")
				return oldObj, nil
			}),
			rest.ValidateAllObjectFunc,
			rest.ValidateAllObjectUpdateFunc,
			false,
			&metav1.UpdateOptions{DryRun: dryRunOpts},
		)
	case opDelete:
		returned, _, err = store.Delete(
			ctx,
			"pod-a",
			func(ctx context.Context, obj runtime.Object) error {
				mutateAnnotations(t, obj, "validated", "true")
				return nil
			},
			&metav1.DeleteOptions{
				GracePeriodSeconds: ptr.To[int64](0),
				DryRun:             dryRunOpts,
			},
		)
	case opGracefulDelete:
		returned, _, err = store.Delete(
			ctx,
			"pod-a",
			func(ctx context.Context, obj runtime.Object) error {
				mutateAnnotations(t, obj, "validated", "true")
				return nil
			},
			&metav1.DeleteOptions{
				GracePeriodSeconds: ptr.To[int64](30),
				DryRun:             dryRunOpts,
			},
		)
	default:
		t.Fatalf("Unknown operation %q", op)
	}
	if err != nil {
		t.Fatalf("Failed to execute %s (dryRun=%v): %v", op, dryRun, err)
	}

	wantAfter := returned.DeepCopyObject()
	if dryRun {
		wantAfter = wantBefore
	} else if op == opDelete {
		wantAfter = nil
	}

	// Mutating the returned object must also not affect any cached object.
	mutateAnnotations(t, returned, "returned", "true")

	if diff := cmp.Diff(wantBefore, before); diff != "" {
		t.Errorf("Previously fetched cached object changed (-want +got):\n%s", diff)
	}

	after, err := store.Get(ctx, "pod-a", &metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		after, err = nil, nil
	}
	if err != nil {
		t.Fatalf("Failed to get pod after %s: %v", op, err)
	}
	if diff := cmp.Diff(wantAfter, after); diff != "" {
		t.Errorf("Cached object mismatch (-want +got):\n%s", diff)
	}
}

func newTestCachedStore(t *testing.T, codec runtime.Codec, newFunc, newListFunc func() runtime.Object) (factory.DestroyFunc, *Store) {
	t.Helper()
	destroy, store := newTestStoreRegistry(t, scheme, codec, newFunc, newListFunc, true, getPodAttrs)
	store.DeleteStrategy = &testConditionalGracefulStrategy{testRESTStrategy{scheme, names.SimpleNameGenerator, true, false, true}}
	store.ReturnDeletedObject = true
	store.AfterDelete = func(obj runtime.Object, options *metav1.DeleteOptions) {
		mutateAnnotations(t, obj, "after-delete", "true")
	}
	return destroy, store
}

type testConditionalGracefulStrategy struct {
	testRESTStrategy
}

func (t testConditionalGracefulStrategy) CheckGracefulDelete(ctx context.Context, obj runtime.Object, options *metav1.DeleteOptions) bool {
	return options.GracePeriodSeconds != nil && *options.GracePeriodSeconds > 0
}

func newStructuredTestPod() runtime.Object {
	return &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:                  "default",
			Name:                       "pod-a",
			Generation:                 1,
			DeletionGracePeriodSeconds: ptr.To[int64](-5),
			Annotations:                map[string]string{"existing": "val"},
		},
	}
}

func newUnstructuredTestPod() runtime.Object {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": examplev1.SchemeGroupVersion.String(),
			"kind":       "Pod",
			"metadata": map[string]interface{}{
				"namespace":                  "default",
				"name":                       "pod-a",
				"generation":                 int64(1),
				"deletionGracePeriodSeconds": int64(-5),
				"annotations": map[string]interface{}{
					"existing": "val",
				},
			},
		},
	}
}

func mutateAnnotations(t *testing.T, obj runtime.Object, key, val string) {
	t.Helper()
	accessor, err := meta.Accessor(obj)
	if err != nil {
		t.Fatalf("meta.Accessor failed: %v", err)
	}
	annotations := accessor.GetAnnotations()
	annotations[key] = val
	delete(annotations, "existing")
	accessor.SetAnnotations(annotations)
}
