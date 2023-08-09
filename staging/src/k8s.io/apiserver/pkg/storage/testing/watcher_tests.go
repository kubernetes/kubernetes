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

package testing

import (
	"context"
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/value"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
)

func RunTestWatch(ctx context.Context, t *testing.T, store storage.Interface) {
	testWatch(ctx, t, store, false)
	testWatch(ctx, t, store, true)
}

// It tests that
// - first occurrence of objects should notify Add event
// - update should trigger Modified event
// - update that gets filtered should trigger Deleted event
func testWatch(ctx context.Context, t *testing.T, store storage.Interface, recursive bool) {
	basePod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: ""},
	}
	basePodAssigned := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: "bar"},
	}

	tests := []struct {
		name       string
		namespace  string
		key        string
		pred       storage.SelectionPredicate
		watchTests []*testWatchStruct
	}{{
		name:       "create a key",
		namespace:  fmt.Sprintf("test-ns-1-%t", recursive),
		watchTests: []*testWatchStruct{{basePod, true, watch.Added}},
		pred:       storage.Everything,
	}, {
		name:       "key updated to match predicate",
		namespace:  fmt.Sprintf("test-ns-2-%t", recursive),
		watchTests: []*testWatchStruct{{basePod, false, ""}, {basePodAssigned, true, watch.Added}},
		pred: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.ParseSelectorOrDie("spec.nodename=bar"),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"spec.nodename": pod.Spec.NodeName}, nil
			},
		},
	}, {
		name:       "update",
		namespace:  fmt.Sprintf("test-ns-3-%t", recursive),
		watchTests: []*testWatchStruct{{basePod, true, watch.Added}, {basePodAssigned, true, watch.Modified}},
		pred:       storage.Everything,
	}, {
		name:       "delete because of being filtered",
		namespace:  fmt.Sprintf("test-ns-4-%t", recursive),
		watchTests: []*testWatchStruct{{basePod, true, watch.Added}, {basePodAssigned, true, watch.Deleted}},
		pred: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.ParseSelectorOrDie("spec.nodename!=bar"),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"spec.nodename": pod.Spec.NodeName}, nil
			},
		},
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			watchKey := fmt.Sprintf("pods/%s", tt.namespace)
			key := watchKey + "/foo"
			if !recursive {
				watchKey = key
			}

			w, err := store.Watch(ctx, watchKey, storage.ListOptions{ResourceVersion: "0", Predicate: tt.pred, Recursive: recursive})
			if err != nil {
				t.Fatalf("Watch failed: %v", err)
			}
			var prevObj *example.Pod
			for _, watchTest := range tt.watchTests {
				out := &example.Pod{}
				err := store.GuaranteedUpdate(ctx, key, out, true, nil, storage.SimpleUpdate(
					func(runtime.Object) (runtime.Object, error) {
						obj := watchTest.obj.DeepCopy()
						obj.Namespace = tt.namespace
						return obj, nil
					}), nil)
				if err != nil {
					t.Fatalf("GuaranteedUpdate failed: %v", err)
				}
				if watchTest.expectEvent {
					expectObj := out
					if watchTest.watchType == watch.Deleted {
						expectObj = prevObj
						expectObj.ResourceVersion = out.ResourceVersion
					}
					testCheckResult(t, watchTest.watchType, w, expectObj)
				}
				prevObj = out
			}
			w.Stop()
			testCheckStop(t, w)
		})
	}
}

// RunTestWatchFromZero tests that
// - watch from 0 should sync up and grab the object added before
// - watch from 0 is able to return events for objects whose previous version has been compacted
func RunTestWatchFromZero(ctx context.Context, t *testing.T, store storage.Interface, compaction Compaction) {
	key, storedObj := testPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}})

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckResult(t, watch.Added, w, storedObj)
	w.Stop()

	// Update
	out := &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, out, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			return &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns", Annotations: map[string]string{"a": "1"}}}, nil
		}), nil)
	if err != nil {
		t.Fatalf("GuaranteedUpdate failed: %v", err)
	}

	// Make sure when we watch from 0 we receive an ADDED event
	w, err = store.Watch(ctx, key, storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckResult(t, watch.Added, w, out)
	w.Stop()

	if compaction == nil {
		t.Skip("compaction callback not provided")
	}

	// Update again
	out = &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, out, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			return &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}}, nil
		}), nil)
	if err != nil {
		t.Fatalf("GuaranteedUpdate failed: %v", err)
	}

	// Compact previous versions
	compaction(ctx, t, out.ResourceVersion)

	// Make sure we can still watch from 0 and receive an ADDED event
	w, err = store.Watch(ctx, key, storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckResult(t, watch.Added, w, out)
}

func RunTestDeleteTriggerWatch(ctx context.Context, t *testing.T, store storage.Interface) {
	key, storedObj := testPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}})
	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: storedObj.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	if err := store.Delete(ctx, key, &example.Pod{}, nil, storage.ValidateAllObjectFunc, nil); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	testCheckEventType(t, watch.Deleted, w)
}

func RunTestWatchFromNoneZero(ctx context.Context, t *testing.T, store storage.Interface) {
	key, storedObj := testPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}})

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: storedObj.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	out := &example.Pod{}
	store.GuaranteedUpdate(ctx, key, out, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			newObj := storedObj.DeepCopy()
			newObj.Annotations = map[string]string{"version": "2"}
			return newObj, nil
		}), nil)
	testCheckResult(t, watch.Modified, w, out)
}

func RunTestWatchError(ctx context.Context, t *testing.T, store InterfaceWithPrefixTransformer) {
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}}
	key := computePodKey(obj)

	// Compute the initial resource version from which we can start watching later.
	list := &example.PodList{}
	storageOpts := storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       storage.Everything,
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/pods", storageOpts, list); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if err := store.GuaranteedUpdate(ctx, key, &example.Pod{}, true, nil, storage.SimpleUpdate(
		func(runtime.Object) (runtime.Object, error) {
			return obj, nil
		}), nil); err != nil {
		t.Fatalf("GuaranteedUpdate failed: %v", err)
	}

	// Now trigger watch error by injecting failing transformer.
	revertTransformer := store.UpdatePrefixTransformer(
		func(previousTransformer *PrefixTransformer) value.Transformer {
			return &failingTransformer{}
		})
	defer revertTransformer()

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: list.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckEventType(t, watch.Error, w)
}

func RunTestWatchContextCancel(ctx context.Context, t *testing.T, store storage.Interface) {
	canceledCtx, cancel := context.WithCancel(ctx)
	cancel()
	// When we watch with a canceled context, we should detect that it's context canceled.
	// We won't take it as error and also close the watcher.
	w, err := store.Watch(canceledCtx, "/pods/not-existing", storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       storage.Everything,
	})
	if err != nil {
		t.Fatal(err)
	}

	select {
	case _, ok := <-w.ResultChan():
		if ok {
			t.Error("ResultChan() should be closed")
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timeout after %v", wait.ForeverTestTimeout)
	}
}

func RunTestWatchDeleteEventObjectHaveLatestRV(ctx context.Context, t *testing.T, store storage.Interface) {
	key, storedObj := testPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}})

	watchCtx, _ := context.WithTimeout(ctx, wait.ForeverTestTimeout)
	w, err := store.Watch(watchCtx, key, storage.ListOptions{ResourceVersion: storedObj.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}

	deletedObj := &example.Pod{}
	if err := store.Delete(ctx, key, deletedObj, &storage.Preconditions{}, storage.ValidateAllObjectFunc, nil); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// Verify that ResourceVersion has changed on deletion.
	if storedObj.ResourceVersion == deletedObj.ResourceVersion {
		t.Fatalf("ResourceVersion didn't changed on deletion: %s", deletedObj.ResourceVersion)
	}

	select {
	case event := <-w.ResultChan():
		watchedDeleteObj := event.Object.(*example.Pod)
		if e, a := deletedObj.ResourceVersion, watchedDeleteObj.ResourceVersion; e != a {
			t.Errorf("Unexpected resource version: %v, expected %v", a, e)
		}
	}
}

func RunTestWatchInitializationSignal(ctx context.Context, t *testing.T, store storage.Interface) {
	ctx, _ = context.WithTimeout(ctx, 5*time.Second)
	initSignal := utilflowcontrol.NewInitializationSignal()
	ctx = utilflowcontrol.WithInitializationSignal(ctx, initSignal)

	key, storedObj := testPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}})
	_, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: storedObj.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}

	initSignal.Wait()
}

// RunOptionalTestProgressNotify tests ProgressNotify feature of ListOptions.
// Given this feature is currently not explicitly used by higher layers of Kubernetes
// (it rather is used by wrappers of storage.Interface to implement its functionalities)
// this test is currently considered optional.
func RunOptionalTestProgressNotify(ctx context.Context, t *testing.T, store storage.Interface) {
	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name", Namespace: "test-ns"}}
	key := computePodKey(input)
	out := &example.Pod{}
	if err := store.Create(ctx, key, input, out, 0); err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	validateResourceVersion := resourceVersionNotOlderThan(out.ResourceVersion)

	opts := storage.ListOptions{
		ResourceVersion: out.ResourceVersion,
		Predicate:       storage.Everything,
		ProgressNotify:  true,
	}
	w, err := store.Watch(ctx, key, opts)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}

	// when we send a bookmark event, the client expects the event to contain an
	// object of the correct type, but with no fields set other than the resourceVersion
	testCheckResultFunc(t, watch.Bookmark, w, func(object runtime.Object) error {
		// first, check that we have the correct resource version
		obj, ok := object.(metav1.Object)
		if !ok {
			return fmt.Errorf("got %T, not metav1.Object", object)
		}
		if err := validateResourceVersion(obj.GetResourceVersion()); err != nil {
			return err
		}

		// then, check that we have the right type and content
		pod, ok := object.(*example.Pod)
		if !ok {
			return fmt.Errorf("got %T, not *example.Pod", object)
		}
		pod.ResourceVersion = ""
		ExpectNoDiff(t, "bookmark event should contain an object with no fields set other than resourceVersion", &example.Pod{}, pod)
		return nil
	})
}

type testWatchStruct struct {
	obj         *example.Pod
	expectEvent bool
	watchType   watch.EventType
}
