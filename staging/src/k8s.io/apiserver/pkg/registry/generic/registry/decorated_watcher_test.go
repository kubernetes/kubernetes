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

package registry

import (
	"bytes"
	"context"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

type fakeCacheableObject struct {
	lock         sync.RWMutex
	obj          runtime.Object
	decorateOnce sync.Once
	decorator    func(runtime.Object)
}

func (f *fakeCacheableObject) GetObjectKind() schema.ObjectKind {
	return f.obj.GetObjectKind()
}

func (f *fakeCacheableObject) DeepCopyObject() runtime.Object {
	f.lock.RLock()
	defer f.lock.RUnlock()
	cp := &fakeCacheableObject{obj: f.obj.DeepCopyObject()}
	cp.Decorate(f.decorator)
	return cp
}

func (f *fakeCacheableObject) CacheEncode(id runtime.Identifier, encode func(runtime.Object, io.Writer) error, w io.Writer) error {
	return encode(f.GetObject(), w)
}

func (f *fakeCacheableObject) GetObject() runtime.Object {
	f.lock.RLock()
	defer f.lock.RUnlock()
	obj := f.obj.DeepCopyObject()
	if f.decorator != nil {
		f.decorator(obj)
	}
	return obj
}

func (f *fakeCacheableObject) Decorate(decorator func(runtime.Object)) {
	if decorator == nil {
		return
	}
	f.decorateOnce.Do(func() {
		f.lock.Lock()
		defer f.lock.Unlock()
		f.decorator = decorator
	})
}

func TestDecoratedWatcher(t *testing.T) {
	w := watch.NewFake()
	decorator := func(obj runtime.Object) {
		if pod, ok := obj.(*example.Pod); ok {
			pod.Annotations = map[string]string{"decorated": "true"}
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	dw := newDecoratedWatcher(ctx, w, decorator)
	defer dw.Stop()

	cachedPod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "cached"}}

	go func() {
		w.Error(&metav1.Status{Status: "Failure"})
		w.Add(&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
		w.Error(&metav1.Status{Status: "Failure"})
		w.Modify(&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
		w.Error(&metav1.Status{Status: "Failure"})
		w.Delete(&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
		w.Action(watch.Bookmark, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
		w.Add(&fakeCacheableObject{obj: cachedPod})
		w.Modify(&fakeCacheableObject{obj: cachedPod})
		w.Delete(&fakeCacheableObject{obj: cachedPod})
	}()

	expectErrorEvent(t, dw) // expect error is plumbed and doesn't force close the watcher
	expectPodEvent(t, dw, watch.Added, true)
	expectErrorEvent(t, dw) // expect error is plumbed and doesn't force close the watcher
	expectPodEvent(t, dw, watch.Modified, true)
	expectErrorEvent(t, dw) // expect error is plumbed and doesn't force close the watcher
	expectPodEvent(t, dw, watch.Deleted, true)
	expectPodEvent(t, dw, watch.Bookmark, false)
	expectPodEvent(t, dw, watch.Added, true)
	expectPodEvent(t, dw, watch.Modified, true)
	expectPodEvent(t, dw, watch.Deleted, true)

	if cachedPod.Annotations["decorated"] != "" {
		t.Errorf("expected underlying cached object not to be mutated, got annotations: %v", cachedPod.Annotations)
	}

	// cancel the passed-in context to simulate request timeout
	cancel()

	// expect the decorated channel to be closed
	select {
	case e, ok := <-dw.ResultChan():
		if ok {
			t.Errorf("expected result chan closed, got %#v", e)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timeout after %v", wait.ForeverTestTimeout)
	}

	// expect the underlying watcher to have been stopped as a result of the context cancellation
	if !w.IsStopped() {
		t.Errorf("expected underlying watcher to be stopped")
	}
}

func expectPodEvent(t *testing.T, dw *decoratedWatcher, watchType watch.EventType, decorated bool) {
	select {
	case e := <-dw.ResultChan():
		obj := e.Object
		if co, ok := obj.(runtime.CacheableObject); ok {
			obj = co.GetObject()
		}
		pod, ok := obj.(*example.Pod)
		if !ok {
			t.Fatalf("Should received object of type *api.Pod, get type (%T)", obj)
		}
		if (pod.Annotations["decorated"] == "true") != decorated {
			t.Fatalf("pod.Annotations[\"decorated\"]==\"true\", want=%v, get=%v", decorated, pod.Annotations["decorated"] == "true")
		}
		if e.Type != watchType {
			t.Fatalf("expected type %s, got %s", watchType, e.Type)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
}

func expectErrorEvent(t *testing.T, dw *decoratedWatcher) {
	select {
	case e := <-dw.ResultChan():
		_, ok := e.Object.(*metav1.Status)
		if !ok {
			t.Fatalf("Should received object of type *metav1.Status, get type (%T)", e.Object)
		}
		if e.Type != watch.Error {
			t.Fatalf("expected type %s, got %s", watch.Error, e.Type)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
}

func TestStoreWatchDecoratedWithCacher(t *testing.T) {
	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := newTestGenericStoreRegistry(t, scheme, true)
	defer destroyFunc()

	var decorateCalls atomic.Int32
	registry.Decorator = func(obj runtime.Object) {
		if pod, ok := obj.(*example.Pod); ok {
			decorateCalls.Add(1)
			if pod.Annotations == nil {
				pod.Annotations = map[string]string{}
			}
			pod.Annotations["decorated"] = "true"
		}
	}

	w1, err := registry.WatchPredicate(genericapirequest.NewContext(), matchEverything(), "0", nil)
	if err != nil {
		t.Fatalf("Watch w1 failed: %v", err)
	}
	defer w1.Stop()

	w2, err := registry.WatchPredicate(genericapirequest.NewContext(), matchEverything(), "0", nil)
	if err != nil {
		t.Fatalf("Watch w2 failed: %v", err)
	}
	defer w2.Stop()

	pod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	created, err := registry.Create(testContext, pod, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	var encodeCalls atomic.Int32
	encodeDecorated := func(obj runtime.Object, w io.Writer) error {
		encodeCalls.Add(1)
		p := obj.(*example.Pod)
		_, err := w.Write([]byte(p.Annotations["decorated"]))
		return err
	}

	assertSharedCacheEncode := func(wantType watch.EventType) {
		t.Helper()
		decorateCalls.Store(0)
		encodeCalls.Store(0)
		for i, w := range []watch.Interface{w1, w2} {
			select {
			case e := <-w.ResultChan():
				if e.Type != wantType {
					t.Fatalf("watcher %d: expected event type %v, got %v", i, wantType, e.Type)
				}
				co, ok := e.Object.(runtime.CacheableObject)
				if !ok {
					t.Fatalf("watcher %d: expected runtime.CacheableObject, got %T", i, e.Object)
				}
				var buf bytes.Buffer
				if err := co.CacheEncode("test-codec", encodeDecorated, &buf); err != nil {
					t.Fatalf("watcher %d: CacheEncode failed: %v", i, err)
				}
				if got := buf.String(); got != "true" {
					t.Fatalf("watcher %d: expected decorated=true, got %q", i, got)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("watcher %d: timeout waiting for %v event", i, wantType)
			}
		}
		if got := decorateCalls.Load(); got != 1 {
			t.Fatalf("after %v across 2 watchers: expected 1 decorate call, got %d", wantType, got)
		}
		if got := encodeCalls.Load(); got != 1 {
			t.Fatalf("after %v across 2 watchers: expected 1 encode call, got %d", wantType, got)
		}
	}

	assertSharedCacheEncode(watch.Added)

	toUpdate := created.(*example.Pod).DeepCopy()
	toUpdate.Annotations = nil
	toUpdate.Spec.NodeName = "machine-2"
	if _, _, err := registry.Update(testContext, toUpdate.Name, rest.DefaultUpdatedObjectInfo(toUpdate), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	assertSharedCacheEncode(watch.Modified)

	if _, _, err := registry.Delete(testContext, pod.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	assertSharedCacheEncode(watch.Deleted)
}
