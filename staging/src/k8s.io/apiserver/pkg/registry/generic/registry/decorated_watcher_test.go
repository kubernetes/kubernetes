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
	"context"
	"io"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
)

type fakeCacheableObject struct {
	obj runtime.Object
}

func (f *fakeCacheableObject) GetObjectKind() schema.ObjectKind {
	return f.obj.GetObjectKind()
}

func (f *fakeCacheableObject) DeepCopyObject() runtime.Object {
	return &fakeCacheableObject{obj: f.obj.DeepCopyObject()}
}

func (f *fakeCacheableObject) CacheEncode(id runtime.Identifier, encode func(runtime.Object, io.Writer) error, w io.Writer) error {
	return encode(f.GetObject(), w)
}

func (f *fakeCacheableObject) GetObject() runtime.Object {
	return f.obj.DeepCopyObject()
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
		w.Add(&fakeCacheableObject{obj: cachedPod})
		w.Modify(&fakeCacheableObject{obj: cachedPod})
		w.Delete(&fakeCacheableObject{obj: cachedPod})
	}()

	expectErrorEvent(t, dw) // expect error is plumbed and doesn't force close the watcher
	expectPodEvent(t, dw, watch.Added)
	expectErrorEvent(t, dw) // expect error is plumbed and doesn't force close the watcher
	expectPodEvent(t, dw, watch.Modified)
	expectErrorEvent(t, dw) // expect error is plumbed and doesn't force close the watcher
	expectPodEvent(t, dw, watch.Deleted)
	expectPodEvent(t, dw, watch.Added)
	expectPodEvent(t, dw, watch.Modified)
	expectPodEvent(t, dw, watch.Deleted)

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

func expectPodEvent(t *testing.T, dw *decoratedWatcher, watchType watch.EventType) {
	select {
	case e := <-dw.ResultChan():
		pod, ok := e.Object.(*example.Pod)
		if !ok {
			t.Fatalf("Should received object of type *api.Pod, get type (%T)", e.Object)
		}
		if pod.Annotations["decorated"] != "true" {
			t.Fatalf("pod.Annotations[\"decorated\"], want=%s, get=%s", "true", pod.Labels["decorated"])
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
