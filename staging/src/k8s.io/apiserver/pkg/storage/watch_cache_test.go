/*
Copyright 2014 The Kubernetes Authors.

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
	"strconv"
	"testing"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
)

func makeTestPod(name string, resourceVersion uint64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       "ns",
			Name:            name,
			ResourceVersion: strconv.FormatUint(resourceVersion, 10),
		},
	}
}

// newTestWatchCache just adds a fake clock.
func newTestWatchCache(capacity int) *watchCache {
	keyFunc := func(obj runtime.Object) (string, error) {
		return NamespaceKeyFunc("prefix", obj)
	}
	getAttrsFunc := func(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
		return nil, nil, false, nil
	}
	wc := newWatchCache(capacity, keyFunc, getAttrsFunc)
	wc.clock = clock.NewFakeClock(time.Now())
	return wc
}

func TestWatchCacheBasic(t *testing.T) {
	store := newTestWatchCache(2)

	// Test Add/Update/Delete.
	pod1 := makeTestPod("pod", 1)
	if err := store.Add(pod1); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if item, ok, _ := store.Get(pod1); !ok {
		t.Errorf("didn't find pod")
	} else {
		if !apiequality.Semantic.DeepEqual(&storeElement{Key: "prefix/ns/pod", Object: pod1}, item) {
			t.Errorf("expected %v, got %v", pod1, item)
		}
	}
	pod2 := makeTestPod("pod", 2)
	if err := store.Update(pod2); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if item, ok, _ := store.Get(pod2); !ok {
		t.Errorf("didn't find pod")
	} else {
		if !apiequality.Semantic.DeepEqual(&storeElement{Key: "prefix/ns/pod", Object: pod2}, item) {
			t.Errorf("expected %v, got %v", pod1, item)
		}
	}
	pod3 := makeTestPod("pod", 3)
	if err := store.Delete(pod3); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, ok, _ := store.Get(pod3); ok {
		t.Errorf("found pod")
	}

	// Test List.
	store.Add(makeTestPod("pod1", 4))
	store.Add(makeTestPod("pod2", 5))
	store.Add(makeTestPod("pod3", 6))
	{
		podNames := sets.String{}
		for _, item := range store.List() {
			podNames.Insert(item.(*storeElement).Object.(*v1.Pod).ObjectMeta.Name)
		}
		if !podNames.HasAll("pod1", "pod2", "pod3") {
			t.Errorf("missing pods, found %v", podNames)
		}
		if len(podNames) != 3 {
			t.Errorf("found missing/extra items")
		}
	}

	// Test Replace.
	store.Replace([]interface{}{
		makeTestPod("pod4", 7),
		makeTestPod("pod5", 8),
	}, "8")
	{
		podNames := sets.String{}
		for _, item := range store.List() {
			podNames.Insert(item.(*storeElement).Object.(*v1.Pod).ObjectMeta.Name)
		}
		if !podNames.HasAll("pod4", "pod5") {
			t.Errorf("missing pods, found %v", podNames)
		}
		if len(podNames) != 2 {
			t.Errorf("found missing/extra items")
		}
	}
}

func TestEvents(t *testing.T) {
	store := newTestWatchCache(5)

	store.Add(makeTestPod("pod", 3))

	// Test for Added event.
	{
		_, err := store.GetAllEventsSince(1)
		if err == nil {
			t.Errorf("expected error too old")
		}
		if _, ok := err.(*errors.StatusError); !ok {
			t.Errorf("expected error to be of type StatusError")
		}
	}
	{
		result, err := store.GetAllEventsSince(2)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(result) != 1 {
			t.Fatalf("unexpected events: %v", result)
		}
		if result[0].Type != watch.Added {
			t.Errorf("unexpected event type: %v", result[0].Type)
		}
		pod := makeTestPod("pod", uint64(3))
		if !apiequality.Semantic.DeepEqual(pod, result[0].Object) {
			t.Errorf("unexpected item: %v, expected: %v", result[0].Object, pod)
		}
		if result[0].PrevObject != nil {
			t.Errorf("unexpected item: %v", result[0].PrevObject)
		}
	}

	store.Update(makeTestPod("pod", 4))
	store.Update(makeTestPod("pod", 5))

	// Test with not full cache.
	{
		_, err := store.GetAllEventsSince(1)
		if err == nil {
			t.Errorf("expected error too old")
		}
	}
	{
		result, err := store.GetAllEventsSince(3)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(result) != 2 {
			t.Fatalf("unexpected events: %v", result)
		}
		for i := 0; i < 2; i++ {
			if result[i].Type != watch.Modified {
				t.Errorf("unexpected event type: %v", result[i].Type)
			}
			pod := makeTestPod("pod", uint64(i+4))
			if !apiequality.Semantic.DeepEqual(pod, result[i].Object) {
				t.Errorf("unexpected item: %v, expected: %v", result[i].Object, pod)
			}
			prevPod := makeTestPod("pod", uint64(i+3))
			if !apiequality.Semantic.DeepEqual(prevPod, result[i].PrevObject) {
				t.Errorf("unexpected item: %v, expected: %v", result[i].PrevObject, prevPod)
			}
		}
	}

	for i := 6; i < 10; i++ {
		store.Update(makeTestPod("pod", uint64(i)))
	}

	// Test with full cache - there should be elements from 5 to 9.
	{
		_, err := store.GetAllEventsSince(3)
		if err == nil {
			t.Errorf("expected error too old")
		}
	}
	{
		result, err := store.GetAllEventsSince(4)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(result) != 5 {
			t.Fatalf("unexpected events: %v", result)
		}
		for i := 0; i < 5; i++ {
			pod := makeTestPod("pod", uint64(i+5))
			if !apiequality.Semantic.DeepEqual(pod, result[i].Object) {
				t.Errorf("unexpected item: %v, expected: %v", result[i].Object, pod)
			}
		}
	}

	// Test for delete event.
	store.Delete(makeTestPod("pod", uint64(10)))

	{
		result, err := store.GetAllEventsSince(9)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(result) != 1 {
			t.Fatalf("unexpected events: %v", result)
		}
		if result[0].Type != watch.Deleted {
			t.Errorf("unexpected event type: %v", result[0].Type)
		}
		pod := makeTestPod("pod", uint64(10))
		if !apiequality.Semantic.DeepEqual(pod, result[0].Object) {
			t.Errorf("unexpected item: %v, expected: %v", result[0].Object, pod)
		}
		prevPod := makeTestPod("pod", uint64(9))
		if !apiequality.Semantic.DeepEqual(prevPod, result[0].PrevObject) {
			t.Errorf("unexpected item: %v, expected: %v", result[0].PrevObject, prevPod)
		}
	}
}

func TestWaitUntilFreshAndList(t *testing.T) {
	store := newTestWatchCache(3)

	// In background, update the store.
	go func() {
		store.Add(makeTestPod("foo", 2))
		store.Add(makeTestPod("bar", 5))
	}()

	list, resourceVersion, err := store.WaitUntilFreshAndList(5, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resourceVersion != 5 {
		t.Errorf("unexpected resourceVersion: %v, expected: 5", resourceVersion)
	}
	if len(list) != 2 {
		t.Errorf("unexpected list returned: %#v", list)
	}
}

func TestWaitUntilFreshAndGet(t *testing.T) {
	store := newTestWatchCache(3)

	// In background, update the store.
	go func() {
		store.Add(makeTestPod("foo", 2))
		store.Add(makeTestPod("bar", 5))
	}()

	obj, exists, resourceVersion, err := store.WaitUntilFreshAndGet(5, "prefix/ns/bar", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resourceVersion != 5 {
		t.Errorf("unexpected resourceVersion: %v, expected: 5", resourceVersion)
	}
	if !exists {
		t.Fatalf("no results returned: %#v", obj)
	}
	if !apiequality.Semantic.DeepEqual(&storeElement{Key: "prefix/ns/bar", Object: makeTestPod("bar", 5)}, obj) {
		t.Errorf("unexpected element returned: %#v", obj)
	}
}

func TestWaitUntilFreshAndListTimeout(t *testing.T) {
	store := newTestWatchCache(3)
	fc := store.clock.(*clock.FakeClock)

	// In background, step clock after the below call starts the timer.
	go func() {
		for !fc.HasWaiters() {
			time.Sleep(time.Millisecond)
		}
		fc.Step(blockTimeout)

		// Add an object to make sure the test would
		// eventually fail instead of just waiting
		// forever.
		time.Sleep(30 * time.Second)
		store.Add(makeTestPod("bar", 5))
	}()

	_, _, err := store.WaitUntilFreshAndList(5, nil)
	if err == nil {
		t.Fatalf("unexpected lack of timeout error")
	}
}

type testLW struct {
	ListFunc  func(options metav1.ListOptions) (runtime.Object, error)
	WatchFunc func(options metav1.ListOptions) (watch.Interface, error)
}

func (t *testLW) List(options metav1.ListOptions) (runtime.Object, error) {
	return t.ListFunc(options)
}
func (t *testLW) Watch(options metav1.ListOptions) (watch.Interface, error) {
	return t.WatchFunc(options)
}

func TestReflectorForWatchCache(t *testing.T) {
	store := newTestWatchCache(5)

	{
		_, version, err := store.WaitUntilFreshAndList(0, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if version != 0 {
			t.Errorf("unexpected resource version: %d", version)
		}
	}

	lw := &testLW{
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			fw := watch.NewFake()
			go fw.Stop()
			return fw, nil
		},
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return &v1.PodList{ListMeta: metav1.ListMeta{ResourceVersion: "10"}}, nil
		},
	}
	r := cache.NewReflector(lw, &v1.Pod{}, store, 0)
	r.ListAndWatch(wait.NeverStop)

	{
		_, version, err := store.WaitUntilFreshAndList(10, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if version != 10 {
			t.Errorf("unexpected resource version: %d", version)
		}
	}
}
