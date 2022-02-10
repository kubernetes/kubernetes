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

package cacher

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/client-go/tools/cache"
	testingclock "k8s.io/utils/clock/testing"
)

func makeTestPod(name string, resourceVersion uint64) *v1.Pod {
	return makeTestPodDetails(name, resourceVersion, "some-node", map[string]string{"k8s-app": "my-app"})
}

func makeTestPodDetails(name string, resourceVersion uint64, nodeName string, labels map[string]string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       "ns",
			Name:            name,
			ResourceVersion: strconv.FormatUint(resourceVersion, 10),
			Labels:          labels,
		},
		Spec: v1.PodSpec{
			NodeName: nodeName,
		},
	}
}

func makeTestStoreElement(pod *v1.Pod) *storeElement {
	return &storeElement{
		Key:    "prefix/ns/" + pod.Name,
		Object: pod,
		Labels: labels.Set(pod.Labels),
		Fields: fields.Set{"spec.nodeName": pod.Spec.NodeName},
	}
}

type testWatchCache struct {
	*watchCache
}

func (w *testWatchCache) getAllEventsSince(resourceVersion uint64) ([]*watchCacheEvent, error) {
	cacheInterval, err := w.getCacheIntervalForEvents(resourceVersion)
	if err != nil {
		return nil, err
	}

	result := []*watchCacheEvent{}
	for {
		event, err := cacheInterval.Next()
		if err != nil {
			return nil, err
		}
		if event == nil {
			break
		}
		result = append(result, event)
	}

	return result, nil
}

func (w *testWatchCache) getCacheIntervalForEvents(resourceVersion uint64) (*watchCacheInterval, error) {
	w.RLock()
	defer w.RUnlock()

	return w.getAllEventsSinceLocked(resourceVersion)
}

// newTestWatchCache just adds a fake clock.
func newTestWatchCache(capacity int, indexers *cache.Indexers) *testWatchCache {
	keyFunc := func(obj runtime.Object) (string, error) {
		return storage.NamespaceKeyFunc("prefix", obj)
	}
	getAttrsFunc := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			return nil, nil, fmt.Errorf("not a pod")
		}
		return labels.Set(pod.Labels), fields.Set{"spec.nodeName": pod.Spec.NodeName}, nil
	}
	versioner := etcd3.APIObjectVersioner{}
	mockHandler := func(*watchCacheEvent) {}
	wc := newWatchCache(keyFunc, mockHandler, getAttrsFunc, versioner, indexers, testingclock.NewFakeClock(time.Now()), reflect.TypeOf(&example.Pod{}))
	// To preserve behavior of tests that assume a given capacity,
	// resize it to th expected size.
	wc.capacity = capacity
	wc.cache = make([]*watchCacheEvent, capacity)
	wc.lowerBoundCapacity = min(capacity, defaultLowerBoundCapacity)
	wc.upperBoundCapacity = max(capacity, defaultUpperBoundCapacity)

	return &testWatchCache{watchCache: wc}
}

func TestWatchCacheBasic(t *testing.T) {
	store := newTestWatchCache(2, &cache.Indexers{})

	// Test Add/Update/Delete.
	pod1 := makeTestPod("pod", 1)
	if err := store.Add(pod1); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if item, ok, _ := store.Get(pod1); !ok {
		t.Errorf("didn't find pod")
	} else {
		expected := makeTestStoreElement(makeTestPod("pod", 1))
		if !apiequality.Semantic.DeepEqual(expected, item) {
			t.Errorf("expected %v, got %v", expected, item)
		}
	}
	pod2 := makeTestPod("pod", 2)
	if err := store.Update(pod2); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if item, ok, _ := store.Get(pod2); !ok {
		t.Errorf("didn't find pod")
	} else {
		expected := makeTestStoreElement(makeTestPod("pod", 2))
		if !apiequality.Semantic.DeepEqual(expected, item) {
			t.Errorf("expected %v, got %v", expected, item)
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
		expected := map[string]storeElement{
			"prefix/ns/pod1": *makeTestStoreElement(makeTestPod("pod1", 4)),
			"prefix/ns/pod2": *makeTestStoreElement(makeTestPod("pod2", 5)),
			"prefix/ns/pod3": *makeTestStoreElement(makeTestPod("pod3", 6)),
		}
		items := make(map[string]storeElement)
		for _, item := range store.List() {
			elem := item.(*storeElement)
			items[elem.Key] = *elem
		}
		if !apiequality.Semantic.DeepEqual(expected, items) {
			t.Errorf("expected %v, got %v", expected, items)
		}
	}

	// Test Replace.
	store.Replace([]interface{}{
		makeTestPod("pod4", 7),
		makeTestPod("pod5", 8),
	}, "8")
	{
		expected := map[string]storeElement{
			"prefix/ns/pod4": *makeTestStoreElement(makeTestPod("pod4", 7)),
			"prefix/ns/pod5": *makeTestStoreElement(makeTestPod("pod5", 8)),
		}
		items := make(map[string]storeElement)
		for _, item := range store.List() {
			elem := item.(*storeElement)
			items[elem.Key] = *elem
		}
		if !apiequality.Semantic.DeepEqual(expected, items) {
			t.Errorf("expected %v, got %v", expected, items)
		}
	}
}

func TestEvents(t *testing.T) {
	store := newTestWatchCache(5, &cache.Indexers{})

	// no dynamic-size cache to fit old tests.
	store.lowerBoundCapacity = 5
	store.upperBoundCapacity = 5

	store.Add(makeTestPod("pod", 3))

	// Test for Added event.
	{
		_, err := store.getAllEventsSince(1)
		if err == nil {
			t.Errorf("expected error too old")
		}
		if _, ok := err.(*errors.StatusError); !ok {
			t.Errorf("expected error to be of type StatusError")
		}
	}
	{
		result, err := store.getAllEventsSince(2)
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
		_, err := store.getAllEventsSince(1)
		if err == nil {
			t.Errorf("expected error too old")
		}
	}
	{
		result, err := store.getAllEventsSince(3)
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
		_, err := store.getAllEventsSince(3)
		if err == nil {
			t.Errorf("expected error too old")
		}
	}
	{
		result, err := store.getAllEventsSince(4)
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
		result, err := store.getAllEventsSince(9)
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

func TestMarker(t *testing.T) {
	store := newTestWatchCache(3, &cache.Indexers{})

	// First thing that is called when propagated from storage is Replace.
	store.Replace([]interface{}{
		makeTestPod("pod1", 5),
		makeTestPod("pod2", 9),
	}, "9")

	_, err := store.getAllEventsSince(8)
	if err == nil || !strings.Contains(err.Error(), "too old resource version") {
		t.Errorf("unexpected error: %v", err)
	}
	// Getting events from 8 should return no events,
	// even though there is a marker there.
	result, err := store.getAllEventsSince(9)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("unexpected result: %#v, expected no events", result)
	}

	pod := makeTestPod("pods", 12)
	store.Add(pod)
	// Getting events from 8 should still work and return one event.
	result, err = store.getAllEventsSince(9)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 1 || !apiequality.Semantic.DeepEqual(result[0].Object, pod) {
		t.Errorf("unexpected result: %#v, expected %v", result, pod)
	}
}

func TestWaitUntilFreshAndList(t *testing.T) {
	store := newTestWatchCache(3, &cache.Indexers{
		"l:label": func(obj interface{}) ([]string, error) {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				return nil, fmt.Errorf("not a pod %#v", obj)
			}
			if value, ok := pod.Labels["label"]; ok {
				return []string{value}, nil
			}
			return nil, nil
		},
		"f:spec.nodeName": func(obj interface{}) ([]string, error) {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				return nil, fmt.Errorf("not a pod %#v", obj)
			}
			return []string{pod.Spec.NodeName}, nil
		},
	})

	// In background, update the store.
	go func() {
		store.Add(makeTestPodDetails("pod1", 2, "node1", map[string]string{"label": "value1"}))
		store.Add(makeTestPodDetails("pod2", 3, "node1", map[string]string{"label": "value1"}))
		store.Add(makeTestPodDetails("pod3", 5, "node2", map[string]string{"label": "value2"}))
	}()

	// list by empty MatchValues.
	list, resourceVersion, indexUsed, err := store.WaitUntilFreshAndList(5, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resourceVersion != 5 {
		t.Errorf("unexpected resourceVersion: %v, expected: 5", resourceVersion)
	}
	if len(list) != 3 {
		t.Errorf("unexpected list returned: %#v", list)
	}
	if indexUsed != "" {
		t.Errorf("Used index %q but expected none to be used", indexUsed)
	}

	// list by label index.
	matchValues := []storage.MatchValue{
		{IndexName: "l:label", Value: "value1"},
		{IndexName: "f:spec.nodeName", Value: "node2"},
	}
	list, resourceVersion, indexUsed, err = store.WaitUntilFreshAndList(5, matchValues, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resourceVersion != 5 {
		t.Errorf("unexpected resourceVersion: %v, expected: 5", resourceVersion)
	}
	if len(list) != 2 {
		t.Errorf("unexpected list returned: %#v", list)
	}
	if indexUsed != "l:label" {
		t.Errorf("Used index %q but expected %q", indexUsed, "l:label")
	}

	// list with spec.nodeName index.
	matchValues = []storage.MatchValue{
		{IndexName: "l:not-exist-label", Value: "whatever"},
		{IndexName: "f:spec.nodeName", Value: "node2"},
	}
	list, resourceVersion, indexUsed, err = store.WaitUntilFreshAndList(5, matchValues, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resourceVersion != 5 {
		t.Errorf("unexpected resourceVersion: %v, expected: 5", resourceVersion)
	}
	if len(list) != 1 {
		t.Errorf("unexpected list returned: %#v", list)
	}
	if indexUsed != "f:spec.nodeName" {
		t.Errorf("Used index %q but expected %q", indexUsed, "f:spec.nodeName")
	}

	// list with index not exists.
	matchValues = []storage.MatchValue{
		{IndexName: "l:not-exist-label", Value: "whatever"},
	}
	list, resourceVersion, indexUsed, err = store.WaitUntilFreshAndList(5, matchValues, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resourceVersion != 5 {
		t.Errorf("unexpected resourceVersion: %v, expected: 5", resourceVersion)
	}
	if len(list) != 3 {
		t.Errorf("unexpected list returned: %#v", list)
	}
	if indexUsed != "" {
		t.Errorf("Used index %q but expected none to be used", indexUsed)
	}
}

func TestWaitUntilFreshAndGet(t *testing.T) {
	store := newTestWatchCache(3, &cache.Indexers{})

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
	expected := makeTestStoreElement(makeTestPod("bar", 5))
	if !apiequality.Semantic.DeepEqual(expected, obj) {
		t.Errorf("expected %v, got %v", expected, obj)
	}
}

func TestWaitUntilFreshAndListTimeout(t *testing.T) {
	store := newTestWatchCache(3, &cache.Indexers{})
	fc := store.clock.(*testingclock.FakeClock)

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

	_, _, _, err := store.WaitUntilFreshAndList(5, nil, nil)
	if !errors.IsTimeout(err) {
		t.Errorf("expected timeout error but got: %v", err)
	}
	if !storage.IsTooLargeResourceVersion(err) {
		t.Errorf("expected 'Too large resource version' cause in error but got: %v", err)
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
	store := newTestWatchCache(5, &cache.Indexers{})

	{
		_, version, _, err := store.WaitUntilFreshAndList(0, nil, nil)
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
		_, version, _, err := store.WaitUntilFreshAndList(10, nil, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if version != 10 {
			t.Errorf("unexpected resource version: %d", version)
		}
	}
}

func TestDynamicCache(t *testing.T) {
	tests := []struct {
		name          string
		eventCount    int
		cacheCapacity int
		startIndex    int
		// interval is time duration between adjacent events.
		lowerBoundCapacity int
		upperBoundCapacity int
		interval           time.Duration
		expectCapacity     int
		expectStartIndex   int
	}{
		{
			name:               "[capacity not equals 4*n] events inside eventFreshDuration cause cache expanding",
			eventCount:         5,
			cacheCapacity:      5,
			lowerBoundCapacity: 5 / 2,
			upperBoundCapacity: 5 * 2,
			interval:           eventFreshDuration / 6,
			expectCapacity:     10,
			expectStartIndex:   0,
		},
		{
			name:               "[capacity not equals 4*n] events outside eventFreshDuration without change cache capacity",
			eventCount:         5,
			cacheCapacity:      5,
			lowerBoundCapacity: 5 / 2,
			upperBoundCapacity: 5 * 2,
			interval:           eventFreshDuration / 4,
			expectCapacity:     5,
			expectStartIndex:   0,
		},
		{
			name:               "[capacity not equals 4*n] quarter of recent events outside eventFreshDuration cause cache shrinking",
			eventCount:         5,
			cacheCapacity:      5,
			lowerBoundCapacity: 5 / 2,
			upperBoundCapacity: 5 * 2,
			interval:           eventFreshDuration + time.Second,
			expectCapacity:     2,
			expectStartIndex:   3,
		},
		{
			name:               "[capacity not equals 4*n] quarter of recent events outside eventFreshDuration cause cache shrinking with given lowerBoundCapacity",
			eventCount:         5,
			cacheCapacity:      5,
			lowerBoundCapacity: 3,
			upperBoundCapacity: 5 * 2,
			interval:           eventFreshDuration + time.Second,
			expectCapacity:     3,
			expectStartIndex:   2,
		},
		{
			name:               "[capacity not equals 4*n] events inside eventFreshDuration cause cache expanding with given upperBoundCapacity",
			eventCount:         5,
			cacheCapacity:      5,
			lowerBoundCapacity: 5 / 2,
			upperBoundCapacity: 8,
			interval:           eventFreshDuration / 6,
			expectCapacity:     8,
			expectStartIndex:   0,
		},
		{
			name:               "[capacity not equals 4*n] [startIndex not equal 0] events inside eventFreshDuration cause cache expanding",
			eventCount:         5,
			cacheCapacity:      5,
			startIndex:         3,
			lowerBoundCapacity: 5 / 2,
			upperBoundCapacity: 5 * 2,
			interval:           eventFreshDuration / 6,
			expectCapacity:     10,
			expectStartIndex:   3,
		},
		{
			name:               "[capacity not equals 4*n] [startIndex not equal 0] events outside eventFreshDuration without change cache capacity",
			eventCount:         5,
			cacheCapacity:      5,
			startIndex:         3,
			lowerBoundCapacity: 5 / 2,
			upperBoundCapacity: 5 * 2,
			interval:           eventFreshDuration / 4,
			expectCapacity:     5,
			expectStartIndex:   3,
		},
		{
			name:               "[capacity not equals 4*n] [startIndex not equal 0] quarter of recent events outside eventFreshDuration cause cache shrinking",
			eventCount:         5,
			cacheCapacity:      5,
			startIndex:         3,
			lowerBoundCapacity: 5 / 2,
			upperBoundCapacity: 5 * 2,
			interval:           eventFreshDuration + time.Second,
			expectCapacity:     2,
			expectStartIndex:   6,
		},
		{
			name:               "[capacity not equals 4*n] [startIndex not equal 0] quarter of recent events outside eventFreshDuration cause cache shrinking with given lowerBoundCapacity",
			eventCount:         5,
			cacheCapacity:      5,
			startIndex:         3,
			lowerBoundCapacity: 3,
			upperBoundCapacity: 5 * 2,
			interval:           eventFreshDuration + time.Second,
			expectCapacity:     3,
			expectStartIndex:   5,
		},
		{
			name:               "[capacity not equals 4*n] [startIndex not equal 0] events inside eventFreshDuration cause cache expanding with given upperBoundCapacity",
			eventCount:         5,
			cacheCapacity:      5,
			startIndex:         3,
			lowerBoundCapacity: 5 / 2,
			upperBoundCapacity: 8,
			interval:           eventFreshDuration / 6,
			expectCapacity:     8,
			expectStartIndex:   3,
		},
		{
			name:               "[capacity equals 4*n] events inside eventFreshDuration cause cache expanding",
			eventCount:         8,
			cacheCapacity:      8,
			lowerBoundCapacity: 8 / 2,
			upperBoundCapacity: 8 * 2,
			interval:           eventFreshDuration / 9,
			expectCapacity:     16,
			expectStartIndex:   0,
		},
		{
			name:               "[capacity equals 4*n] events outside eventFreshDuration without change cache capacity",
			eventCount:         8,
			cacheCapacity:      8,
			lowerBoundCapacity: 8 / 2,
			upperBoundCapacity: 8 * 2,
			interval:           eventFreshDuration / 8,
			expectCapacity:     8,
			expectStartIndex:   0,
		},
		{
			name:               "[capacity equals 4*n] quarter of recent events outside eventFreshDuration cause cache shrinking",
			eventCount:         8,
			cacheCapacity:      8,
			lowerBoundCapacity: 8 / 2,
			upperBoundCapacity: 8 * 2,
			interval:           eventFreshDuration/2 + time.Second,
			expectCapacity:     4,
			expectStartIndex:   4,
		},
		{
			name:               "[capacity equals 4*n] quarter of recent events outside eventFreshDuration cause cache shrinking with given lowerBoundCapacity",
			eventCount:         8,
			cacheCapacity:      8,
			lowerBoundCapacity: 7,
			upperBoundCapacity: 8 * 2,
			interval:           eventFreshDuration/2 + time.Second,
			expectCapacity:     7,
			expectStartIndex:   1,
		},
		{
			name:               "[capacity equals 4*n] events inside eventFreshDuration cause cache expanding with given upperBoundCapacity",
			eventCount:         8,
			cacheCapacity:      8,
			lowerBoundCapacity: 8 / 2,
			upperBoundCapacity: 10,
			interval:           eventFreshDuration / 9,
			expectCapacity:     10,
			expectStartIndex:   0,
		},
		{
			name:               "[capacity equals 4*n] [startIndex not equal 0] events inside eventFreshDuration cause cache expanding",
			eventCount:         8,
			cacheCapacity:      8,
			startIndex:         3,
			lowerBoundCapacity: 8 / 2,
			upperBoundCapacity: 8 * 2,
			interval:           eventFreshDuration / 9,
			expectCapacity:     16,
			expectStartIndex:   3,
		},
		{
			name:               "[capacity equals 4*n] [startIndex not equal 0] events outside eventFreshDuration without change cache capacity",
			eventCount:         8,
			cacheCapacity:      8,
			startIndex:         3,
			lowerBoundCapacity: 8 / 2,
			upperBoundCapacity: 8 * 2,
			interval:           eventFreshDuration / 8,
			expectCapacity:     8,
			expectStartIndex:   3,
		},
		{
			name:               "[capacity equals 4*n] [startIndex not equal 0] quarter of recent events outside eventFreshDuration cause cache shrinking",
			eventCount:         8,
			cacheCapacity:      8,
			startIndex:         3,
			lowerBoundCapacity: 8 / 2,
			upperBoundCapacity: 8 * 2,
			interval:           eventFreshDuration/2 + time.Second,
			expectCapacity:     4,
			expectStartIndex:   7,
		},
		{
			name:               "[capacity equals 4*n] [startIndex not equal 0] quarter of recent events outside eventFreshDuration cause cache shrinking with given lowerBoundCapacity",
			eventCount:         8,
			cacheCapacity:      8,
			startIndex:         3,
			lowerBoundCapacity: 7,
			upperBoundCapacity: 8 * 2,
			interval:           eventFreshDuration/2 + time.Second,
			expectCapacity:     7,
			expectStartIndex:   4,
		},
		{
			name:               "[capacity equals 4*n] [startIndex not equal 0] events inside eventFreshDuration cause cache expanding with given upperBoundCapacity",
			eventCount:         8,
			cacheCapacity:      8,
			startIndex:         3,
			lowerBoundCapacity: 8 / 2,
			upperBoundCapacity: 10,
			interval:           eventFreshDuration / 9,
			expectCapacity:     10,
			expectStartIndex:   3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := newTestWatchCache(test.cacheCapacity, &cache.Indexers{})
			store.cache = make([]*watchCacheEvent, test.cacheCapacity)
			store.startIndex = test.startIndex
			store.lowerBoundCapacity = test.lowerBoundCapacity
			store.upperBoundCapacity = test.upperBoundCapacity
			loadEventWithDuration(store, test.eventCount, test.interval)
			nextInterval := store.clock.Now().Add(time.Duration(test.interval.Nanoseconds() * int64(test.eventCount)))
			store.resizeCacheLocked(nextInterval)
			if store.capacity != test.expectCapacity {
				t.Errorf("expect capacity %d, but get %d", test.expectCapacity, store.capacity)
			}

			// check cache's startIndex, endIndex and all elements.
			if store.startIndex != test.expectStartIndex {
				t.Errorf("expect startIndex %d, but get %d", test.expectStartIndex, store.startIndex)
			}
			if store.endIndex != test.startIndex+test.eventCount {
				t.Errorf("expect endIndex %d get %d", test.startIndex+test.eventCount, store.endIndex)
			}
			if !checkCacheElements(store) {
				t.Errorf("some elements locations in cache is wrong")
			}
		})
	}
}

func loadEventWithDuration(cache *testWatchCache, count int, interval time.Duration) {
	for i := 0; i < count; i++ {
		event := &watchCacheEvent{
			Key:        fmt.Sprintf("event-%d", i+cache.startIndex),
			RecordTime: cache.clock.Now().Add(time.Duration(interval.Nanoseconds() * int64(i))),
		}
		cache.cache[(i+cache.startIndex)%cache.capacity] = event
	}
	cache.endIndex = cache.startIndex + count
}

func checkCacheElements(cache *testWatchCache) bool {
	for i := cache.startIndex; i < cache.endIndex; i++ {
		location := i % cache.capacity
		if cache.cache[location].Key != fmt.Sprintf("event-%d", i) {
			return false
		}
	}
	return true
}

func TestCacheIncreaseDoesNotBreakWatch(t *testing.T) {
	store := newTestWatchCache(2, &cache.Indexers{})

	now := store.clock.Now()
	addEvent := func(key string, rv uint64, t time.Time) {
		event := &watchCacheEvent{
			Key:             key,
			ResourceVersion: rv,
			RecordTime:      t,
		}
		store.updateCache(event)
	}

	// Initial LIST comes from the moment of RV=10.
	store.Replace(nil, "10")

	addEvent("key1", 20, now)

	// Force "key1" to rotate our of cache.
	later := now.Add(2 * eventFreshDuration)
	addEvent("key2", 30, later)
	addEvent("key3", 40, later)

	// Force cache resize.
	addEvent("key4", 50, later.Add(time.Second))

	_, err := store.getAllEventsSince(15)
	if err == nil || !strings.Contains(err.Error(), "too old resource version") {
		t.Errorf("unexpected error: %v", err)
	}
}

func BenchmarkWatchCache_updateCache(b *testing.B) {
	store := newTestWatchCache(defaultUpperBoundCapacity, &cache.Indexers{})
	store.cache = store.cache[:0]
	store.upperBoundCapacity = defaultUpperBoundCapacity
	loadEventWithDuration(store, defaultUpperBoundCapacity, 0)
	add := &watchCacheEvent{
		Key:        fmt.Sprintf("event-%d", defaultUpperBoundCapacity),
		RecordTime: store.clock.Now(),
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.updateCache(add)
	}
}
