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
	"context"
	"fmt"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/client-go/tools/cache"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

func MakeTestPod(name string, resourceVersion uint64) *v1.Pod {
	return MakeTestPodDetails(name, resourceVersion, "some-node", map[string]string{"k8s-app": "my-app"})
}

func MakeTestPodDetails(name string, resourceVersion uint64, nodeName string, labels map[string]string) *v1.Pod {
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

type TestWatchCache struct {
	*watchCache

	BookmarkRevision chan int64
	stopCh           chan struct{}
}

func (w *TestWatchCache) getAllEventsSince(resourceVersion uint64, opts storage.ListOptions) ([]*watchCacheEvent, error) {
	cacheInterval, err := w.getCacheIntervalForEvents(resourceVersion, opts)
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

func (w *TestWatchCache) getCacheIntervalForEvents(resourceVersion uint64, opts storage.ListOptions) (*watchCacheInterval, error) {
	w.RLock()
	defer w.RUnlock()

	return w.getAllEventsSinceLocked(resourceVersion, opts)
}

func (w *TestWatchCache) FakeClock() *testingclock.FakeClock {
	return w.clock.(*testingclock.FakeClock)
}

// newTestWatchCache just adds a fake clock.
func NewTestWatchCache(capacity int, indexers *cache.Indexers) *TestWatchCache {
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
	versioner := storage.APIObjectVersioner{}
	mockHandler := func(*watchCacheEvent) {}
	wc := &TestWatchCache{}
	wc.BookmarkRevision = make(chan int64, 1)
	wc.stopCh = make(chan struct{})
	pr := newConditionalProgressRequester(wc.RequestWatchProgress, &immediateTickerFactory{}, nil)
	go pr.Run(wc.stopCh)
	wc.watchCache = newWatchCache(keyFunc, mockHandler, getAttrsFunc, versioner, indexers, testingclock.NewFakeClock(time.Now()), schema.GroupResource{Resource: "pods"}, pr)
	// To preserve behavior of tests that assume a given capacity,
	// resize it to th expected size.
	wc.capacity = capacity
	wc.cache = make([]*watchCacheEvent, capacity)
	wc.lowerBoundCapacity = min(capacity, defaultLowerBoundCapacity)
	wc.upperBoundCapacity = max(capacity, defaultUpperBoundCapacity)

	return wc
}

type immediateTickerFactory struct{}

func (t *immediateTickerFactory) NewTicker(d time.Duration) clock.Ticker {
	return &immediateTicker{stopCh: make(chan struct{})}
}

type immediateTicker struct {
	stopCh chan struct{}
}

func (t *immediateTicker) C() <-chan time.Time {
	ch := make(chan time.Time)
	go func() {
		for {
			select {
			case ch <- time.Now():
			case <-t.stopCh:
				return
			}
		}
	}()
	return ch
}

func (t *immediateTicker) Stop() {
	close(t.stopCh)
}

func (w *TestWatchCache) RequestWatchProgress(ctx context.Context) error {
	go func() {
		select {
		case rev := <-w.BookmarkRevision:
			w.UpdateResourceVersion(fmt.Sprintf("%d", rev))
		case <-ctx.Done():
			return
		}
	}()
	return nil
}

func (w *TestWatchCache) Stop() {
	close(w.stopCh)
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

func loadEventWithDuration(cache *TestWatchCache, count int, interval time.Duration) {
	for i := 0; i < count; i++ {
		event := &watchCacheEvent{
			Key:        fmt.Sprintf("event-%d", i+cache.startIndex),
			RecordTime: cache.clock.Now().Add(time.Duration(interval.Nanoseconds() * int64(i))),
		}
		cache.cache[(i+cache.startIndex)%cache.capacity] = event
	}
	cache.endIndex = cache.startIndex + count
}

func checkCacheElements(cache *TestWatchCache) bool {
	for i := cache.startIndex; i < cache.endIndex; i++ {
		location := i % cache.capacity
		if cache.cache[location].Key != fmt.Sprintf("event-%d", i) {
			return false
		}
	}
	return true
}
