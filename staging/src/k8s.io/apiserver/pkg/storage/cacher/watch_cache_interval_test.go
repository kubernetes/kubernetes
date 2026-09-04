/*
Copyright 2021 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"iter"
	"reflect"
	"sort"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/cacher/store"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func intervalFromEvents(events []*watchCacheEvent) *watchCacheInterval {
	startIndex, endIndex, locker := 0, len(events), &sync.Mutex{}
	indexer := func(i int) *watchCacheEvent {
		if len(events) == 0 {
			return nil
		}
		return events[i]
	}
	indexValidator := func(_ int) bool { return true }

	return newCacheInterval(startIndex, endIndex, indexer, indexValidator, 0, locker)
}

func historySource(wci *watchCacheInterval) *historyCacheIntervalSource {
	return wci.source.(*historyCacheIntervalSource)
}

func bufferFromEvents(events []*watchCacheEvent) *watchCacheIntervalBuffer {
	wcib := &watchCacheIntervalBuffer{
		buffer:     make([]*watchCacheEvent, bufferSize),
		startIndex: 0,
		endIndex:   len(events),
	}
	copy(wcib.buffer, events)

	return wcib
}

func generateEvents(start, end int) []*watchCacheEvent {
	n := end - start
	events := make([]*watchCacheEvent, n)
	for i := 0; i < n; i++ {
		events[i] = &watchCacheEvent{
			Type:   watch.Added,
			Object: makeTestPod(fmt.Sprintf("pod%d", start+i), uint64(start+i)),
		}
	}
	return events
}

func verifyEvent(ok bool, event, expectedEvent *watchCacheEvent) error {
	if !ok {
		return fmt.Errorf("expected event: %#v, got no event", expectedEvent)
	}

	if event == nil {
		return fmt.Errorf("unexpected nil event, expected: %#v", expectedEvent)
	}

	if !reflect.DeepEqual(event, expectedEvent) {
		return fmt.Errorf("expected %v, got %v", *event, *expectedEvent)
	}

	return nil
}

func verifyNoEvent(ok bool, event *watchCacheEvent) error {
	if ok {
		return errors.New("unexpected bool value indicating buffer is not empty")
	}
	if event != nil {
		return fmt.Errorf("unexpected event received, expected: nil, got %v", *event)
	}

	return nil
}

func TestIntervalBufferIsFull(t *testing.T) {
	cases := []struct {
		endIndex int
		expected bool
	}{
		{endIndex: bufferSize - 1, expected: false},
		{endIndex: bufferSize, expected: true},
		{endIndex: bufferSize + 1, expected: true},
	}

	for _, c := range cases {
		wcib := &watchCacheIntervalBuffer{endIndex: c.endIndex}
		actual := wcib.isFull()
		if actual != c.expected {
			t.Errorf("expected %v, got %v", c.expected, actual)
		}
	}
}

func TestIntervalBufferIsEmpty(t *testing.T) {
	cases := []struct {
		startIndex int
		endIndex   int
		expected   bool
	}{
		{startIndex: 0, endIndex: 10, expected: false},
		{startIndex: 5, endIndex: 20, expected: false},
		{startIndex: 50, endIndex: 50, expected: true},
	}

	for _, c := range cases {
		wcib := &watchCacheIntervalBuffer{
			startIndex: c.startIndex,
			endIndex:   c.endIndex,
		}
		actual := wcib.isEmpty()
		if actual != c.expected {
			t.Errorf("expected %v, got %v", c.expected, actual)
		}
	}
}

func TestIntervalBufferNext(t *testing.T) {
	cases := []struct {
		name   string
		events []*watchCacheEvent
	}{
		{
			name: "buffer has elements",
			events: []*watchCacheEvent{
				{Type: watch.Added, Object: makeTestPod("pod1", 1)},
				{Type: watch.Added, Object: makeTestPod("pod2", 2)},
				{Type: watch.Modified, Object: makeTestPod("pod3", 3)},
			},
		},
		{
			name:   "buffer is empty",
			events: []*watchCacheEvent{},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			wcib := bufferFromEvents(c.events)
			for i := 0; i < len(c.events); i++ {
				event, ok := wcib.next()
				if err := verifyEvent(ok, event, c.events[i]); err != nil {
					t.Error(err)
				}
			}
			event, ok := wcib.next()
			if err := verifyNoEvent(ok, event); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestFillBuffer(t *testing.T) {
	cases := []struct {
		name            string
		numEventsToFill int
	}{
		{
			name:            "no events to put in buffer",
			numEventsToFill: 0,
		},
		{
			name:            "less than bufferSize events to put in buffer",
			numEventsToFill: 5,
		},
		{
			name:            "equal to bufferSize events to put in buffer",
			numEventsToFill: bufferSize,
		},
		{
			name:            "greater than bufferSize events to put in buffer",
			numEventsToFill: bufferSize + 5,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			events := generateEvents(0, c.numEventsToFill)
			wci := intervalFromEvents(events)
			src := historySource(wci)

			for i := 0; i < len(events); i++ {
				if i%bufferSize == 0 {
					src.fillBuffer()
				}
				event, ok := src.buffer.next()
				if err := verifyEvent(ok, event, events[i]); err != nil {
					t.Error(err)
				}
				// If we have already received bufferSize number of events,
				// buffer should be empty and we should receive no event.
				if i%bufferSize == bufferSize-1 {
					event, ok := src.buffer.next()
					if err := verifyNoEvent(ok, event); err != nil {
						t.Error(err)
					}
				}
			}
			// buffer should be empty and return no event.
			event, ok := src.buffer.next()
			if err := verifyNoEvent(ok, event); err != nil {
				t.Error(err)
			}
			// Buffer should be empty now, an additional fillBuffer()
			// should make no difference.
			src.fillBuffer()
			event, ok = src.buffer.next()
			if err := verifyNoEvent(ok, event); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestCacheIntervalNextFromWatchCache(t *testing.T) {
	// Have the capacity such that it facilitates
	// filling the interval buffer more than once
	// completely and then some more - 10 here is
	// arbitrary.
	const capacity = 2*bufferSize + 10

	cases := []struct {
		name string
		// The total number of events that the watch
		// cache will be populated with to start with.
		eventsAddedToWatchcache int
		intervalStartIndex      int
	}{
		{
			name:                    "watchCache empty, eventsAddedToWatchcache = 0",
			eventsAddedToWatchcache: 0,
			intervalStartIndex:      0,
		},
		{
			name:                    "watchCache partially propagated, eventsAddedToWatchcache < capacity",
			eventsAddedToWatchcache: bufferSize,
			intervalStartIndex:      0,
		},
		{
			name:                    "watchCache partially propagated, eventsAddedToWatchcache < capacity, intervalStartIndex at some offset",
			eventsAddedToWatchcache: bufferSize,
			intervalStartIndex:      5,
		},
		{
			name:                    "watchCache fully propagated, eventsAddedToWatchcache = capacity",
			eventsAddedToWatchcache: capacity,
			intervalStartIndex:      0,
		},
		{
			name:                    "watchCache fully propagated, eventsAddedToWatchcache = capacity, intervalStartIndex at some offset",
			eventsAddedToWatchcache: capacity,
			intervalStartIndex:      5,
		},
		{
			name:                    "watchCache over propagated, eventsAddedToWatchcache > capacity",
			eventsAddedToWatchcache: capacity + bufferSize,
			intervalStartIndex:      0,
		},
		{
			name:                    "watchCache over propagated, eventsAddedToWatchcache > capacity, intervalStartIndex at some offset",
			eventsAddedToWatchcache: capacity + bufferSize,
			intervalStartIndex:      5,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			wc := newTestWatchCache(capacity, DefaultEventFreshDuration, &cache.Indexers{})
			defer wc.Stop()
			for i := 0; i < c.eventsAddedToWatchcache; i++ {
				wc.Add(makeTestPod(fmt.Sprintf("pod%d", i), uint64(i)))
			}
			indexerFunc := func(i int) *watchCacheEvent {
				return wc.history.cache[i%wc.history.capacity]
			}

			wci := newCacheInterval(
				c.intervalStartIndex,
				wc.history.endIndex,
				indexerFunc,
				wc.history.isIndexValidLocked,
				wc.resourceVersion,
				&wc.RWMutex,
			)
			src := historySource(wci)

			numExpectedEvents := wc.history.endIndex - c.intervalStartIndex
			for i := 0; i < numExpectedEvents; i++ {
				// Simulate and test interval invalidation iff
				// the watchCache itself is not empty.
				if c.eventsAddedToWatchcache > 0 {
					// The points at which we want to artificially
					// invalidate the interval and test its behaviour
					// should be multiples of bufferSize. This is because
					// invalidation only needs to be checked when we are
					// copying over events from the underlying watch cache,
					// i.e. freshly filling in the interval buffer.
					if i%bufferSize == 0 && i != c.eventsAddedToWatchcache {
						originalCacheStartIndex := wc.history.startIndex
						wc.history.startIndex = src.startIndex + 1
						event, err := wci.Next()
						if err == nil {
							t.Errorf("expected non-nil error")
						}
						if event != nil {
							t.Errorf("expected nil event, got %v", *event)
						}
						// Restore startIndex.
						wc.history.startIndex = originalCacheStartIndex
					}
				}

				// Check if the state of the interval buffer is as expected.
				// The interval buffer can be empty either when received is
				// either a multiple of bufferSize (after one complete fill)
				// or when received is equal to the number of expected events.
				// The latter happens when partial filling occurs and no more
				// events are left post the partial fill.
				if src.buffer.isEmpty() != (i%bufferSize == 0 || i == numExpectedEvents) {
					t.Error("expected empty interval buffer")
					return
				}

				event, err := wci.Next()
				if err != nil {
					t.Errorf("unexpected error: %v", err)
					return
				}

				expectedIndex := (c.intervalStartIndex + i) % wc.history.capacity
				expectedEvent := wc.history.cache[expectedIndex]
				if err := verifyEvent(true, event, expectedEvent); err != nil {
					t.Error(err)
				}
			}
			event, err := wci.Next()
			ok := err != nil
			if err := verifyNoEvent(ok, event); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestCacheIntervalNextFromSnapshot(t *testing.T) {
	getAttrsFunc := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			return nil, nil, fmt.Errorf("not a pod")
		}
		return labels.Set(pod.Labels), fields.Set{"spec.nodeName": pod.Spec.NodeName}, nil
	}
	const numEvents = 2*bufferSize + 50
	indexer := store.NewIndexer(nil)
	events := make(map[string]*watchCacheEvent)
	var rv uint64 = 1 // arbitrary number; rv till which the watch cache has progressed.

	for i := 0; i < numEvents; i++ {
		elem := makeTestStoreElement(makeTestPod(fmt.Sprintf("pod%d", i), uint64(i)))
		objLabels, objFields, err := getAttrsFunc(elem.Object)
		if err != nil {
			t.Fatal(err)
		}
		events[elem.Key] = &watchCacheEvent{
			Type:            watch.Added,
			Object:          elem.Object,
			ObjLabels:       objLabels,
			ObjFields:       objFields,
			Key:             elem.Key,
			ResourceVersion: rv,
		}
		indexer.Add(elem)
	}

	wci := newCacheIntervalFromSnapshot(rv, indexer.Clone())

	for i := 0; i < numEvents; i++ {
		event, err := wci.Next()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if event == nil {
			t.Error("unexpected nil event")
			break
		}
		expectedEvent, ok := events[event.Key]
		if !ok {
			t.Fatalf("event with key %s not found", event.Key)
		}
		if !reflect.DeepEqual(event, expectedEvent) {
			t.Errorf("expected: %v, got %v", *events[event.Key], *event)
		}
	}

	// All events should have been consumed.
	event, err := wci.Next()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event != nil {
		t.Errorf("expected nil event after consuming all events, got %v", *event)
	}
}

func TestCacheIntervalFromSnapshotSorted(t *testing.T) {
	cases := []struct {
		name    string
		indexer store.Indexer
	}{
		{"btree", store.NewIndexer(nil)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const n = 2*bufferSize + 50
			// Insert in reverse-key order so any code path that returns
			// items in insertion order trivially fails the sorted check below.
			for i := n - 1; i >= 0; i-- {
				key := fmt.Sprintf("pod-%08d", i)
				elem := makeTestStoreElement(makeTestPod(key, uint64(i)))
				err := tc.indexer.Add(elem)
				if err != nil {
					t.Fatal(err)
				}
			}

			wci := newCacheIntervalFromSnapshot(n, tc.indexer.Clone())

			got := make([]string, 0, n)
			for range n {
				ev, err := wci.Next()
				if err != nil {
					t.Fatal(err)
				}
				got = append(got, ev.Key)
			}
			if !sort.StringsAreSorted(got) {
				t.Errorf("events not sorted by key: %v", got)
			}
		})
	}
}

func TestGetIntervalFromStoreLocked(t *testing.T) {
	for _, snapshotting := range []bool{true, false} {
		for _, tc := range []struct {
			name          string
			key           string
			matchesSingle bool
			expectKeys    []string
		}{
			{name: "all", expectKeys: []string{"/prefix/ns/pod1", "/prefix/ns/pod2"}},
			{name: "single existing", key: "/prefix/ns/pod2", matchesSingle: true, expectKeys: []string{"/prefix/ns/pod2"}},
			{name: "single missing", key: "/prefix/ns/pod3", matchesSingle: true, expectKeys: nil},
		} {
			t.Run(fmt.Sprintf("ListFromCacheSnapshot=%v/%s", snapshotting, tc.name), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, snapshotting)
				wc := newTestWatchCache(3, DefaultEventFreshDuration, &cache.Indexers{})
				defer wc.Stop()
				for _, pod := range []*v1.Pod{makeTestPod("pod2", 100), makeTestPod("pod1", 101)} {
					if err := wc.Add(pod); err != nil {
						t.Fatal(err)
					}
				}

				wc.Lock()
				wci, err := wc.getIntervalFromStoreLocked(tc.key, tc.matchesSingle)
				wc.Unlock()
				if err != nil {
					t.Fatal(err)
				}
				// The interval must not observe writes made after it was taken.
				if err := wc.Add(makeTestPod("pod0", 102)); err != nil {
					t.Fatal(err)
				}

				var keys []string
				for {
					event, err := wci.Next()
					if err != nil {
						t.Fatal(err)
					}
					if event == nil {
						break
					}
					if event.ResourceVersion != 101 {
						t.Errorf("event for %s has resourceVersion %d, want 101", event.Key, event.ResourceVersion)
					}
					keys = append(keys, event.Key)
				}
				if !reflect.DeepEqual(keys, tc.expectKeys) {
					t.Errorf("got keys %v, want %v", keys, tc.expectKeys)
				}
			})
		}
	}
}

type countingSnapshot struct {
	items            []interface{}
	rangePrefixCalls int
}

func (s *countingSnapshot) GetByKey(string) (interface{}, bool, error) {
	return nil, false, nil
}

func (s *countingSnapshot) RangePrefix(_, continueKey string) iter.Seq2[*store.Element, error] {
	s.rangePrefixCalls++
	return func(yield func(*store.Element, error) bool) {
		for _, item := range s.items {
			if elem := item.(*store.Element); elem.Key >= continueKey && !yield(elem, nil) {
				return
			}
		}
	}
}

func (s *countingSnapshot) Count(_, _ string) int {
	return len(s.items)
}

func TestSnapshotCacheIntervalSourceEmpty(t *testing.T) {
	snap := &countingSnapshot{}
	wci := newCacheIntervalFromSnapshot(100, snap)

	for range 2 {
		event, err := wci.Next()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if event != nil {
			t.Errorf("expected nil event from empty snapshot, got %v", *event)
		}
	}
	if snap.rangePrefixCalls != 1 {
		t.Errorf("expected the snapshot to be ranged once, got %d", snap.rangePrefixCalls)
	}
}
