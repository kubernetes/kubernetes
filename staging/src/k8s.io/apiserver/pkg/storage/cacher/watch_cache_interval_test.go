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
	"context"
	"errors"
	"fmt"
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
	"k8s.io/apiserver/pkg/storage"
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

func TestCacheIntervalNextFromStore(t *testing.T) {
	getAttrsFunc := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			return nil, nil, fmt.Errorf("not a pod")
		}
		return labels.Set(pod.Labels), fields.Set{"spec.nodeName": pod.Spec.NodeName}, nil
	}
	const numEvents = 50
	store := store.NewWatchCacheStorage(nil, nil)
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
		store.Add(elem)
	}

	wci, err := newCacheIntervalFromStore(rv, store, "", false)
	if err != nil {
		t.Fatal(err)
	}

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

// TestCacheIntervalFromStoreSorted verifies intervals built from WatchCacheStorage
// (btree, ordered list, and index snapshots) return events sorted by Key.
func TestCacheIntervalFromStoreSorted(t *testing.T) {
	nodeIndexers := &cache.Indexers{
		"f:spec.nodeName": func(obj interface{}) ([]string, error) {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				return nil, fmt.Errorf("not a pod %#v", obj)
			}
			return []string{pod.Spec.NodeName}, nil
		},
	}
	cases := []struct {
		name         string
		indexer      *store.WatchCacheStorage
		makeInterval func(t *testing.T, rv uint64, s *store.WatchCacheStorage) (*watchCacheInterval, error)
	}{
		{
			name:    "btree",
			indexer: store.NewWatchCacheStorage(nil, nil),
			makeInterval: func(_ *testing.T, rv uint64, s *store.WatchCacheStorage) (*watchCacheInterval, error) {
				return newCacheIntervalFromStore(rv, s, "", false)
			},
		},
		{
			name:    "btree lazy snapshot",
			indexer: store.NewWatchCacheStorage(nil, nil),
			makeInterval: func(t *testing.T, rv uint64, s *store.WatchCacheStorage) (*watchCacheInterval, error) {
				snap, ok := s.LatestSnapshotLocked()
				if !ok {
					t.Fatal("expected LatestSnapshotLocked to succeed")
				}
				return newCacheIntervalFromLazySnapshot(rv, snap, ""), nil
			},
		},
		{
			name:    "ordered list lazy snapshot",
			indexer: store.NewWatchCacheStorage(nil, nil),
			makeInterval: func(t *testing.T, rv uint64, s *store.WatchCacheStorage) (*watchCacheInterval, error) {
				s.MarkConsistent(false)
				if _, ok := s.LatestSnapshotLocked(); ok {
					t.Fatal("expected LatestSnapshotLocked to be false when inconsistent")
				}
				snap, err := s.GetLatestSnapshotOrBuildLocked("", "")
				if err != nil {
					return nil, err
				}
				return newCacheIntervalFromLazySnapshot(rv, snap, ""), nil
			},
		},
		{
			name:    "indexed lazy snapshot",
			indexer: store.NewWatchCacheStorage(nil, nodeIndexers),
			makeInterval: func(_ *testing.T, rv uint64, s *store.WatchCacheStorage) (*watchCacheInterval, error) {
				snap, err := s.GetByIndexSnapshot("f:spec.nodeName", "some-node")
				if err != nil {
					return nil, err
				}
				return newCacheIntervalFromLazySnapshot(rv, snap, ""), nil
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const n = 50
			// Insert in reverse-key order with increasing resourceVersions so
			// snapshots are recorded and any code path that returns items in
			// insertion order fails the sorted check below.
			for i := n - 1; i >= 0; i-- {
				rv := uint64(n - i)
				key := fmt.Sprintf("pod-%08d", i)
				elem := makeTestStoreElement(makeTestPod(key, rv))
				_, err := tc.indexer.UpdateStoreLocked(watch.Added, elem, rv)
				if err != nil {
					t.Fatal(err)
				}
			}

			wci, err := tc.makeInterval(t, n, tc.indexer)
			if err != nil {
				t.Fatal(err)
			}

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

// TestCacheIntervalSourceSelection verifies that getIntervalFromStoreLocked builds the
// interval from the lazy snapshot source when not matching a single key (supporting
// snapshotting, index prefiltering, and key prefix scoping) and falls back to the
// eager snapshot source when matching a single key.
func TestCacheIntervalSourceSelection(t *testing.T) {
	makeNamespacedPod := func(namespace, name string, rv uint64, nodeName string) *v1.Pod {
		pod := makeTestPodDetails(name, rv, nodeName, nil)
		pod.Namespace = namespace
		return pod
	}

	cases := []struct {
		name             string
		snapshottingOn   bool
		key              string
		matchesSingle    bool
		matchValues      []storage.MatchValue
		wantLazySnapshot bool
		wantKeys         []string
	}{
		{
			name:             "snapshotting enabled serves from lazy snapshot",
			snapshottingOn:   true,
			key:              "/prefix/",
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod1", "/prefix/ns1/pod2", "/prefix/ns2/pod3"},
		},
		{
			name:             "snapshotting disabled serves from lazy snapshot",
			snapshottingOn:   false,
			key:              "/prefix/",
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod1", "/prefix/ns1/pod2", "/prefix/ns2/pod3"},
		},
		{
			name:             "snapshotting disabled with key prefix scopes lazy snapshot",
			snapshottingOn:   false,
			key:              "/prefix/ns1/",
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod1", "/prefix/ns1/pod2"},
		},
		{
			name:             "indexed matchValues serves matching items from lazy snapshot",
			snapshottingOn:   true,
			key:              "/prefix/",
			matchValues:      []storage.MatchValue{{IndexName: "f:spec.nodeName", Value: "node1"}},
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod1", "/prefix/ns2/pod3"},
		},
		{
			name:             "indexed matchValues with snapshotting disabled serves matching items from lazy snapshot",
			snapshottingOn:   false,
			key:              "/prefix/",
			matchValues:      []storage.MatchValue{{IndexName: "f:spec.nodeName", Value: "node1"}},
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod1", "/prefix/ns2/pod3"},
		},
		{
			name:             "indexed matchValues with key prefix scopes to prefix",
			snapshottingOn:   true,
			key:              "/prefix/ns1/",
			matchValues:      []storage.MatchValue{{IndexName: "f:spec.nodeName", Value: "node1"}},
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod1"},
		},
		{
			name:             "indexed matchValues with zero matching items returns empty",
			snapshottingOn:   true,
			key:              "/prefix/",
			matchValues:      []storage.MatchValue{{IndexName: "f:spec.nodeName", Value: "node-empty"}},
			wantLazySnapshot: true,
			wantKeys:         nil,
		},
		{
			name:             "non-existent index falls back to full snapshot with prefix",
			snapshottingOn:   true,
			key:              "/prefix/ns1/",
			matchValues:      []storage.MatchValue{{IndexName: "f:nonexistent", Value: "val"}},
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod1", "/prefix/ns1/pod2"},
		},
		{
			name:           "multiple matchValues skips missing index and uses matching index",
			snapshottingOn: true,
			key:            "/prefix/",
			matchValues: []storage.MatchValue{
				{IndexName: "f:nonexistent", Value: "val"},
				{IndexName: "f:spec.nodeName", Value: "node2"},
			},
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod2"},
		},
		{
			name:             "key prefix scopes lazy snapshot",
			snapshottingOn:   true,
			key:              "/prefix/ns1/",
			wantLazySnapshot: true,
			wantKeys:         []string{"/prefix/ns1/pod1", "/prefix/ns1/pod2"},
		},
		{
			name:             "single key match serves from eager snapshot",
			snapshottingOn:   true,
			key:              "/prefix/ns1/pod1",
			matchesSingle:    true,
			wantLazySnapshot: false,
			wantKeys:         []string{"/prefix/ns1/pod1"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, tc.snapshottingOn)
			wc := newTestWatchCache(3, DefaultEventFreshDuration, &cache.Indexers{
				"f:spec.nodeName": func(obj interface{}) ([]string, error) {
					pod, ok := obj.(*v1.Pod)
					if !ok {
						return nil, fmt.Errorf("not a pod %#v", obj)
					}
					return []string{pod.Spec.NodeName}, nil
				},
			})
			defer wc.Stop()
			if err := wc.Add(makeNamespacedPod("ns1", "pod1", 100, "node1")); err != nil {
				t.Fatal(err)
			}
			if err := wc.Add(makeNamespacedPod("ns1", "pod2", 101, "node2")); err != nil {
				t.Fatal(err)
			}
			if err := wc.Add(makeNamespacedPod("ns2", "pod3", 102, "node1")); err != nil {
				t.Fatal(err)
			}

			wc.Lock()
			wci, err := wc.getIntervalFromStoreLocked(context.Background(), tc.key, tc.matchesSingle, tc.matchValues)
			wc.Unlock()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var lazySrc *lazySnapshotCacheIntervalSource
			if tc.wantLazySnapshot {
				var ok bool
				lazySrc, ok = wci.source.(*lazySnapshotCacheIntervalSource)
				if !ok {
					t.Fatalf("expected *lazySnapshotCacheIntervalSource, got %T", wci.source)
				}
				if lazySrc.snapshot == nil {
					t.Errorf("expected snapshot to be set before first Next() call")
				}
			} else {
				if _, ok := wci.source.(*snapshotCacheIntervalSource); !ok {
					t.Errorf("expected *snapshotCacheIntervalSource, got %T", wci.source)
				}
			}

			var gotKeys []string
			for {
				ev, err := wci.Next()
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if ev == nil {
					break
				}
				gotKeys = append(gotKeys, ev.Key)
			}
			if !reflect.DeepEqual(gotKeys, tc.wantKeys) {
				t.Errorf("expected keys %v, got %v", tc.wantKeys, gotKeys)
			}
			if lazySrc != nil {
				if lazySrc.snapshot != nil {
					t.Errorf("expected snapshot reference to be cleared after loading")
				}
				if lazySrc.items != nil {
					t.Errorf("expected items slice to be cleared after exhaustion")
				}
			}
		})
	}
}

type countingSnapshot struct {
	items                  []interface{}
	orderedListPrefixCalls int
}

func (s *countingSnapshot) GetByKey(string) (interface{}, bool, error) {
	return nil, false, nil
}

func (s *countingSnapshot) OrderedListPrefix(_, _ string) ([]interface{}, error) {
	s.orderedListPrefixCalls++
	return s.items, nil
}

func (s *countingSnapshot) RangePrefix(_, _ string) store.Range {
	return nil
}

// TestLazySnapshotCacheIntervalSourceEmpty checks that on an empty snapshot Next() returns
// no events, and that repeated calls still read the snapshot only once.
func TestLazySnapshotCacheIntervalSourceEmpty(t *testing.T) {
	snap := &countingSnapshot{}
	wci := newCacheIntervalFromLazySnapshot(100, snap, "")

	for range 2 {
		event, err := wci.Next()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if event != nil {
			t.Errorf("expected nil event from empty snapshot, got %v", *event)
		}
	}
	if snap.orderedListPrefixCalls != 1 {
		t.Errorf("expected OrderedListPrefix to be called once, got %d", snap.orderedListPrefixCalls)
	}
}
