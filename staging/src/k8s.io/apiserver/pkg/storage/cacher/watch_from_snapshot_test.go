/*
Copyright The Kubernetes Authors.

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
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/clock"

	example "k8s.io/apiserver/pkg/apis/example"
)

// This test basically covers the happy path of watch-list, receiving Initial events i.e; SendInitialEvents=True
// it is getting all synthetic ADD events, once those are done, it gets a bookmark saying the initial part of watch-list is finished,
// so it can transition to a regular watch which listens to changes as they come in as those gets dispatched to the channel
func TestTryWatchFromSnapshot_HappyPath(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)
	forceRequestWatchProgressSupport(t)

	backingStorage := &cachertesting.MockStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Add pods to the cache to populate the store and snapshots.
	pods := make([]*example.Pod, 5)
	for i := range pods {
		pods[i] = &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 100+i),
			},
		}
		if err := cacher.watchCache.Add(pods[i]); err != nil {
			t.Fatalf("failed to add pod-%d: %v", i, err)
		}
	}

	// Verify snapshots are available (processEvent creates them).
	if cacher.watchCache.snapshots == nil {
		t.Fatal("snapshots should not be nil")
	}
	if !cacher.watchCache.snapshottingEnabled.Load() {
		t.Fatal("snapshotting should be enabled")
	}

	// Issue a WatchList (SendInitialEvents=true).
	trueVal := true
	pred := storage.Everything
	pred.AllowWatchBookmarks = true
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// setting up a watch-list here to watch pods from the store that was set above
	// this should read first part of the watch-list i.e; from the snapshot as its enabled
	w, err := cacher.Watch(ctx, "/pods/ns", storage.ListOptions{
		ResourceVersion:   "100",
		SendInitialEvents: &trueVal,
		Predicate:         pred,
	})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w.Stop()

	// Collect initial events, should be all 5 pods as Added.
	var receivedEvents []watch.Event
	timeout := time.After(5 * time.Second)
	for range 5 {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("watch channel closed unexpectedly")
			}
			receivedEvents = append(receivedEvents, event)
		case <-timeout:
			t.Fatalf("timed out, got %d events", len(receivedEvents))
		}
	}

	// Verify all are Added events , because initial events are synthethic events which always will be as ADD events
	for i, event := range receivedEvents {
		if event.Type != watch.Added {
			t.Errorf("event %d: expected Added, got %v", i, event.Type)
		}
	}

	// Expect a bookmark event (InitialEventsEnd). This is how watch-list clients know, synthentic events are synced, so the actual watch kicks in
	select {
	case event, ok := <-w.ResultChan():
		if !ok {
			t.Fatal("watch channel closed unexpectedly")
		}
		if event.Type != watch.Bookmark {
			t.Errorf("expected Bookmark event, got %v", event.Type)
		}
	case <-timeout:
		t.Fatal("timed out waiting for bookmark event")
	}

	// Now add a new pod, should arrive as a live event because initial events are already synced, new pod means new watch-event
	newPod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "pod-new",
			Namespace:       "ns",
			ResourceVersion: "200",
		},
	}
	// new pod is being added to watchcache, which means this will internall call processEvents as how in real code we use it when it comes from etcd
	if err := cacher.watchCache.Add(newPod); err != nil {
		t.Fatalf("failed to add new pod: %v", err)
	}
	// this select is checking if receiver listening on those pod events has seen this new event that was added after initial events were synced, this should come from dispatch part, not from snapshot
	select {
	case event, ok := <-w.ResultChan():
		if !ok {
			t.Fatal("watch channel closed unexpectedly")
		}
		if event.Type != watch.Added {
			t.Errorf("expected Added for live event, got %v", event.Type)
		}
		accessor, err := meta.Accessor(event.Object)
		if err != nil {
			t.Fatalf("failed to get accessor: %v", err)
		}
		if accessor.GetName() != "pod-new" {
			t.Errorf("expected pod-new, got %s", accessor.GetName())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for live event")
	}
}

// This test basically should fallback to current approach when snapshots are unavailable,
// and watch-list should still work as expected, i.e; it gets events from store, and construct the events under the w.rlock
func TestTryWatchFromSnapshot_FallbackWhenSnapshotUnavailable(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)
	forceRequestWatchProgressSupport(t)

	backingStorage := &cachertesting.MockStorage{}
	fakeClock := clock.RealClock{}
	cacher, _, err := newTestCacherWithoutSyncing(backingStorage, fakeClock)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := cacher.ready.wait(ctx); err != nil {
		t.Fatalf("error waiting for the cache to be ready: %v", err)
	}

	// Disable snapshotting to force fallback, because new approach specifically requires snapshots to be enabled to take advantage to serving the initial events for watch-list from snapshot reference
	cacher.watchCache.snapshottingEnabled.Store(false)

	// Add 3 pods to the watch-cache, this is later used to read them as initial ADD events
	for i := range 3 {
		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 100+i),
			},
		}
		if err := cacher.watchCache.Add(pod); err != nil {
			t.Fatalf("failed to add pod-%d: %v", i, err)
		}
	}

	// Issue WatchList — should fall back to original path since snapshotting is disabled, so initial synthetic events will be contrsuted from store and sent
	trueVal := true
	pred := storage.Everything
	pred.AllowWatchBookmarks = true
	w, err := cacher.Watch(ctx, "/pods/ns", storage.ListOptions{
		ResourceVersion:   "100",
		SendInitialEvents: &trueVal,
		Predicate:         pred,
	})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w.Stop()

	// Should still receive all 3 pods (via fallback path) , this one is basically ensuring old way of watch-list for initial events code path still works
	for range 3 {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("watch channel closed unexpectedly")
			}
			if event.Type != watch.Added {
				t.Errorf("expected Added, got %v", event.Type)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for event")
		}
	}
}

// This test is for like watching a single pod, which is already O(1), there is no need for it go through optimization code path
func TestTryWatchFromSnapshot_SingleKeyFallback(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)
	forceRequestWatchProgressSupport(t)

	backingStorage := &cachertesting.MockStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	pod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "specific-pod",
			Namespace:       "ns",
			ResourceVersion: "100",
		},
	}
	// add single pod to watch-cache
	if err := cacher.watchCache.Add(pod); err != nil {
		t.Fatalf("failed to add pod: %v", err)
	}

	// Watch a specific pod by field selector — triggers matchesSingle=true.
	trueVal := true
	pred := storage.Everything
	pred.AllowWatchBookmarks = true
	pred.Field = fields.OneTermEqualSelector("metadata.name", "specific-pod")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// establishing a watch using selector predicate, so its watching a specific pod instead a list of pods
	w, err := cacher.Watch(ctx, "/pods/ns/specific-pod", storage.ListOptions{
		// given new pod above is create with RV=100, it is forcing new pod delivers as new event, because watch is set with rv 99
		ResourceVersion:   "99",
		SendInitialEvents: &trueVal,
		Predicate:         pred,
	})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w.Stop()

	// given new code optimization path explicitly checks if it matches single key, returns false to force it to go back to old code path
	select {
	case event, ok := <-w.ResultChan():
		if !ok {
			t.Fatal("watch channel closed unexpectedly")
		}
		if event.Type != watch.Added {
			t.Errorf("expected Added, got %v", event.Type)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for event")
	}
}

// This test basically for regular LIST+ watch semantics, to ensure it uses the old code path,
// given LIST+Watch semantic doesn't enter into the non-optimized path, no reason to create snapshot reference under watch-cache read lock nor require sending those initial events for client informer to sync as it happens through LIST,
// The old code path already takes care of registering watch to subscribe to new events as they come, so no reason to enter new path
// this test ensures old path works without an issue
func TestTryWatchFromSnapshot_NonWatchListFallback(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)
	forceRequestWatchProgressSupport(t)

	backingStorage := &cachertesting.MockStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// adding 3 pods to wathc-cache store
	for i := range 3 {
		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 100+i),
			},
		}
		if err := cacher.watchCache.Add(pod); err != nil {
			t.Fatalf("failed to add pod-%d: %v", i, err)
		}
	}

	// Watch with SendInitialEvents=false uses original code path
	falseVal := false
	pred := storage.Everything
	pred.AllowWatchBookmarks = true
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// given it is setting RV=102, which mean it already knows until 102
	w, err := cacher.Watch(ctx, "/pods/ns", storage.ListOptions{
		ResourceVersion:   "102",
		SendInitialEvents: &falseVal,
		Predicate:         pred,
	})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w.Stop()

	// With SendInitialEvents=false, no initial events should be delivered.
	// Add a new pod to get a live event.
	newPod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "pod-live",
			Namespace:       "ns",
			ResourceVersion: "200",
		},
	}
	// adding new pod to watch-cache
	if err := cacher.watchCache.Add(newPod); err != nil {
		t.Fatalf("failed to add pod: %v", err)
	}

	select {
	case event, ok := <-w.ResultChan():
		if !ok {
			t.Fatal("watch channel closed unexpectedly")
		}
		if event.Type != watch.Added {
			t.Errorf("expected Added, got %v", event.Type)
		}
		accessor, err := meta.Accessor(event.Object)
		if err != nil {
			t.Fatalf("failed to get accessor: %v", err)
		}
		// ensuring we are getting the new pod with rv=200 is delivered as new watch event
		if accessor.GetName() != "pod-live" {
			t.Errorf("expected pod-live, got %s", accessor.GetName())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for live event")
	}
}

// This test verifies that no events are lost from the time snapshot reference is captured and watcher is registered
// We want a test to ensure no new events are processed from etcd into watch-cache, and dispatched to watcher, after snapshot is taken and before watcher is registered to ensure client doesn't loose events
// In the new proposal, given we have Rlock on watch-cache, writers cannot process in the meantime, so no new events should arrive until we finish watcher registration,
// And to register watcher we acquire c.lock  under watch-cache lock.
func TestTryWatchFromSnapshot_NoEventGap(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)
	forceRequestWatchProgressSupport(t)

	backingStorage := &cachertesting.MockStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Add 3 pods to watch-cache
	for i := range 3 {
		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 100+i),
			},
		}
		if err := cacher.watchCache.Add(pod); err != nil {
			t.Fatalf("failed to add pod-%d: %v", i, err)
		}
	}

	// Start WatchList.
	trueVal := true
	pred := storage.Everything
	pred.AllowWatchBookmarks = true
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// establishing watcch list  with sendInitialevents True
	w, err := cacher.Watch(ctx, "/pods/ns", storage.ListOptions{
		ResourceVersion:   "100",
		SendInitialEvents: &trueVal,
		Predicate:         pred,
	})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w.Stop()

	// add new pods immediately i.e; to ensure if watch-cache advances if it has write lock, due to process events from etcd in real world
	for i := 3; i < 6; i++ {
		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 200+i),
			},
		}
		if err := cacher.watchCache.Add(pod); err != nil {
			t.Fatalf("failed to add pod-%d: %v", i, err)
		}
	}

	// Collect all events: 3 initial + bookmark + 3 live = 7 total.
	// which should deliver as new events, ensuring that events are not lost while above after watcher is registered and reads initial events from snapshot
	var allEvents []watch.Event
	timeout := time.After(10 * time.Second)
	for range 7 {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatalf("watch channel closed after %d events", len(allEvents))
			}
			allEvents = append(allEvents, event)
		case <-timeout:
			t.Fatalf("timed out after receiving %d events, expected 7", len(allEvents))
		}
	}

	// First 3 should be Added (initial from snapshot).
	for i := range 3 {
		if allEvents[i].Type != watch.Added {
			t.Errorf("event %d: expected Added, got %v", i, allEvents[i].Type)
		}
	}

	// Event 4 should be Bookmark (InitialEventsEnd).
	if allEvents[3].Type != watch.Bookmark {
		t.Errorf("event 3: expected Bookmark, got %v", allEvents[3].Type)
	}

	// Events 5-7 should be Added (live events via c.input), that is from the dispatch events part
	for i := 4; i < 7; i++ {
		if allEvents[i].Type != watch.Added {
			t.Errorf("event %d: expected Added, got %v", i, allEvents[i].Type)
		}
	}
}
