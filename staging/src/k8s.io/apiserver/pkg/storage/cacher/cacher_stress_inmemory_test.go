// +build stress

/*
Copyright 2025 The Kubernetes Authors.

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
	"sync"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/client-go/tools/cache"
)

// TestSendInitialEvents_50kPods_InMemory_StressTest validates the IsInitialEvent fix
// at production scale (50k pods) without requiring etcd.
//
// This test directly exercises the watchCache store layer with aggressive concurrent
// mutations to attempt reproducing the race condition from issue #134831 where
// MODIFIED/DELETED events appear before BOOKMARK in sendInitialEvents watches.
//
// Run with: go test -tags=stress -v -timeout=1m -run TestSendInitialEvents_50kPods_InMemory
//
// Performance: ~150ms, ~100MB memory (vs 10-20min and 2GB with full etcd test)
func TestSendInitialEvents_50kPods_InMemory_StressTest(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	t.Log("Starting in-memory 50k pod stress test...")

	// Use the same setup as the real tests
	twc := newTestWatchCache(100000, time.Hour, &cache.Indexers{})
	wc := twc.watchCache

	// Populate store with 50k pods
	numPods := 50000
	t.Logf("Creating %d fake pods in store...", numPods)
	startCreate := time.Now()

	pods := make([]*example.Pod, numPods)
	for i := 0; i < numPods; i++ {
		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "default",
				ResourceVersion: fmt.Sprintf("%d", i+1),
				Labels:          map[string]string{"app": "test"},
			},
			Spec: example.PodSpec{
				NodeName: fmt.Sprintf("node-%d", i%100),
			},
		}
		pods[i] = pod

		// Add to store
		elem := &storeElement{
			Key:    fmt.Sprintf("/pods/default/pod-%d", i),
			Object: pod,
			Labels: labels.Set(pod.Labels),
			Fields: fields.Set{"metadata.name": pod.Name},
		}
		wc.store.Add(elem)

		if (i+1)%10000 == 0 {
			t.Logf("Created %d/%d pods...", i+1, numPods)
		}
	}

	t.Logf("Created %d pods in %v", numPods, time.Since(startCreate))

	// Start aggressive concurrent modifications to the SAME pod objects
	// This simulates the race where objects are mutated while iterating
	stopMutations := make(chan struct{})
	var mutWg sync.WaitGroup

	t.Log("Starting aggressive concurrent mutations...")
	for worker := 0; worker < 10; worker++ {
		mutWg.Add(1)
		go func(workerID int) {
			defer mutWg.Done()
			for {
				select {
				case <-stopMutations:
					return
				default:
					// Randomly mutate pod objects
					podIdx := (workerID * 5000) % numPods
					pod := pods[podIdx]

					// Mutate labels (this is the potential race!)
					pod.Labels["worker"] = fmt.Sprintf("%d", workerID)
					pod.Labels["mutation"] = fmt.Sprintf("%d", time.Now().UnixNano())
					pod.Spec.NodeName = fmt.Sprintf("node-%d", (podIdx+workerID)%100)
				}
			}
		}(worker)
	}

	// Give mutations time to start
	time.Sleep(50 * time.Millisecond)

	// Now create the cacheInterval (simulating sendInitialEvents snapshot)
	t.Log("Creating cacheInterval from store (snapshot)...")
	startSnapshot := time.Now()

	wc.RLock()
	cacheInterval, err := wc.getIntervalFromStoreLocked("", false)
	wc.RUnlock()

	if err != nil {
		t.Fatalf("Failed to create interval: %v", err)
	}

	t.Logf("Created interval in %v", time.Since(startSnapshot))

	// Process all events from the interval
	t.Log("Processing initial events...")
	var events []*watchCacheEvent
	eventsByType := make(map[watch.EventType]int)

	for {
		event, err := cacheInterval.Next()
		if err != nil {
			t.Fatalf("Error getting next event: %v", err)
		}
		if event == nil {
			break
		}

		events = append(events, event)
		eventsByType[event.Type]++

		if len(events)%10000 == 0 {
			t.Logf("Processed %d events...", len(events))
		}
	}

	// Stop mutations
	close(stopMutations)
	mutWg.Wait()

	// Analyze results
	t.Logf("Processed %d total initial events", len(events))
	t.Logf("Event type breakdown: %+v", eventsByType)

	// Check for violations
	violations := 0
	violationDetails := make(map[watch.EventType]int)

	for i, event := range events {
		// Verify IsInitialEvent flag is set
		if !event.IsInitialEvent {
			t.Errorf("Event #%d missing IsInitialEvent flag!", i)
		}

		// Verify type is ADDED
		if event.Type != watch.Added {
			violations++
			violationDetails[event.Type]++
			if violations <= 10 {
				t.Errorf("Event #%d has type %v (expected ADDED), key=%s", i, event.Type, event.Key)
			}
		}

		// Verify no PrevObject
		if event.PrevObject != nil {
			t.Errorf("Event #%d has PrevObject (should be nil), key=%s", i, event.Key)
		}
	}

	if violations > 0 {
		t.Errorf("FAILURE: Found %d non-ADDED events in initial snapshot (issue #134831 reproduced!)", violations)
		for eventType, count := range violationDetails {
			t.Errorf("  %s events: %d", eventType, count)
		}
		t.Error("This violates the API contract - initial events should always be ADDED")
	} else {
		t.Logf("SUCCESS: All %d initial events were ADDED ✓", len(events))
		t.Log("Race condition did not manifest in this run")
	}

	// Now test with the cacheWatcher to verify the fix works
	t.Log("Testing convertToWatchEvent with the fix...")
	filter := func(_ string, _ labels.Set, _ fields.Set) bool { return true }
	cw := newCacheWatcher(
		100,
		filter,
		func(bool) {},
		&storage.APIObjectVersioner{},
		time.Time{},
		false,
		schema.GroupResource{Resource: "pods"},
		"test-watcher",
	)

	fixedEvents := 0
	for _, event := range events {
		watchEvent := cw.convertToWatchEvent(event)
		if watchEvent != nil {
			if watchEvent.Type != watch.Added {
				t.Errorf("convertToWatchEvent returned non-ADDED type %v even with IsInitialEvent=true!", watchEvent.Type)
			}
			fixedEvents++
		}
	}

	t.Logf("Converted %d events through cacheWatcher - all forced to ADDED ✓", fixedEvents)
}
