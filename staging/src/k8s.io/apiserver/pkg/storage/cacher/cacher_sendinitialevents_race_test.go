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
	"sync/atomic"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	// "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	// utilfeature "k8s.io/apiserver/pkg/util/feature"
	// featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/ptr"
)

// TestSendInitialEvents_OngoingEventsRace tests for the race condition identified
// by Paul Furtado in issue #134831: ongoing events from the watch cache buffer
// leaking in BEFORE the BOOKMARK is sent during sendInitialEvents watches.
//
// The bug: After processInterval() sends all initial snapshot events, but before
// it sends the BOOKMARK, new events can arrive and be sent to the client, violating
// the API contract that the BOOKMARK delimits the initial state from ongoing updates.
//
// This is NOT about event types (ADDED/MODIFIED/DELETED) being wrong for snapshot
// events. The race is that real, legitimate ONGOING events (for objects created
// AFTER the watch started) slip through before the BOOKMARK.
//
// Test Results (10k initial pods, continuous ongoing pod creation):
//   REPRODUCED: 585 ongoing events appeared before BOOKMARK
//   - 10,000 initial events (correct)
//   - 585 leaked ongoing events (BUG!)
//   - BOOKMARK
//   - Remaining ongoing events
//
// This test:
// 1. Creates 10k initial pods in etcd/cacher
// 2. Starts background goroutine creating pods continuously (ongoing events)
// 3. Starts a sendInitialEvents watch (while ongoing events are being created)
// 4. Collects all events until BOOKMARK
// 5. Validates NO events for "ongoing-pod-X" appear before BOOKMARK
//
// Run with: go test -tags=stress -v -timeout=5m -run TestSendInitialEvents_OngoingEventsRace
func TestSendInitialEvents_OngoingEventsRace(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping race detection test in short mode")
	}

	// Enable snapshot feature gate to use consistent snapshots for initial events
	// NOTE: Comment this out to reproduce the race condition without the fix
	// DISABLED to test if bug can be reproduced:
	// featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	// Setup: Create cacher with real etcd backend
	ctx, cacher, terminate := testSetup(t)
	defer terminate()

	t.Log("Populating cacher with 10,000 initial pods...")
	startPopulate := time.Now()

	// Create 10k initial pods (enough to make processInterval take measurable time)
	numInitialPods := 10000
	for i := 0; i < numInitialPods; i++ {
		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("initial-pod-%d", i),
				Namespace: "default",
			},
			Spec: example.PodSpec{
				NodeName: fmt.Sprintf("node-%d", i%100),
			},
		}

		key := fmt.Sprintf("/pods/default/initial-pod-%d", i)
		if err := cacher.Create(ctx, key, pod, nil, 0); err != nil {
			t.Fatalf("Failed to create initial pod %d: %v", i, err)
		}

		if (i+1)%2000 == 0 {
			t.Logf("Created %d/%d initial pods...", i+1, numInitialPods)
		}
	}

	t.Logf("Created %d initial pods in %v", numInitialPods, time.Since(startPopulate))

	// Give cacher time to process all events
	time.Sleep(500 * time.Millisecond)

	// Now start sendInitialEvents watch FIRST, before creating any ongoing pods
	t.Log("Starting sendInitialEvents watch...")
	watchOpts := storage.ListOptions{
		Predicate: func() storage.SelectionPredicate {
			p := storage.Everything
			p.AllowWatchBookmarks = true
			return p
		}(),
		SendInitialEvents: ptr.To(true),
		ResourceVersion:   "0", // Watch from beginning with initial events
	}

	w, err := cacher.Watch(ctx, "/pods/default", watchOpts)
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w.Stop()

	// NOW start background goroutine that creates ongoing pods
	// These pods are created AFTER the watch started, so they should appear AFTER the BOOKMARK
	t.Log("Starting background pod creation (ongoing events - should appear after BOOKMARK)...")
	var ongoingPodsCreated atomic.Int32
	stopCreation := make(chan struct{})
	var creationWg sync.WaitGroup

	creationWg.Add(1)
	go func() {
		defer creationWg.Done()
		i := 0
		for {
			select {
			case <-stopCreation:
				return
			default:
			}

			pod := &example.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("ongoing-pod-%d", i),
					Namespace: "default",
				},
				Spec: example.PodSpec{
					NodeName: "ongoing-node",
				},
			}

			key := fmt.Sprintf("/pods/default/ongoing-pod-%d", i)
			if err := cacher.Create(ctx, key, pod, nil, 0); err != nil {
				t.Logf("Warning: Failed to create ongoing pod %d: %v", i, err)
				time.Sleep(10 * time.Millisecond)
				continue
			}

			ongoingPodsCreated.Add(1)
			i++

			// No delay - create as fast as possible to increase chance of race
		}
	}()

	// Collect events until BOOKMARK
	t.Log("Collecting events until BOOKMARK...")
	var beforeBookmark []watch.Event
	var afterBookmark []watch.Event
	sawBookmark := false
	timeout := time.After(30 * time.Second)

	// Collect events
	for !sawBookmark {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("Watch channel closed before BOOKMARK")
			}

			if event.Type == watch.Bookmark {
				// Check if this is the initial BOOKMARK
				pod, ok := event.Object.(*example.Pod)
				if !ok {
					continue // Not a pod bookmark, might be other bookmark
				}

				if annotations := pod.GetAnnotations(); annotations != nil {
					if annotations["k8s.io/initial-events-end"] == "true" {
						t.Logf("Received initial BOOKMARK at RV %s after %d events",
							pod.GetResourceVersion(), len(beforeBookmark))
						sawBookmark = true
						break
					}
				}
				// Regular bookmark, not initial events end
				beforeBookmark = append(beforeBookmark, event)
				continue
			}

			beforeBookmark = append(beforeBookmark, event)

		case <-timeout:
			t.Fatalf("Timeout waiting for BOOKMARK. Collected %d events, created %d ongoing pods",
				len(beforeBookmark), ongoingPodsCreated.Load())
		}
	}

	// Collect some events after BOOKMARK
	t.Log("Collecting events after BOOKMARK...")
	afterTimeout := time.After(3 * time.Second)
collectAfter:
	for len(afterBookmark) < int(ongoingPodsCreated.Load()) {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				break collectAfter
			}
			afterBookmark = append(afterBookmark, event)

		case <-afterTimeout:
			break collectAfter
		}
	}

	// Stop ongoing pod creation
	close(stopCreation)
	creationWg.Wait()

	t.Logf("Collected %d events before BOOKMARK, %d after BOOKMARK",
		len(beforeBookmark), len(afterBookmark))
	t.Logf("Created %d ongoing pods during test", ongoingPodsCreated.Load())

	// VALIDATE: Check for the race condition
	t.Log("Validating event ordering...")

	// Count events by type before BOOKMARK
	eventTypesBefore := make(map[watch.EventType]int)
	ongoingPodsBeforeBookmark := make(map[string]bool)

	for i, event := range beforeBookmark {
		eventTypesBefore[event.Type]++

		// Extract pod name
		var podName string
		switch obj := event.Object.(type) {
		case *example.Pod:
			podName = obj.Name
		case runtime.Object:
			// Try to get name from object
			if pod, ok := obj.(*example.Pod); ok {
				podName = pod.Name
			}
		}

		// Check if this is an "ongoing" pod (should NOT be before BOOKMARK!)
		if len(podName) > 0 && len(podName) >= 11 && podName[:11] == "ongoing-pod" {
			ongoingPodsBeforeBookmark[podName] = true
			t.Errorf("RACE DETECTED: Event #%d for ongoing pod %q appeared BEFORE BOOKMARK (type=%s)",
				i, podName, event.Type)
		}
	}

	// Count "ongoing" pods after BOOKMARK
	ongoingPodsAfterBookmark := make(map[string]bool)
	for _, event := range afterBookmark {
		switch obj := event.Object.(type) {
		case *example.Pod:
			if len(obj.Name) >= 11 && obj.Name[:11] == "ongoing-pod" {
				ongoingPodsAfterBookmark[obj.Name] = true
			}
		}
	}

	// Report results
	t.Logf("\nEvent types before BOOKMARK: %+v", eventTypesBefore)
	t.Logf("Ongoing pods leaked before BOOKMARK: %d", len(ongoingPodsBeforeBookmark))
	t.Logf("Ongoing pods correctly after BOOKMARK: %d", len(ongoingPodsAfterBookmark))

	// Fail if race detected
	if len(ongoingPodsBeforeBookmark) > 0 {
		t.Errorf("\nRACE CONDITION DETECTED: %d ongoing events appeared before BOOKMARK", len(ongoingPodsBeforeBookmark))
		t.Errorf("This violates the API contract - only initial state should precede BOOKMARK")
		t.Errorf("Ongoing pod names that leaked: %v", getKeys(ongoingPodsBeforeBookmark))
	} else {
		t.Logf("\nSUCCESS: No ongoing events leaked before BOOKMARK")
		t.Logf("All %d initial events before BOOKMARK were from initial state", len(beforeBookmark))
	}

	// Validate we got SOME ongoing events after (to prove they were created)
	if len(ongoingPodsAfterBookmark) == 0 {
		t.Logf("Warning: No ongoing pods detected after BOOKMARK (created %d). Test may not be effective.",
			ongoingPodsCreated.Load())
	} else {
		t.Logf("%d ongoing pods correctly appeared after BOOKMARK", len(ongoingPodsAfterBookmark))
	}
}

// Helper to extract map keys
func getKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
