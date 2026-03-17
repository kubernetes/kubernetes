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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/utils/ptr"
)

// TestSendInitialEvents_RealRaceCondition reproduces Paul Furtado's bug:
// Pods that appear in ADDED events (from initial snapshot) ALSO appear as
// MODIFIED/DELETED events BEFORE the BOOKMARK.
//
// The scenario:
// 1. Create 10k pods
// 2. Start watch with sendInitialEvents=true
// 3. IMMEDIATELY start modifying those SAME pods
// 4. Verify: modified events should appear AFTER BOOKMARK, not before
func TestSendInitialEvents_RealRaceCondition(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping race detection test in short mode")
	}

	ctx, cacher, terminate := testSetup(t)
	defer terminate()

	t.Log("Creating 10,000 pods...")
	podNames := make([]string, 10000)
	for i := 0; i < 10000; i++ {
		podName := fmt.Sprintf("test-pod-%d", i)
		podNames[i] = podName

		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:       podName,
				Namespace:  "default",
				Annotations: map[string]string{"version": "1"},
			},
		}

		key := fmt.Sprintf("/pods/default/%s", podName)
		if err := cacher.Create(ctx, key, pod, nil, 0); err != nil {
			t.Fatalf("Failed to create pod %s: %v", podName, err)
		}

		if (i+1)%2000 == 0 {
			t.Logf("Created %d/10000 pods...", i+1)
		}
	}

	// Wait for all events to be processed
	time.Sleep(1 * time.Second)

	t.Log("Starting watch...")
	watchOpts := storage.ListOptions{
		Predicate: func() storage.SelectionPredicate {
			p := storage.Everything
			p.AllowWatchBookmarks = true
			return p
		}(),
		SendInitialEvents: ptr.To(true),
		ResourceVersion:   "0",
	}

	w, err := cacher.Watch(ctx, "/pods/default", watchOpts)
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w.Stop()

	// IMMEDIATELY start modifying pods while initial events are being sent
	t.Log("Starting modifications of existing pods...")
	var wg sync.WaitGroup
	stopMod := make(chan struct{})

	for workerID := 0; workerID < 10; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for {
				select {
				case <-stopMod:
					return
				default:
					// Modify random pods
					podIdx := (id * 1000) % len(podNames)
					podName := podNames[podIdx]
					key := fmt.Sprintf("/pods/default/%s", podName)

					// Get current pod
					currentPod := &example.Pod{}
					if err := cacher.Get(ctx, key, storage.GetOptions{}, currentPod); err != nil {
						continue
					}

					// Modify it
					if currentPod.Annotations == nil {
						currentPod.Annotations = make(map[string]string)
					}
					currentPod.Annotations["modified"] = fmt.Sprintf("worker-%d-%d", id, time.Now().UnixNano())

					// Update
					if err := cacher.GuaranteedUpdate(ctx, key, currentPod, false, nil,
						storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
							return currentPod, nil
						}), nil); err != nil {
						// Ignore errors
					}

					time.Sleep(1 * time.Millisecond)
				}
			}
		}(workerID)
	}

	// Collect all events
	t.Log("Collecting events...")
	var beforeBookmark []watch.Event
	var afterBookmark []watch.Event
	sawBookmark := false
	timeout := time.After(30 * time.Second)

	// Track which pods we saw in initial events
	seenInInitial := make(map[string]bool)

loop:
	for {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				break loop
			}

			if event.Type == watch.Bookmark {
				pod, ok := event.Object.(*example.Pod)
				if ok && pod.GetAnnotations() != nil {
					if pod.GetAnnotations()["k8s.io/initial-events-end"] == "true" {
						t.Logf("Received BOOKMARK after %d events", len(beforeBookmark))
						sawBookmark = true
						close(stopMod)
						// Collect a few more events after bookmark
						collectAfter := time.After(2 * time.Second)
					collectLoop:
						for {
							select {
							case evt, ok := <-w.ResultChan():
								if !ok {
									break collectLoop
								}
								afterBookmark = append(afterBookmark, evt)
							case <-collectAfter:
								break collectLoop
							}
						}
						break loop
					}
				}
			}

			if !sawBookmark {
				beforeBookmark = append(beforeBookmark, event)
				// Track pods seen in initial events
				if pod, ok := event.Object.(*example.Pod); ok {
					seenInInitial[pod.Name] = true
				}
			}

		case <-timeout:
			t.Fatalf("Timeout waiting for BOOKMARK")
		}
	}

	wg.Wait()

	t.Logf("Collected %d events before BOOKMARK, %d after", len(beforeBookmark), len(afterBookmark))

	// VALIDATE: Check if any MODIFIED/DELETED events before BOOKMARK are for pods we already saw as ADDED
	duplicates := 0
	for i, event := range beforeBookmark {
		if event.Type == watch.Modified || event.Type == watch.Deleted {
			if pod, ok := event.Object.(*example.Pod); ok {
				if seenInInitial[pod.Name] {
					duplicates++
					if duplicates <= 10 {
						t.Errorf("Event #%d: Pod %q appeared as %s before BOOKMARK, but was already in initial ADDED events!",
							i, pod.Name, event.Type)
					}
				}
			}
		}
	}

	if duplicates > 0 {
		t.Errorf("\nBUG REPRODUCED: %d MODIFIED/DELETED events before BOOKMARK for pods already sent as ADDED", duplicates)
		t.Error("This is the exact bug Paul Furtado reported!")
	} else {
		t.Log("\nNo duplicate events detected - bug not reproduced (or already fixed)")
	}
}
