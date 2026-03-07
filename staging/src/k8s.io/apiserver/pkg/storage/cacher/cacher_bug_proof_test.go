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
	"context"
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/utils/ptr"
)

// TestSendInitialEvents_BugProof uses a controlled scenario to definitively prove
// whether the race condition exists.
//
// Strategy:
// 1. Create N pods
// 2. Start watch with sendInitialEvents=true
// 3. IMMEDIATELY start modifying a SPECIFIC pod (pod-0) continuously
// 4. Track ALL events for pod-0 before BOOKMARK
// 5. If pod-0 appears more than once before BOOKMARK, bug is proven
//
// The key insight: If we modify pod-0 fast enough and continuously, and the
// interval reads from live store, we should see pod-0 appearing multiple times.
func TestSendInitialEvents_BugProof(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping bug proof test in short mode")
	}

	ctx, cacher, terminate := testSetup(t)
	defer terminate()

	// Create 1000 pods
	numPods := 1000
	t.Logf("Creating %d pods...", numPods)
	for i := 0; i < numPods; i++ {
		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:        fmt.Sprintf("pod-%d", i),
				Namespace:   "default",
				Annotations: map[string]string{"version": "1"},
			},
		}
		key := fmt.Sprintf("/pods/default/pod-%d", i)
		if err := cacher.Create(ctx, key, pod, nil, 0); err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}
	}

	// Wait for processing
	time.Sleep(500 * time.Millisecond)

	// Start RAPID modifications of pod-0 BEFORE starting watch
	modifyStop := make(chan struct{})
	modifyStarted := make(chan struct{})
	go func() {
		close(modifyStarted)
		ticker := time.NewTicker(1 * time.Millisecond) // Very fast
		defer ticker.Stop()
		version := 2
		for {
			select {
			case <-modifyStop:
				return
			case <-ticker.C:
				key := "/pods/default/pod-0"
				pod := &example.Pod{}
				if err := cacher.Get(ctx, key, storage.GetOptions{}, pod); err != nil {
					continue
				}
				pod.Annotations["version"] = fmt.Sprintf("%d", version)
				version++
				_ = cacher.GuaranteedUpdate(ctx, key, &example.Pod{}, false, nil,
					storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						return pod, nil
					}), nil)
			}
		}
	}()
	<-modifyStarted

	// Let modifications happen for a bit
	time.Sleep(100 * time.Millisecond)

	t.Log("Starting watch with sendInitialEvents=true...")
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

	// Collect ALL events before BOOKMARK with timestamps
	type timedEvent struct {
		event watch.Event
		time  time.Time
		seq   int
	}
	var allEvents []timedEvent
	sawBookmark := false
	timeout := time.After(30 * time.Second)
	seq := 0

	for !sawBookmark {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("Watch closed before BOOKMARK")
			}
			seq++
			allEvents = append(allEvents, timedEvent{event: event, time: time.Now(), seq: seq})

			if event.Type == watch.Bookmark {
				if pod, ok := event.Object.(*example.Pod); ok {
					if pod.GetAnnotations() != nil && pod.GetAnnotations()["k8s.io/initial-events-end"] == "true" {
						t.Logf("BOOKMARK received after %d events", len(allEvents)-1)
						sawBookmark = true
					}
				}
			}
		case <-timeout:
			t.Fatalf("Timeout waiting for BOOKMARK")
		}
	}

	close(modifyStop)

	// Analyze: Find pod-0 events
	var pod0Events []timedEvent
	for _, te := range allEvents {
		if te.event.Type == watch.Bookmark {
			break
		}
		if pod, ok := te.event.Object.(*example.Pod); ok {
			if pod.Name == "pod-0" {
				pod0Events = append(pod0Events, te)
			}
		}
	}

	t.Logf("Found %d events for pod-0 before BOOKMARK:", len(pod0Events))
	for i, te := range pod0Events {
		pod := te.event.Object.(*example.Pod)
		version := pod.Annotations["version"]
		t.Logf("  Event #%d (seq=%d): Type=%s, Version=%s, Time=%v",
			i+1, te.seq, te.event.Type, version, te.time.Format("15:04:05.000"))
	}

	if len(pod0Events) > 1 {
		t.Errorf("\nBUG PROVEN: pod-0 appeared %d times before BOOKMARK!", len(pod0Events))
		t.Error("This definitively proves the race condition exists")
	} else if len(pod0Events) == 1 {
		t.Log("\nNo bug detected - pod-0 appeared only once before BOOKMARK")
	} else {
		t.Error("WARNING: pod-0 not found in events - test may be ineffective")
	}
}
