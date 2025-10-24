// +build stress

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

// TestSendInitialEvents_DefinitiveRaceProof definitively proves the bug:
//
// Timeline:
// T0: Snapshot created at RV=10000 (contains pod-0...pod-9999)
// T1: Watcher REGISTERED → starts receiving dispatched events
// T2: pod-5000 modified → RV=10001 → dispatched → queued in watcher.input
// T3: processInterval STARTS sending initial events (slow, takes seconds)
// T4: pod-7000 modified → RV=10002 → dispatched → queued in watcher.input
// T5: processInterval sends BOOKMARK
// T6: process() drains watcher.input → sends RV=10001, RV=10002
//
// BUG: Client sees pod-5000 ADDED (initial), then pod-5000 MODIFIED (before BOOKMARK)
func TestSendInitialEvents_DefinitiveRaceProof(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	defer terminate()

	// Create 50k pods to make processInterval slow
	numPods := 50000
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
		if (i+1)%10000 == 0 {
			t.Logf("Created %d/%d...", i+1, numPods)
		}
	}

	// Wait for processing
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

	// IMMEDIATELY start modifying pods (to create events that queue up)
	t.Log("Triggering modifications to create race condition...")
	modifyCtx, cancelModify := context.WithCancel(ctx)
	defer cancelModify()

	// Continuously modify pods in background
	go func() {
		for i := 0; ; i++ {
			select {
			case <-modifyCtx.Done():
				return
			default:
				podIdx := i % numPods
				key := fmt.Sprintf("/pods/default/pod-%d", podIdx)

				pod := &example.Pod{}
				if err := cacher.Get(ctx, key, storage.GetOptions{}, pod); err != nil {
					continue
				}

				pod.Annotations["modified-at"] = fmt.Sprintf("%d", time.Now().UnixNano())

				// This creates an event that gets dispatched
				_ = cacher.GuaranteedUpdate(ctx, key, &example.Pod{}, false, nil,
					storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
						return pod, nil
					}), nil)

				time.Sleep(100 * time.Microsecond) // Fast modifications
			}
		}
	}()

	// Collect events
	podsSeen := make(map[string][]watch.EventType) // Track all events per pod
	sawBookmark := false
	timeout := time.After(60 * time.Second)

	t.Log("Collecting events...")
	eventCount := 0
loop:
	for {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				break loop
			}

			eventCount++
			if eventCount%10000 == 0 {
				t.Logf("Received %d events...", eventCount)
			}

			if event.Type == watch.Bookmark {
				if pod, ok := event.Object.(*example.Pod); ok {
					if pod.GetAnnotations() != nil && pod.GetAnnotations()["k8s.io/initial-events-end"] == "true" {
						t.Logf("BOOKMARK received after %d events", eventCount-1)
						sawBookmark = true

						// Collect a bit more after bookmark
						afterBookmarkTimeout := time.After(2 * time.Second)
					afterLoop:
						for {
							select {
							case evt, ok := <-w.ResultChan():
								if !ok {
									break afterLoop
								}
								if pod, ok := evt.Object.(*example.Pod); ok {
									podsSeen[pod.Name] = append(podsSeen[pod.Name], evt.Type)
								}
								eventCount++
							case <-afterBookmarkTimeout:
								break afterLoop
							}
						}
						break loop
					}
				}
			}

			if pod, ok := event.Object.(*example.Pod); ok {
				if !sawBookmark {
					podsSeen[pod.Name] = append(podsSeen[pod.Name], event.Type)
				}
			}

		case <-timeout:
			t.Fatalf("Timeout waiting for BOOKMARK")
		}
	}

	cancelModify()
	t.Logf("Total events collected: %d", eventCount)

	// ANALYZE: Find pods that appeared multiple times BEFORE bookmark
	violations := 0
	for podName, eventTypes := range podsSeen {
		if len(eventTypes) > 1 {
			// This pod appeared multiple times before bookmark!
			violations++
			if violations <= 10 {
				t.Errorf("RACE DETECTED: Pod %q appeared %d times before BOOKMARK: %v",
					podName, len(eventTypes), eventTypes)
			}
		}
	}

	if violations > 0 {
		t.Errorf("\nBUG CONFIRMED: %d pods appeared multiple times before BOOKMARK", violations)
		t.Error("This proves events from watcher.input are leaking before the BOOKMARK!")
		t.Error("Root cause: Watcher registered BEFORE processInterval sends bookmark")
	} else {
		t.Log("\nNo violations - either bug is fixed or didn't reproduce")
	}
}
