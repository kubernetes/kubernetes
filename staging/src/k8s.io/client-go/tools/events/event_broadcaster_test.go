/*
Copyright 2022 The Kubernetes Authors.

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

package events

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/clock"
)

func TestPendingEventRemainsCountedWhenRequeued(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	createStarted := make(chan struct{})
	releaseCreate := make(chan struct{})
	sink := &testEventSeriesSink{
		OnCreate: func(event *eventsv1.Event) (*eventsv1.Event, error) {
			close(createStarted)
			<-releaseCreate
			return event, nil
		},
	}
	broadcaster := newBroadcaster(sink, 0, map[eventKey]*eventsv1.Event{}).(*eventBroadcasterImpl)
	t.Cleanup(func() {
		broadcaster.eventQueue.ShutDown()
		broadcaster.Shutdown()
	})

	event := &eventsv1.Event{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: metav1.NamespaceDefault},
		Regarding: corev1.ObjectReference{
			Namespace: metav1.NamespaceDefault,
			Name:      "pod",
		},
		Reason: "FailedScheduling",
		Action: "Scheduling",
	}
	key := getKey(event)
	broadcaster.recordToSink(ctx, event, clock.RealClock{})

	queuedKey, shutdown := broadcaster.eventQueue.Get()
	if shutdown || queuedKey != key {
		t.Fatalf("expected the first event key from the queue, got key %#v, shutdown %t", queuedKey, shutdown)
	}
	firstAttemptDone := make(chan struct{})
	go func() {
		defer close(firstAttemptDone)
		broadcaster.processNextItem(ctx, key)
	}()

	select {
	case <-createStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for the first recording attempt")
	}

	// This transitions the cached Event into a series and marks the key dirty
	// while the first recording attempt is still in progress.
	broadcaster.recordToSink(ctx, event, clock.RealClock{})
	close(releaseCreate)
	select {
	case <-firstAttemptDone:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for the first recording attempt to finish")
	}

	broadcaster.mu.Lock()
	_, remainsPending := broadcaster.pending[key]
	broadcaster.mu.Unlock()
	if !remainsPending {
		t.Fatal("expected a dirty key to remain pending until its next recording attempt")
	}

	broadcaster.eventQueue.Done(key)
	if got := broadcaster.eventQueue.Len(); got != 1 {
		t.Fatalf("expected the dirty key to be requeued, got queue length %d", got)
	}
	queuedKey, shutdown = broadcaster.eventQueue.Get()
	if shutdown || queuedKey != key {
		t.Fatalf("expected the dirty event key from the queue, got key %#v, shutdown %t", queuedKey, shutdown)
	}
	broadcaster.processNextItem(ctx, key)
	broadcaster.eventQueue.Done(key)

	broadcaster.mu.Lock()
	_, remainsPending = broadcaster.pending[key]
	broadcaster.mu.Unlock()
	if remainsPending {
		t.Fatal("expected the key to stop counting as pending after its latest generation was recorded")
	}
}

func TestRecordToSinkDropsNewEventsWhenTooManyPending(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	makeEvent := func(name string) *eventsv1.Event {
		return &eventsv1.Event{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: metav1.NamespaceDefault},
			Regarding: corev1.ObjectReference{
				Namespace: metav1.NamespaceDefault,
				Name:      name,
			},
			Reason: "FailedScheduling",
			Action: "Scheduling",
		}
	}
	broadcaster := newBroadcaster(&testEventSeriesSink{}, 0, map[eventKey]*eventsv1.Event{}).(*eventBroadcasterImpl)
	t.Cleanup(broadcaster.Shutdown)
	// Simulate a slow sink: maxPendingEvents keys are cached and still
	// awaiting recording, plus one cached key that was already recorded.
	for i := range maxPendingEvents {
		event := makeEvent(fmt.Sprintf("pod-%d", i))
		broadcaster.eventCache[getKey(event)] = event
		broadcaster.pending[getKey(event)] = 1
	}
	recordedEvent := makeEvent("pod-recorded")
	broadcaster.eventCache[getKey(recordedEvent)] = recordedEvent

	newEvent := makeEvent("pod-over-capacity")
	broadcaster.recordToSink(ctx, newEvent, clock.RealClock{})

	broadcaster.mu.Lock()
	_, cached := broadcaster.eventCache[getKey(newEvent)]
	pendingLen := len(broadcaster.pending)
	broadcaster.mu.Unlock()
	if cached {
		t.Error("expected an event for a new key to be dropped while too many events await recording")
	}
	if pendingLen != maxPendingEvents {
		t.Errorf("expected pending to stay at %d keys, got %d", maxPendingEvents, pendingLen)
	}
	if got := broadcaster.eventQueue.Len(); got != 0 {
		t.Errorf("expected no queued keys for a dropped event, got %d", got)
	}

	// A series transition for a cached, non-pending key must still be
	// aggregated, but its recording is deferred instead of queued.
	broadcaster.recordToSink(ctx, makeEvent("pod-recorded"), clock.RealClock{})
	broadcaster.mu.Lock()
	series := broadcaster.eventCache[getKey(recordedEvent)].Series
	_, isPending := broadcaster.pending[getKey(recordedEvent)]
	broadcaster.mu.Unlock()
	if series == nil || series.Count != 2 {
		t.Errorf("expected an isomorphic event to start a series with count 2, got %+v", series)
	}
	if isPending {
		t.Error("expected the deferred series transition not to become pending")
	}
	if got := broadcaster.eventQueue.Len(); got != 0 {
		t.Errorf("expected the deferred series transition not to be queued, got queue length %d", got)
	}

	// A series transition for a key that is already pending stays allowed.
	broadcaster.recordToSink(ctx, makeEvent("pod-0"), clock.RealClock{})
	broadcaster.mu.Lock()
	series = broadcaster.eventCache[getKey(makeEvent("pod-0"))].Series
	broadcaster.mu.Unlock()
	if series == nil || series.Count != 2 {
		t.Errorf("expected an isomorphic event for a pending key to start a series with count 2, got %+v", series)
	}
	if got := broadcaster.eventQueue.Len(); got != 1 {
		t.Errorf("expected the series transition for a pending key to be queued, got queue length %d", got)
	}
}

func TestStartRecordingAfterShutdownKeepsFailing(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	broadcaster := newBroadcaster(&testEventSeriesSink{}, 0, map[eventKey]*eventsv1.Event{}).(*eventBroadcasterImpl)
	broadcaster.Shutdown()
	if err := broadcaster.StartRecordingToSinkWithContext(ctx); err == nil {
		t.Fatal("expected an error when starting recording after shutdown")
	}
	if err := broadcaster.StartRecordingToSinkWithContext(ctx); err == nil {
		t.Fatal("expected a repeated start attempt to report the failure, not success")
	}
}

func TestShutdownStopsRecordingWorkers(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	broadcaster := newBroadcaster(&testEventSeriesSink{}, 0, map[eventKey]*eventsv1.Event{}).(*eventBroadcasterImpl)
	if err := broadcaster.StartRecordingToSinkWithContext(ctx); err != nil {
		t.Fatalf("unexpected error starting recording: %v", err)
	}
	// A repeated start must be a no-op: even with an already canceled
	// context it must not spawn goroutines that shut down the shared queue.
	canceledCtx, cancel := context.WithCancel(ctx)
	cancel()
	if err := broadcaster.StartRecordingToSinkWithContext(canceledCtx); err != nil {
		t.Fatalf("unexpected error on repeated start: %v", err)
	}
	time.Sleep(10 * time.Millisecond)
	if broadcaster.eventQueue.ShuttingDown() {
		t.Fatal("expected a repeated start to be a no-op, but the event queue was shut down")
	}

	broadcaster.Shutdown()
	if !broadcaster.eventQueue.ShuttingDown() {
		t.Fatal("expected Shutdown to shut down the event queue and stop the workers")
	}
}

func TestRecordEventToSink(t *testing.T) {
	nonIsomorphicEvent := eventsv1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: metav1.NamespaceDefault,
		},
		Series: nil,
	}

	isomorphicEvent := *nonIsomorphicEvent.DeepCopy()
	isomorphicEvent.Series = &eventsv1.EventSeries{Count: 2}

	testCases := []struct {
		name                  string
		eventsToRecord        []eventsv1.Event
		expectedRecordedEvent eventsv1.Event
	}{
		{
			name: "record one Event",
			eventsToRecord: []eventsv1.Event{
				nonIsomorphicEvent,
			},
			expectedRecordedEvent: nonIsomorphicEvent,
		},
		{
			name: "record one Event followed by an isomorphic one",
			eventsToRecord: []eventsv1.Event{
				nonIsomorphicEvent,
				isomorphicEvent,
			},
			expectedRecordedEvent: isomorphicEvent,
		},
		{
			name: "record one isomorphic Event before the original",
			eventsToRecord: []eventsv1.Event{
				isomorphicEvent,
				nonIsomorphicEvent,
			},
			expectedRecordedEvent: isomorphicEvent,
		},
		{
			name: "record one isomorphic Event without one already existing",
			eventsToRecord: []eventsv1.Event{
				isomorphicEvent,
			},
			expectedRecordedEvent: isomorphicEvent,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			kubeClient := fake.NewSimpleClientset()
			eventSink := &EventSinkImpl{Interface: kubeClient.EventsV1()}

			for _, ev := range tc.eventsToRecord {
				recordEvent(ctx, eventSink, &ev)
			}

			recordedEvents, err := kubeClient.EventsV1().Events(metav1.NamespaceDefault).List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				t.Errorf("expected to be able to list Events from fake client")
			}

			if len(recordedEvents.Items) != 1 {
				t.Errorf("expected one Event to be recorded, found: %d", len(recordedEvents.Items))
			}

			recordedEvent := recordedEvents.Items[0]
			if !reflect.DeepEqual(recordedEvent, tc.expectedRecordedEvent) {
				t.Errorf("expected to have recorded Event: %#+v, got: %#+v\n diff: %s", tc.expectedRecordedEvent, recordedEvent, cmp.Diff(tc.expectedRecordedEvent, recordedEvent))
			}
		})
	}
}
