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
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2/ktesting"
)

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

func TestEventBroadcasterConcurrencyLimit(t *testing.T) {
	// Create a mock EventSink that blocks on writes and tracks active concurrent calls
	var mu sync.Mutex
	activeCalls := 0
	maxActiveCalls := 0
	releaseCh := make(chan struct{})

	mockSink := &mockBlockingSink{
		createFunc: func(ctx context.Context, event *eventsv1.Event) (*eventsv1.Event, error) {
			mu.Lock()
			activeCalls++
			if activeCalls > maxActiveCalls {
				maxActiveCalls = activeCalls
			}
			mu.Unlock()

			defer func() {
				mu.Lock()
				activeCalls--
				mu.Unlock()
			}()

			// Block until signaled to release
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-releaseCh:
				return event, nil
			}
		},
	}

	broadcaster := NewBroadcasterWithOptions(mockSink, WithMaxConcurrentRecording(2))
	defer broadcaster.Shutdown()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := broadcaster.StartRecordingToSinkWithContext(ctx); err != nil {
		t.Fatalf("unexpected error starting recording: %v", err)
	}

	recorder := broadcaster.NewRecorder(scheme.Scheme, "test-controller")

	// Emit 5 distinct events. Since maxConcurrentRecording is 2, only 2 events
	// should be processed concurrently by the sink. The rest should block in recordToSink.
	for i := 0; i < 5; i++ {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("pod-%d", i),
				Namespace: "default",
			},
		}
		recorder.Eventf(pod, nil, corev1.EventTypeNormal, "Reason", "Action", "Message %d", i)
	}

	// Give it a little time to spawn goroutines and execute them up to the concurrency limit
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	currentMaxActive := maxActiveCalls
	mu.Unlock()

	if currentMaxActive != 2 {
		t.Errorf("Expected exactly 2 concurrent active calls to the sink, got %d", currentMaxActive)
	}

	// Signal the blocking sink to release
	close(releaseCh)
}

type mockBlockingSink struct {
	createFunc func(context.Context, *eventsv1.Event) (*eventsv1.Event, error)
}

func (m *mockBlockingSink) Create(ctx context.Context, event *eventsv1.Event) (*eventsv1.Event, error) {
	return m.createFunc(ctx, event)
}

func (m *mockBlockingSink) Update(ctx context.Context, event *eventsv1.Event) (*eventsv1.Event, error) {
	return event, nil
}

func (m *mockBlockingSink) Patch(ctx context.Context, event *eventsv1.Event, data []byte) (*eventsv1.Event, error) {
	return event, nil
}
