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
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
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

// TestConcurrentIsomorphicEventfRace stresses concurrent isomorphic Eventf calls; run with -race.
func TestConcurrentIsomorphicEventfRace(t *testing.T) {
	const concurrency = 100

	kubeClient := fake.NewSimpleClientset()
	eventSink := &EventSinkImpl{Interface: kubeClient.EventsV1()}
	eventBroadcaster := newBroadcaster(eventSink, 100*time.Millisecond, map[eventKey]*eventsv1.Event{})

	stopCh := make(chan struct{})
	defer close(stopCh)
	eventBroadcaster.StartRecordingToSink(stopCh)

	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "test")
	regarding := &v1.ObjectReference{Name: "foo", Namespace: metav1.NamespaceDefault, UID: "bar"}

	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go func() {
			defer wg.Done()
			<-start
			recorder.Eventf(regarding, nil, v1.EventTypeNormal, "memoryPressure", "killed", "memory pressure")
		}()
	}
	close(start)
	wg.Wait()

	err := wait.PollUntilContextTimeout(context.Background(), 50*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		events, listErr := kubeClient.EventsV1().Events(metav1.NamespaceDefault).List(ctx, metav1.ListOptions{})
		if listErr != nil {
			return false, listErr
		}
		if len(events.Items) != 1 {
			return false, nil
		}
		if events.Items[0].Series == nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		events, _ := kubeClient.EventsV1().Events(metav1.NamespaceDefault).List(context.Background(), metav1.ListOptions{})
		t.Fatalf("expected exactly 1 Event with non-nil Series, got %d events: %#v", len(events.Items), events.Items)
	}
}
