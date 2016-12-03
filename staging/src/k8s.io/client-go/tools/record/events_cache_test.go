/*
Copyright 2015 The Kubernetes Authors.

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

package record

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/client-go/pkg/api/v1"
	metav1 "k8s.io/client-go/pkg/apis/meta/v1"
	"k8s.io/client-go/pkg/util/clock"
	"k8s.io/client-go/pkg/util/diff"
)

func makeObjectReference(kind, name, namespace string) v1.ObjectReference {
	return v1.ObjectReference{
		Kind:       kind,
		Name:       name,
		Namespace:  namespace,
		UID:        "C934D34AFB20242",
		APIVersion: "version",
	}
}

func makeEvent(reason, message string, involvedObject v1.ObjectReference) v1.Event {
	eventTime := metav1.Now()
	event := v1.Event{
		Reason:         reason,
		Message:        message,
		InvolvedObject: involvedObject,
		Source: v1.EventSource{
			Component: "kubelet",
			Host:      "kublet.node1",
		},
		Count:          1,
		FirstTimestamp: eventTime,
		LastTimestamp:  eventTime,
		Type:           v1.EventTypeNormal,
	}
	return event
}

func makeEvents(num int, template v1.Event) []v1.Event {
	events := []v1.Event{}
	for i := 0; i < num; i++ {
		events = append(events, template)
	}
	return events
}

func makeUniqueEvents(num int) []v1.Event {
	events := []v1.Event{}
	kind := "Pod"
	for i := 0; i < num; i++ {
		reason := strings.Join([]string{"reason", string(i)}, "-")
		message := strings.Join([]string{"message", string(i)}, "-")
		name := strings.Join([]string{"pod", string(i)}, "-")
		namespace := strings.Join([]string{"ns", string(i)}, "-")
		involvedObject := makeObjectReference(kind, name, namespace)
		events = append(events, makeEvent(reason, message, involvedObject))
	}
	return events
}

func makeSimilarEvents(num int, template v1.Event, messagePrefix string) []v1.Event {
	events := makeEvents(num, template)
	for i := range events {
		events[i].Message = strings.Join([]string{messagePrefix, string(i), events[i].Message}, "-")
	}
	return events
}

func setCount(event v1.Event, count int) v1.Event {
	event.Count = int32(count)
	return event
}

func validateEvent(messagePrefix string, actualEvent *v1.Event, expectedEvent *v1.Event, t *testing.T) (*v1.Event, error) {
	recvEvent := *actualEvent
	expectCompression := expectedEvent.Count > 1
	t.Logf("%v - expectedEvent.Count is %d\n", messagePrefix, expectedEvent.Count)
	// Just check that the timestamp was set.
	if recvEvent.FirstTimestamp.IsZero() || recvEvent.LastTimestamp.IsZero() {
		t.Errorf("%v - timestamp wasn't set: %#v", messagePrefix, recvEvent)
	}
	actualFirstTimestamp := recvEvent.FirstTimestamp
	actualLastTimestamp := recvEvent.LastTimestamp
	if actualFirstTimestamp.Equal(actualLastTimestamp) {
		if expectCompression {
			t.Errorf("%v - FirstTimestamp (%q) and LastTimestamp (%q) must be different to indicate event compression happened, but were the same. Actual Event: %#v", messagePrefix, actualFirstTimestamp, actualLastTimestamp, recvEvent)
		}
	} else {
		if expectedEvent.Count == 1 {
			t.Errorf("%v - FirstTimestamp (%q) and LastTimestamp (%q) must be equal to indicate only one occurrence of the event, but were different. Actual Event: %#v", messagePrefix, actualFirstTimestamp, actualLastTimestamp, recvEvent)
		}
	}
	// Temp clear time stamps for comparison because actual values don't matter for comparison
	recvEvent.FirstTimestamp = expectedEvent.FirstTimestamp
	recvEvent.LastTimestamp = expectedEvent.LastTimestamp
	// Check that name has the right prefix.
	if n, en := recvEvent.Name, expectedEvent.Name; !strings.HasPrefix(n, en) {
		t.Errorf("%v - Name '%v' does not contain prefix '%v'", messagePrefix, n, en)
	}
	recvEvent.Name = expectedEvent.Name
	if e, a := expectedEvent, &recvEvent; !reflect.DeepEqual(e, a) {
		t.Errorf("%v - diff: %s", messagePrefix, diff.ObjectGoPrintDiff(e, a))
	}
	recvEvent.FirstTimestamp = actualFirstTimestamp
	recvEvent.LastTimestamp = actualLastTimestamp
	return actualEvent, nil
}

// TestDefaultEventFilterFunc ensures that no events are filtered
func TestDefaultEventFilterFunc(t *testing.T) {
	event := makeEvent("end-of-world", "it was fun", makeObjectReference("Pod", "pod1", "other"))
	if DefaultEventFilterFunc(&event) {
		t.Fatalf("DefaultEventFilterFunc should always return false")
	}
}

// TestEventAggregatorByReasonFunc ensures that two events are aggregated if they vary only by event.message
func TestEventAggregatorByReasonFunc(t *testing.T) {
	event1 := makeEvent("end-of-world", "it was fun", makeObjectReference("Pod", "pod1", "other"))
	event2 := makeEvent("end-of-world", "it was awful", makeObjectReference("Pod", "pod1", "other"))
	event3 := makeEvent("nevermind", "it was a bug", makeObjectReference("Pod", "pod1", "other"))

	aggKey1, localKey1 := EventAggregatorByReasonFunc(&event1)
	aggKey2, localKey2 := EventAggregatorByReasonFunc(&event2)
	aggKey3, _ := EventAggregatorByReasonFunc(&event3)

	if aggKey1 != aggKey2 {
		t.Errorf("Expected %v equal %v", aggKey1, aggKey2)
	}
	if localKey1 == localKey2 {
		t.Errorf("Expected %v to not equal %v", aggKey1, aggKey3)
	}
	if aggKey1 == aggKey3 {
		t.Errorf("Expected %v to not equal %v", aggKey1, aggKey3)
	}
}

// TestEventAggregatorByReasonMessageFunc validates the proper output for an aggregate message
func TestEventAggregatorByReasonMessageFunc(t *testing.T) {
	expected := "(events with common reason combined)"
	event1 := makeEvent("end-of-world", "it was fun", makeObjectReference("Pod", "pod1", "other"))
	if actual := EventAggregatorByReasonMessageFunc(&event1); expected != actual {
		t.Errorf("Expected %v got %v", expected, actual)
	}
}

// TestEventCorrelator validates proper counting, aggregation of events
func TestEventCorrelator(t *testing.T) {
	firstEvent := makeEvent("first", "i am first", makeObjectReference("Pod", "my-pod", "my-ns"))
	duplicateEvent := makeEvent("duplicate", "me again", makeObjectReference("Pod", "my-pod", "my-ns"))
	uniqueEvent := makeEvent("unique", "snowflake", makeObjectReference("Pod", "my-pod", "my-ns"))
	similarEvent := makeEvent("similar", "similar message", makeObjectReference("Pod", "my-pod", "my-ns"))
	aggregateEvent := makeEvent(similarEvent.Reason, EventAggregatorByReasonMessageFunc(&similarEvent), similarEvent.InvolvedObject)
	scenario := map[string]struct {
		previousEvents  []v1.Event
		newEvent        v1.Event
		expectedEvent   v1.Event
		intervalSeconds int
	}{
		"create-a-single-event": {
			previousEvents:  []v1.Event{},
			newEvent:        firstEvent,
			expectedEvent:   setCount(firstEvent, 1),
			intervalSeconds: 5,
		},
		"the-same-event-should-just-count": {
			previousEvents:  makeEvents(1, duplicateEvent),
			newEvent:        duplicateEvent,
			expectedEvent:   setCount(duplicateEvent, 2),
			intervalSeconds: 5,
		},
		"the-same-event-should-just-count-even-if-more-than-aggregate": {
			previousEvents:  makeEvents(defaultAggregateMaxEvents, duplicateEvent),
			newEvent:        duplicateEvent,
			expectedEvent:   setCount(duplicateEvent, defaultAggregateMaxEvents+1),
			intervalSeconds: 5,
		},
		"create-many-unique-events": {
			previousEvents:  makeUniqueEvents(30),
			newEvent:        uniqueEvent,
			expectedEvent:   setCount(uniqueEvent, 1),
			intervalSeconds: 5,
		},
		"similar-events-should-aggregate-event": {
			previousEvents:  makeSimilarEvents(defaultAggregateMaxEvents-1, similarEvent, similarEvent.Message),
			newEvent:        similarEvent,
			expectedEvent:   setCount(aggregateEvent, 1),
			intervalSeconds: 5,
		},
		"similar-events-many-times-should-count-the-aggregate": {
			previousEvents:  makeSimilarEvents(defaultAggregateMaxEvents, similarEvent, similarEvent.Message),
			newEvent:        similarEvent,
			expectedEvent:   setCount(aggregateEvent, 2),
			intervalSeconds: 5,
		},
		"similar-events-whose-interval-is-greater-than-aggregate-interval-do-not-aggregate": {
			previousEvents:  makeSimilarEvents(defaultAggregateMaxEvents-1, similarEvent, similarEvent.Message),
			newEvent:        similarEvent,
			expectedEvent:   setCount(similarEvent, 1),
			intervalSeconds: defaultAggregateIntervalInSeconds,
		},
	}

	for testScenario, testInput := range scenario {
		eventInterval := time.Duration(testInput.intervalSeconds) * time.Second
		clock := clock.IntervalClock{Time: time.Now(), Duration: eventInterval}
		correlator := NewEventCorrelator(&clock)
		for i := range testInput.previousEvents {
			event := testInput.previousEvents[i]
			now := metav1.NewTime(clock.Now())
			event.FirstTimestamp = now
			event.LastTimestamp = now
			result, err := correlator.EventCorrelate(&event)
			if err != nil {
				t.Errorf("scenario %v: unexpected error playing back prevEvents %v", testScenario, err)
			}
			correlator.UpdateState(result.Event)
		}

		// update the input to current clock value
		now := metav1.NewTime(clock.Now())
		testInput.newEvent.FirstTimestamp = now
		testInput.newEvent.LastTimestamp = now
		result, err := correlator.EventCorrelate(&testInput.newEvent)
		if err != nil {
			t.Errorf("scenario %v: unexpected error correlating input event %v", testScenario, err)
		}

		_, err = validateEvent(testScenario, result.Event, &testInput.expectedEvent, t)
		if err != nil {
			t.Errorf("scenario %v: unexpected error validating result %v", testScenario, err)
		}
	}
}
