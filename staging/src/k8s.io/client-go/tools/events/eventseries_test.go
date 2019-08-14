/*
Copyright 2019 The Kubernetes Authors.

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
	"strconv"
	"testing"
	"time"

	"os"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/api/events/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/scheme"
	ref "k8s.io/client-go/tools/reference"
)

type testEventSeriesSink struct {
	OnCreate func(e *v1beta1.Event) (*v1beta1.Event, error)
	OnUpdate func(e *v1beta1.Event) (*v1beta1.Event, error)
	OnPatch  func(e *v1beta1.Event, p []byte) (*v1beta1.Event, error)
}

// Create records the event for testing.
func (t *testEventSeriesSink) Create(e *v1beta1.Event) (*v1beta1.Event, error) {
	if t.OnCreate != nil {
		return t.OnCreate(e)
	}
	return e, nil
}

// Update records the event for testing.
func (t *testEventSeriesSink) Update(e *v1beta1.Event) (*v1beta1.Event, error) {
	if t.OnUpdate != nil {
		return t.OnUpdate(e)
	}
	return e, nil
}

// Patch records the event for testing.
func (t *testEventSeriesSink) Patch(e *v1beta1.Event, p []byte) (*v1beta1.Event, error) {
	if t.OnPatch != nil {
		return t.OnPatch(e, p)
	}
	return e, nil
}

func TestEventSeriesf(t *testing.T) {
	hostname, _ := os.Hostname()

	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			SelfLink:  "/api/v1/namespaces/baz/pods/foo",
			Name:      "foo",
			Namespace: "baz",
			UID:       "bar",
		},
	}

	regarding, err := ref.GetPartialReference(scheme.Scheme, testPod, ".spec.containers[1]")
	if err != nil {
		t.Fatal(err)
	}

	related, err := ref.GetPartialReference(scheme.Scheme, testPod, ".spec.containers[0]")
	if err != nil {
		t.Fatal(err)
	}

	expectedEvent := &v1beta1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "baz",
		},
		EventTime:           metav1.MicroTime{time.Now()},
		ReportingController: "eventTest",
		ReportingInstance:   "eventTest-" + hostname,
		Action:              "started",
		Reason:              "test",
		Regarding:           *regarding,
		Related:             related,
		Note:                "some verbose message: 1",
		Type:                v1.EventTypeNormal,
	}

	isomorphicEvent := expectedEvent.DeepCopy()

	nonIsomorphicEvent := expectedEvent.DeepCopy()
	nonIsomorphicEvent.Action = "stopped"

	expectedEvent.Series = &v1beta1.EventSeries{Count: 1}
	table := []struct {
		regarding    k8sruntime.Object
		related      k8sruntime.Object
		actual       *v1beta1.Event
		elements     []interface{}
		expect       *v1beta1.Event
		expectUpdate bool
	}{
		{
			regarding:    regarding,
			related:      related,
			actual:       isomorphicEvent,
			elements:     []interface{}{1},
			expect:       expectedEvent,
			expectUpdate: true,
		},
		{
			regarding:    regarding,
			related:      related,
			actual:       nonIsomorphicEvent,
			elements:     []interface{}{1},
			expect:       nonIsomorphicEvent,
			expectUpdate: false,
		},
	}

	stopCh := make(chan struct{})

	createEvent := make(chan *v1beta1.Event)
	updateEvent := make(chan *v1beta1.Event)
	patchEvent := make(chan *v1beta1.Event)

	testEvents := testEventSeriesSink{
		OnCreate: func(event *v1beta1.Event) (*v1beta1.Event, error) {
			createEvent <- event
			return event, nil
		},
		OnUpdate: func(event *v1beta1.Event) (*v1beta1.Event, error) {
			updateEvent <- event
			return event, nil
		},
		OnPatch: func(event *v1beta1.Event, patch []byte) (*v1beta1.Event, error) {
			// event we receive is already patched, usually the sink uses it only to retrieve the name and namespace, here
			// we'll use it directly
			patchEvent <- event
			return event, nil
		},
	}
	eventBroadcaster := newBroadcaster(&testEvents, 0, map[eventKey]*v1beta1.Event{})
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "eventTest")
	eventBroadcaster.StartRecordingToSink(stopCh)
	recorder.Eventf(regarding, related, isomorphicEvent.Type, isomorphicEvent.Reason, isomorphicEvent.Action, isomorphicEvent.Note, []interface{}{1})
	// read from the chan as this was needed only to populate the cache
	<-createEvent
	for index, item := range table {
		actual := item.actual
		recorder.Eventf(item.regarding, item.related, actual.Type, actual.Reason, actual.Action, actual.Note, item.elements)
		// validate event
		if item.expectUpdate {
			actualEvent := <-patchEvent
			t.Logf("%v - validating event affected by patch request", index)
			validateEvent(strconv.Itoa(index), true, actualEvent, item.expect, t)
		} else {
			actualEvent := <-createEvent
			t.Logf("%v - validating event affected by a create request", index)
			validateEvent(strconv.Itoa(index), false, actualEvent, item.expect, t)
		}
	}
	close(stopCh)
}

func validateEvent(messagePrefix string, expectedUpdate bool, actualEvent *v1beta1.Event, expectedEvent *v1beta1.Event, t *testing.T) {
	recvEvent := *actualEvent

	// Just check that the timestamp was set.
	if recvEvent.EventTime.IsZero() {
		t.Errorf("%v - timestamp wasn't set: %#v", messagePrefix, recvEvent)
	}

	if expectedUpdate {
		if recvEvent.Series == nil {
			t.Errorf("%v - Series was nil but expected: %#v", messagePrefix, recvEvent.Series)

		} else {
			if recvEvent.Series.Count != expectedEvent.Series.Count {
				t.Errorf("%v - Series mismatch actual was: %#v but expected: %#v", messagePrefix, recvEvent.Series, expectedEvent.Series)
			}
		}

		// Check that name has the right prefix.
		if n, en := recvEvent.Name, expectedEvent.Name; !strings.HasPrefix(n, en) {
			t.Errorf("%v - Name '%v' does not contain prefix '%v'", messagePrefix, n, en)
		}
	} else {
		if recvEvent.Series != nil {
			t.Errorf("%v - series was expected to be nil but was: %#v", messagePrefix, recvEvent.Series)
		}
	}

}

func TestFinishSeries(t *testing.T) {
	hostname, _ := os.Hostname()
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			SelfLink:  "/api/v1/namespaces/baz/pods/foo",
			Name:      "foo",
			Namespace: "baz",
			UID:       "bar",
		},
	}
	regarding, err := ref.GetPartialReference(scheme.Scheme, testPod, ".spec.containers[1]")
	if err != nil {
		t.Fatal(err)
	}
	related, err := ref.GetPartialReference(scheme.Scheme, testPod, ".spec.containers[0]")
	if err != nil {
		t.Fatal(err)
	}
	LastObservedTime := metav1.MicroTime{Time: time.Now().Add(-9 * time.Minute)}

	createEvent := make(chan *v1beta1.Event, 10)
	updateEvent := make(chan *v1beta1.Event, 10)
	patchEvent := make(chan *v1beta1.Event, 10)
	testEvents := testEventSeriesSink{
		OnCreate: func(event *v1beta1.Event) (*v1beta1.Event, error) {
			createEvent <- event
			return event, nil
		},
		OnUpdate: func(event *v1beta1.Event) (*v1beta1.Event, error) {
			updateEvent <- event
			return event, nil
		},
		OnPatch: func(event *v1beta1.Event, patch []byte) (*v1beta1.Event, error) {
			// event we receive is already patched, usually the sink uses it
			// only to retrieve the name and namespace, here we'll use it directly
			patchEvent <- event
			return event, nil
		},
	}
	cache := map[eventKey]*v1beta1.Event{}
	eventBroadcaster := newBroadcaster(&testEvents, 0, cache).(*eventBroadcasterImpl)
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "k8s.io/kube-foo").(*recorderImpl)
	cachedEvent := recorder.makeEvent(regarding, related, metav1.MicroTime{time.Now()}, v1.EventTypeNormal, "test", "some verbose message: 1", "eventTest", "eventTest-"+hostname, "started")
	nonFinishedEvent := cachedEvent.DeepCopy()
	nonFinishedEvent.ReportingController = "nonFinished-controller"
	cachedEvent.Series = &v1beta1.EventSeries{
		Count:            10,
		LastObservedTime: LastObservedTime,
	}
	cache[getKey(cachedEvent)] = cachedEvent
	cache[getKey(nonFinishedEvent)] = nonFinishedEvent
	eventBroadcaster.finishSeries()
	select {
	case actualEvent := <-patchEvent:
		t.Logf("validating event affected by patch request")
		eventBroadcaster.mu.Lock()
		defer eventBroadcaster.mu.Unlock()
		if len(cache) != 1 {
			t.Errorf("cache should be empty, but instead got a size of %v", len(cache))
		}
		if !actualEvent.Series.LastObservedTime.Equal(&cachedEvent.Series.LastObservedTime) {
			t.Errorf("series was expected be seen with LastObservedTime %v, but instead got %v ", cachedEvent.Series.LastObservedTime, actualEvent.Series.LastObservedTime)
		}
		// check that we emitted only one event
		if len(patchEvent) != 0 || len(createEvent) != 0 || len(updateEvent) != 0 {
			t.Errorf("exactly one event should be emitted, but got %v", len(patchEvent))
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
}

func TestRefreshExistingEventSeries(t *testing.T) {
	hostname, _ := os.Hostname()
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			SelfLink:  "/api/v1/namespaces/baz/pods/foo",
			Name:      "foo",
			Namespace: "baz",
			UID:       "bar",
		},
	}
	regarding, err := ref.GetPartialReference(scheme.Scheme, testPod, ".spec.containers[1]")
	if err != nil {
		t.Fatal(err)
	}
	related, err := ref.GetPartialReference(scheme.Scheme, testPod, ".spec.containers[0]")
	if err != nil {
		t.Fatal(err)
	}
	LastObservedTime := metav1.MicroTime{Time: time.Now().Add(-9 * time.Minute)}

	createEvent := make(chan *v1beta1.Event, 10)
	updateEvent := make(chan *v1beta1.Event, 10)
	patchEvent := make(chan *v1beta1.Event, 10)
	testEvents := testEventSeriesSink{
		OnCreate: func(event *v1beta1.Event) (*v1beta1.Event, error) {
			createEvent <- event
			return event, nil
		},
		OnUpdate: func(event *v1beta1.Event) (*v1beta1.Event, error) {
			updateEvent <- event
			return event, nil
		},
		OnPatch: func(event *v1beta1.Event, patch []byte) (*v1beta1.Event, error) {
			// event we receive is already patched, usually the sink uses it
			//only to retrieve the name and namespace, here we'll use it directly.
			patchEvent <- event
			return event, nil
		},
	}
	cache := map[eventKey]*v1beta1.Event{}
	eventBroadcaster := newBroadcaster(&testEvents, 0, cache).(*eventBroadcasterImpl)
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "k8s.io/kube-foo").(*recorderImpl)
	cachedEvent := recorder.makeEvent(regarding, related, metav1.MicroTime{time.Now()}, v1.EventTypeNormal, "test", "some verbose message: 1", "eventTest", "eventTest-"+hostname, "started")
	cachedEvent.Series = &v1beta1.EventSeries{
		Count:            10,
		LastObservedTime: LastObservedTime,
	}
	cacheKey := getKey(cachedEvent)
	cache[cacheKey] = cachedEvent

	eventBroadcaster.refreshExistingEventSeries()
	select {
	case <-patchEvent:
		t.Logf("validating event affected by patch request")
		eventBroadcaster.mu.Lock()
		defer eventBroadcaster.mu.Unlock()
		if len(cache) != 1 {
			t.Errorf("cache should be with same size, but instead got a size of %v", len(cache))
		}
		// check that we emitted only one event
		if len(patchEvent) != 0 || len(createEvent) != 0 || len(updateEvent) != 0 {
			t.Errorf("exactly one event should be emitted, but got %v", len(patchEvent))
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
}
