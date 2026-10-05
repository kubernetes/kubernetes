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
	"context"
	"strconv"
	"testing"
	"time"

	"os"
	"strings"

	v1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	fake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	ref "k8s.io/client-go/tools/reference"
	"k8s.io/klog/v2/ktesting"
	testclocks "k8s.io/utils/clock/testing"
)

type testEventSeriesSink struct {
	OnCreate func(e *eventsv1.Event) (*eventsv1.Event, error)
	OnUpdate func(e *eventsv1.Event) (*eventsv1.Event, error)
	OnPatch  func(e *eventsv1.Event, p []byte) (*eventsv1.Event, error)
}

// Create records the event for testing.
func (t *testEventSeriesSink) Create(ctx context.Context, e *eventsv1.Event) (*eventsv1.Event, error) {
	if t.OnCreate != nil {
		return t.OnCreate(e)
	}
	return e, nil
}

// Update records the event for testing.
func (t *testEventSeriesSink) Update(ctx context.Context, e *eventsv1.Event) (*eventsv1.Event, error) {
	if t.OnUpdate != nil {
		return t.OnUpdate(e)
	}
	return e, nil
}

// Patch records the event for testing.
func (t *testEventSeriesSink) Patch(ctx context.Context, e *eventsv1.Event, p []byte) (*eventsv1.Event, error) {
	if t.OnPatch != nil {
		return t.OnPatch(e, p)
	}
	return e, nil
}

func TestEventSeriesf(t *testing.T) {
	hostname, _ := os.Hostname()

	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
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

	expectedEvent := &eventsv1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "baz",
		},
		EventTime:           metav1.MicroTime{Time: time.Now()},
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

	expectedEvent.Series = &eventsv1.EventSeries{Count: 2}
	table := []struct {
		regarding    k8sruntime.Object
		related      k8sruntime.Object
		actual       *eventsv1.Event
		elements     []interface{}
		expect       *eventsv1.Event
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

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	createEvent := make(chan *eventsv1.Event)
	updateEvent := make(chan *eventsv1.Event)
	patchEvent := make(chan *eventsv1.Event)

	testEvents := testEventSeriesSink{
		OnCreate: func(event *eventsv1.Event) (*eventsv1.Event, error) {
			createEvent <- event
			return event, nil
		},
		OnUpdate: func(event *eventsv1.Event) (*eventsv1.Event, error) {
			updateEvent <- event
			return event, nil
		},
		OnPatch: func(event *eventsv1.Event, patch []byte) (*eventsv1.Event, error) {
			// event we receive is already patched, usually the sink uses it only to retrieve the name and namespace, here
			// we'll use it directly
			patchEvent <- event
			return event, nil
		},
	}
	eventBroadcaster := newBroadcaster(&testEvents, 0, map[eventKey]*eventsv1.Event{})
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "eventTest")
	broadcaster := eventBroadcaster.(*eventBroadcasterImpl)
	// Don't call StartRecordingToSink, as we don't need neither refreshing event
	// series nor finishing them in this tests and additional events updated would
	// race with our expected ones.
	err = broadcaster.startRecordingEvents(ctx)
	if err != nil {
		t.Fatal(err)
	}
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
}

// TestEventSeriesWithEventSinkImplRace verifies that when Events are emitted to
// an EventSink consecutively there is no data race.  This test is meant to be
// run with the `-race` option.
func TestEventSeriesWithEventSinkImplRace(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()

	eventSink := &EventSinkImpl{Interface: kubeClient.EventsV1()}
	eventBroadcaster := NewBroadcaster(eventSink)

	stopCh := make(chan struct{})
	eventBroadcaster.StartRecordingToSink(stopCh)

	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "test")

	recorder.Eventf(&v1.ObjectReference{}, nil, v1.EventTypeNormal, "reason", "action", "", "")
	recorder.Eventf(&v1.ObjectReference{}, nil, v1.EventTypeNormal, "reason", "action", "", "")

	err := wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (done bool, err error) {
		events, err := kubeClient.EventsV1().Events(metav1.NamespaceDefault).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
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
		t.Fatal("expected that 2 identical Eventf calls would result in the creation of an Event with a Serie")
	}
}

func validateEvent(messagePrefix string, expectedUpdate bool, actualEvent *eventsv1.Event, expectedEvent *eventsv1.Event, t *testing.T) {
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
	_, ctx := ktesting.NewTestContext(t)
	hostname, _ := os.Hostname()
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
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

	createEvent := make(chan *eventsv1.Event, 10)
	updateEvent := make(chan *eventsv1.Event, 10)
	patchEvent := make(chan *eventsv1.Event, 10)
	testEvents := testEventSeriesSink{
		OnCreate: func(event *eventsv1.Event) (*eventsv1.Event, error) {
			createEvent <- event
			return event, nil
		},
		OnUpdate: func(event *eventsv1.Event) (*eventsv1.Event, error) {
			updateEvent <- event
			return event, nil
		},
		OnPatch: func(event *eventsv1.Event, patch []byte) (*eventsv1.Event, error) {
			// event we receive is already patched, usually the sink uses it
			// only to retrieve the name and namespace, here we'll use it directly
			patchEvent <- event
			return event, nil
		},
	}
	cache := map[eventKey]*eventsv1.Event{}
	eventBroadcaster := newBroadcaster(&testEvents, 0, cache).(*eventBroadcasterImpl)
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "k8s.io/kube-foo").(*recorderImplLogger)
	cachedEvent := recorder.makeEvent(regarding, related, metav1.MicroTime{Time: time.Now()}, nil, v1.EventTypeNormal, "test", "some verbose message: 1", "eventTest", "eventTest-"+hostname, "started")
	nonFinishedEvent := cachedEvent.DeepCopy()
	nonFinishedEvent.ReportingController = "nonFinished-controller"
	cachedEvent.Series = &eventsv1.EventSeries{
		Count:            10,
		LastObservedTime: LastObservedTime,
	}
	cache[getKey(cachedEvent)] = cachedEvent
	cache[getKey(nonFinishedEvent)] = nonFinishedEvent
	eventBroadcaster.finishSeries(ctx)
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
	_, ctx := ktesting.NewTestContext(t)
	hostname, _ := os.Hostname()
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
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
	createEvent := make(chan *eventsv1.Event, 10)
	updateEvent := make(chan *eventsv1.Event, 10)
	patchEvent := make(chan *eventsv1.Event, 10)

	table := []struct {
		patchFunc func(event *eventsv1.Event, patch []byte) (*eventsv1.Event, error)
	}{
		{
			patchFunc: func(event *eventsv1.Event, patch []byte) (*eventsv1.Event, error) {
				// event we receive is already patched, usually the sink uses it
				//only to retrieve the name and namespace, here we'll use it directly.
				patchEvent <- event
				return event, nil
			},
		},
		{
			patchFunc: func(event *eventsv1.Event, patch []byte) (*eventsv1.Event, error) {
				// we simulate an apiserver error here
				patchEvent <- nil
				return nil, &restclient.RequestConstructionError{}
			},
		},
	}
	for _, item := range table {
		testEvents := testEventSeriesSink{
			OnCreate: func(event *eventsv1.Event) (*eventsv1.Event, error) {
				createEvent <- event
				return event, nil
			},
			OnUpdate: func(event *eventsv1.Event) (*eventsv1.Event, error) {
				updateEvent <- event
				return event, nil
			},
			OnPatch: item.patchFunc,
		}
		cache := map[eventKey]*eventsv1.Event{}
		eventBroadcaster := newBroadcaster(&testEvents, 0, cache).(*eventBroadcasterImpl)
		recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "k8s.io/kube-foo").(*recorderImplLogger)
		cachedEvent := recorder.makeEvent(regarding, related, metav1.MicroTime{Time: time.Now()}, nil, v1.EventTypeNormal, "test", "some verbose message: 1", "eventTest", "eventTest-"+hostname, "started")
		cachedEvent.Series = &eventsv1.EventSeries{
			Count:            10,
			LastObservedTime: LastObservedTime,
		}
		cacheKey := getKey(cachedEvent)
		cache[cacheKey] = cachedEvent

		eventBroadcaster.refreshExistingEventSeries(ctx)
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
			cacheEvent, exists := cache[cacheKey]

			if cacheEvent == nil || !exists {
				t.Errorf("expected event to exist and not being nil, but instead event: %v and exists: %v", cacheEvent, exists)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
		}
	}
}

// newCachedSeriesEvent creates an Event with a Series and adds it to the
// broadcaster's cache, returning the Event and its cache key.
func newCachedSeriesEvent(t *testing.T, eventBroadcaster *eventBroadcasterImpl, lastObservedTime time.Time) (*eventsv1.Event, eventKey) {
	t.Helper()
	hostname, _ := os.Hostname()
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
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
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "k8s.io/kube-foo").(*recorderImplLogger)
	cachedEvent := recorder.makeEvent(regarding, related, metav1.MicroTime{Time: time.Now()}, nil, v1.EventTypeNormal, "test", "some verbose message: 1", "eventTest", "eventTest-"+hostname, "started")
	cachedEvent.Series = &eventsv1.EventSeries{
		Count:            10,
		LastObservedTime: metav1.MicroTime{Time: lastObservedTime},
	}
	key := getKey(cachedEvent)
	eventBroadcaster.mu.Lock()
	defer eventBroadcaster.mu.Unlock()
	eventBroadcaster.eventCache[key] = cachedEvent
	return cachedEvent, key
}

// TestRefreshExistingEventSeriesConcurrentCacheUpdate verifies that
// refreshExistingEventSeries only updates a cache entry that still belongs to
// the same series after the lock was released for the API call, and that it
// keeps the most recent observations recorded while the call was in flight.
func TestRefreshExistingEventSeriesConcurrentCacheUpdate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	observedAgainTime := metav1.MicroTime{Time: time.Now().Add(time.Minute)}

	table := []struct {
		name string
		// mutate runs while the API call is in flight, i.e. with the
		// broadcaster's lock released, and simulates a concurrent recorder.
		mutate func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event)
		verify func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event)
	}{
		{
			name: "cache entry replaced by a new isomorphic series is not overwritten",
			mutate: func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				replacement := original.DeepCopy()
				replacement.Name = original.Name + "-replacement"
				replacement.ResourceVersion = ""
				replacement.Series = &eventsv1.EventSeries{Count: 2, LastObservedTime: observedAgainTime}
				e.eventCache[key] = replacement
			},
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				cached, exists := e.eventCache[key]
				if !exists {
					t.Fatal("expected the replacement series to remain in the cache")
				}
				if cached.Name != original.Name+"-replacement" {
					t.Errorf("expected cache to hold the replacement series %q, but got %q", original.Name+"-replacement", cached.Name)
				}
				if cached.ResourceVersion != "" {
					t.Errorf("expected the replacement series to be untouched, but got ResourceVersion %q from the recorded event", cached.ResourceVersion)
				}
				if cached.Series == nil || cached.Series.Count != 2 {
					t.Errorf("expected the replacement series to keep Count 2, but got %v", cached.Series)
				}
			},
		},
		{
			name: "cache entry observed again keeps the most recent observations",
			mutate: func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				e.eventCache[key].Series.Count = 11
				e.eventCache[key].Series.LastObservedTime = observedAgainTime
			},
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				cached, exists := e.eventCache[key]
				if !exists {
					t.Fatal("expected the series to remain in the cache")
				}
				if cached.Name != original.Name {
					t.Errorf("expected cache to hold series %q, but got %q", original.Name, cached.Name)
				}
				if cached.ResourceVersion != "recorded" {
					t.Errorf("expected cache to be updated with the recorded event, but got ResourceVersion %q", cached.ResourceVersion)
				}
				if cached.Series == nil || cached.Series.Count != 11 {
					t.Errorf("expected the refreshed series to keep Count 11, but got %v", cached.Series)
				}
				if cached.Series != nil && !cached.Series.LastObservedTime.Equal(&observedAgainTime) {
					t.Errorf("expected the refreshed series to keep LastObservedTime %v, but got %v", observedAgainTime, cached.Series.LastObservedTime)
				}
			},
		},
		{
			name: "cache entry finished while refreshing is not re-added",
			mutate: func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				delete(e.eventCache, key)
			},
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				if cached, exists := e.eventCache[key]; exists {
					t.Errorf("expected the finished series to stay out of the cache, but got %v", cached)
				}
			},
		},
		{
			name: "cache entry replaced by a singleton event is not overwritten",
			mutate: func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				replacement := original.DeepCopy()
				replacement.Name = original.Name + "-singleton"
				replacement.Series = nil
				e.eventCache[key] = replacement
			},
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				cached, exists := e.eventCache[key]
				if !exists {
					t.Fatal("expected the singleton event to remain in the cache")
				}
				if cached.Name != original.Name+"-singleton" || cached.Series != nil {
					t.Errorf("expected cache to hold the untouched singleton event, but got %v", cached)
				}
			},
		},
	}
	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			var eventBroadcaster *eventBroadcasterImpl
			var original *eventsv1.Event
			var key eventKey
			patches := 0
			testEvents := testEventSeriesSink{
				OnPatch: func(event *eventsv1.Event, patch []byte) (*eventsv1.Event, error) {
					patches++
					if event.Name != original.Name {
						t.Errorf("expected to patch series %q, but got %q", original.Name, event.Name)
					}
					eventBroadcaster.mu.Lock()
					defer eventBroadcaster.mu.Unlock()
					item.mutate(eventBroadcaster, key, original)
					recorded := event.DeepCopy()
					recorded.ResourceVersion = "recorded"
					return recorded, nil
				},
			}
			eventBroadcaster = newBroadcaster(&testEvents, 0, map[eventKey]*eventsv1.Event{}).(*eventBroadcasterImpl)
			original, key = newCachedSeriesEvent(t, eventBroadcaster, time.Now())

			eventBroadcaster.refreshExistingEventSeries(ctx)

			if patches != 1 {
				t.Errorf("expected exactly one patch request, but got %d", patches)
			}
			eventBroadcaster.mu.Lock()
			defer eventBroadcaster.mu.Unlock()
			if len(eventBroadcaster.eventCache) > 1 {
				t.Errorf("expected at most one cache entry, but got %d", len(eventBroadcaster.eventCache))
			}
			item.verify(t, eventBroadcaster, key, original)
		})
	}
}

// TestFinishSeriesConcurrentCacheUpdate verifies that finishSeries only
// deletes a cache entry that still belongs to the same series and was not
// observed again after the lock was released for the API call.
func TestFinishSeriesConcurrentCacheUpdate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	observedAgainTime := metav1.MicroTime{Time: time.Now()}

	table := []struct {
		name string
		// mutate runs while the API call is in flight, i.e. with the
		// broadcaster's lock released, and simulates a concurrent recorder.
		mutate func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event)
		verify func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event)
	}{
		{
			name: "unchanged series is deleted",
			mutate: func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
			},
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				if cached, exists := e.eventCache[key]; exists {
					t.Errorf("expected the finished series to be deleted from the cache, but got %v", cached)
				}
			},
		},
		{
			name: "cache entry replaced by a new isomorphic series is not deleted",
			mutate: func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				replacement := original.DeepCopy()
				replacement.Name = original.Name + "-replacement"
				replacement.Series = &eventsv1.EventSeries{Count: 2, LastObservedTime: observedAgainTime}
				e.eventCache[key] = replacement
			},
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				cached, exists := e.eventCache[key]
				if !exists {
					t.Fatal("expected the replacement series to remain in the cache")
				}
				if cached.Name != original.Name+"-replacement" {
					t.Errorf("expected cache to hold the replacement series %q, but got %q", original.Name+"-replacement", cached.Name)
				}
				if cached.Series == nil || cached.Series.Count != 2 {
					t.Errorf("expected the replacement series to keep Count 2, but got %v", cached.Series)
				}
			},
		},
		{
			name: "cache entry observed again is not deleted",
			mutate: func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				e.eventCache[key].Series.Count = 11
				e.eventCache[key].Series.LastObservedTime = observedAgainTime
			},
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				cached, exists := e.eventCache[key]
				if !exists {
					t.Fatal("expected the series observed again to remain in the cache")
				}
				if cached.Name != original.Name {
					t.Errorf("expected cache to hold series %q, but got %q", original.Name, cached.Name)
				}
				if cached.Series == nil || cached.Series.Count != 11 {
					t.Errorf("expected the series to keep Count 11, but got %v", cached.Series)
				}
			},
		},
		{
			name: "cache entry replaced by a singleton event is not deleted",
			mutate: func(e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				replacement := original.DeepCopy()
				replacement.Name = original.Name + "-singleton"
				replacement.Series = nil
				e.eventCache[key] = replacement
			},
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				cached, exists := e.eventCache[key]
				if !exists {
					t.Fatal("expected the singleton event to remain in the cache")
				}
				if cached.Name != original.Name+"-singleton" || cached.Series != nil {
					t.Errorf("expected cache to hold the untouched singleton event, but got %v", cached)
				}
			},
		},
	}
	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			var eventBroadcaster *eventBroadcasterImpl
			var original *eventsv1.Event
			var key eventKey
			patches := 0
			testEvents := testEventSeriesSink{
				OnPatch: func(event *eventsv1.Event, patch []byte) (*eventsv1.Event, error) {
					patches++
					if event.Name != original.Name {
						t.Errorf("expected to patch series %q, but got %q", original.Name, event.Name)
					}
					if event.Series == nil || event.Series.Count != 10 {
						t.Errorf("expected to record the final Count 10, but got %v", event.Series)
					}
					eventBroadcaster.mu.Lock()
					defer eventBroadcaster.mu.Unlock()
					item.mutate(eventBroadcaster, key, original)
					return event, nil
				},
			}
			eventBroadcaster = newBroadcaster(&testEvents, 0, map[eventKey]*eventsv1.Event{}).(*eventBroadcasterImpl)
			original, key = newCachedSeriesEvent(t, eventBroadcaster, time.Now().Add(-finishTime-time.Minute))

			eventBroadcaster.finishSeries(ctx)

			if patches != 1 {
				t.Errorf("expected exactly one patch request, but got %d", patches)
			}
			eventBroadcaster.mu.Lock()
			defer eventBroadcaster.mu.Unlock()
			if len(eventBroadcaster.eventCache) > 1 {
				t.Errorf("expected at most one cache entry, but got %d", len(eventBroadcaster.eventCache))
			}
			item.verify(t, eventBroadcaster, key, original)
		})
	}
}

// TestSeriesHousekeepingWithConcurrentRecordToSink verifies that events
// recorded through recordToSink while finishSeries or
// refreshExistingEventSeries are in the middle of an API call are neither
// blocked by that call nor lost: an isomorphic event increments the cached
// series and prevents finishSeries from deleting it, and a non-isomorphic
// event is recorded right away.
func TestSeriesHousekeepingWithConcurrentRecordToSink(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	observedAgainTime := time.Now().Add(time.Minute)

	table := []struct {
		name             string
		lastObservedTime time.Time
		housekeeping     func(e *eventBroadcasterImpl, ctx context.Context)
		verify           func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event)
	}{
		{
			name:             "finishSeries does not delete a series observed while patching",
			lastObservedTime: time.Now().Add(-finishTime - time.Minute),
			housekeeping:     (*eventBroadcasterImpl).finishSeries,
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				cached, exists := e.eventCache[key]
				if !exists {
					t.Fatal("expected the series observed while patching to remain in the cache")
				}
				if cached.Name != original.Name {
					t.Errorf("expected cache to hold series %q, but got %q", original.Name, cached.Name)
				}
				if cached.Series == nil || cached.Series.Count != 11 {
					t.Errorf("expected the series to have Count 11, but got %v", cached.Series)
				}
			},
		},
		{
			name:             "refreshExistingEventSeries keeps observations recorded while patching",
			lastObservedTime: time.Now(),
			housekeeping:     (*eventBroadcasterImpl).refreshExistingEventSeries,
			verify: func(t *testing.T, e *eventBroadcasterImpl, key eventKey, original *eventsv1.Event) {
				cached, exists := e.eventCache[key]
				if !exists {
					t.Fatal("expected the series to remain in the cache")
				}
				if cached.Name != original.Name {
					t.Errorf("expected cache to hold series %q, but got %q", original.Name, cached.Name)
				}
				if cached.ResourceVersion != "recorded" {
					t.Errorf("expected cache to be updated with the recorded event, but got ResourceVersion %q", cached.ResourceVersion)
				}
				if cached.Series == nil || cached.Series.Count != 11 {
					t.Errorf("expected the refreshed series to have Count 11, but got %v", cached.Series)
				}
				if cached.Series != nil && !cached.Series.LastObservedTime.Time.Equal(observedAgainTime) {
					t.Errorf("expected the refreshed series to keep LastObservedTime %v, but got %v", observedAgainTime, cached.Series.LastObservedTime)
				}
			},
		},
	}
	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			patchStarted := make(chan struct{})
			releasePatch := make(chan struct{})
			createdEvent := make(chan *eventsv1.Event, 1)
			patches := 0
			testEvents := testEventSeriesSink{
				OnCreate: func(event *eventsv1.Event) (*eventsv1.Event, error) {
					createdEvent <- event
					return event, nil
				},
				OnPatch: func(event *eventsv1.Event, patch []byte) (*eventsv1.Event, error) {
					patches++
					close(patchStarted)
					<-releasePatch
					recorded := event.DeepCopy()
					recorded.ResourceVersion = "recorded"
					return recorded, nil
				},
			}
			eventBroadcaster := newBroadcaster(&testEvents, 0, map[eventKey]*eventsv1.Event{}).(*eventBroadcasterImpl)
			original, key := newCachedSeriesEvent(t, eventBroadcaster, item.lastObservedTime)

			housekeepingDone := make(chan struct{})
			go func() {
				defer close(housekeepingDone)
				item.housekeeping(eventBroadcaster, ctx)
			}()
			select {
			case <-patchStarted:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("timeout after %v waiting for the patch request", wait.ForeverTestTimeout)
			}

			// While the API call is in flight, record an isomorphic event and
			// an unrelated event. Neither must wait for the API call to end.
			isomorphicEvent := original.DeepCopy()
			isomorphicEvent.Name = original.Name + "-isomorphic"
			isomorphicEvent.Series = nil
			otherEvent := original.DeepCopy()
			otherEvent.Name = original.Name + "-other"
			otherEvent.Reason = "other"
			otherEvent.Series = nil
			recordingDone := make(chan struct{})
			go func() {
				defer close(recordingDone)
				eventBroadcaster.recordToSink(ctx, isomorphicEvent, testclocks.NewFakeClock(observedAgainTime))
				eventBroadcaster.recordToSink(ctx, otherEvent, testclocks.NewFakeClock(observedAgainTime))
			}()
			select {
			case <-recordingDone:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("timeout after %v: recordToSink was blocked by the in-flight API call", wait.ForeverTestTimeout)
			}
			select {
			case created := <-createdEvent:
				if created.Name != otherEvent.Name {
					t.Errorf("expected the unrelated event %q to be created, but got %q", otherEvent.Name, created.Name)
				}
			default:
				t.Error("expected the unrelated event to be created while the API call was in flight")
			}

			close(releasePatch)
			select {
			case <-housekeepingDone:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("timeout after %v waiting for housekeeping to finish", wait.ForeverTestTimeout)
			}

			eventBroadcaster.mu.Lock()
			defer eventBroadcaster.mu.Unlock()
			if patches != 1 {
				t.Errorf("expected exactly one patch request, but got %d", patches)
			}
			if len(createdEvent) != 0 {
				t.Errorf("expected exactly one create request, but got %d more", len(createdEvent))
			}
			if len(eventBroadcaster.eventCache) != 2 {
				t.Errorf("expected the series and the unrelated event in the cache, but got %d entries", len(eventBroadcaster.eventCache))
			}
			otherCached, exists := eventBroadcaster.eventCache[getKey(otherEvent)]
			if !exists || otherCached.Series != nil {
				t.Errorf("expected the unrelated event to be cached as a singleton, but got %v", otherCached)
			}
			item.verify(t, eventBroadcaster, key, original)
		})
	}
}
