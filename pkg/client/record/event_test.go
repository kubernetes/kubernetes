/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
)

func init() {
	// Don't bother sleeping between retries.
	sleepDuration = 0
}

type testEventSink struct {
	OnCreate func(e *api.Event) (*api.Event, error)
	OnUpdate func(e *api.Event) (*api.Event, error)
	OnPatch  func(e *api.Event, p []byte) (*api.Event, error)
}

// CreateEvent records the event for testing.
func (t *testEventSink) Create(e *api.Event) (*api.Event, error) {
	if t.OnCreate != nil {
		return t.OnCreate(e)
	}
	return e, nil
}

// UpdateEvent records the event for testing.
func (t *testEventSink) Update(e *api.Event) (*api.Event, error) {
	if t.OnUpdate != nil {
		return t.OnUpdate(e)
	}
	return e, nil
}

// PatchEvent records the event for testing.
func (t *testEventSink) Patch(e *api.Event, p []byte) (*api.Event, error) {
	if t.OnPatch != nil {
		return t.OnPatch(e, p)
	}
	return e, nil
}

type OnCreateFunc func(*api.Event) (*api.Event, error)

func OnCreateFactory(testCache map[string]*api.Event, createEvent chan<- *api.Event) OnCreateFunc {
	return func(event *api.Event) (*api.Event, error) {
		testCache[getEventKey(event)] = event
		createEvent <- event
		return event, nil
	}
}

type OnPatchFunc func(*api.Event, []byte) (*api.Event, error)

func OnPatchFactory(testCache map[string]*api.Event, patchEvent chan<- *api.Event) OnPatchFunc {
	return func(event *api.Event, patch []byte) (*api.Event, error) {
		cachedEvent, found := testCache[getEventKey(event)]
		if !found {
			return nil, fmt.Errorf("unexpected error: couldn't find Event in testCache. Try to find Event: %v", event)
		}
		originalData, err := json.Marshal(cachedEvent)
		if err != nil {
			return nil, fmt.Errorf("unexpected error: %v", err)
		}
		patched, err := strategicpatch.StrategicMergePatch(originalData, patch, event)
		if err != nil {
			return nil, fmt.Errorf("unexpected error: %v", err)
		}
		patchedObj := &api.Event{}
		err = json.Unmarshal(patched, patchedObj)
		if err != nil {
			return nil, fmt.Errorf("unexpected error: %v", err)
		}
		patchEvent <- patchedObj
		return patchedObj, nil
	}
}

func TestEventf(t *testing.T) {
	testPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			SelfLink:  "/api/version/pods/foo",
			Name:      "foo",
			Namespace: "baz",
			UID:       "bar",
		},
	}
	testPod2 := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			SelfLink:  "/api/version/pods/foo",
			Name:      "foo",
			Namespace: "baz",
			UID:       "differentUid",
		},
	}
	testRef, err := api.GetPartialReference(testPod, "spec.containers[2]")
	testRef2, err := api.GetPartialReference(testPod2, "spec.containers[3]")
	if err != nil {
		t.Fatal(err)
	}
	table := []struct {
		obj          runtime.Object
		reason       string
		messageFmt   string
		elements     []interface{}
		expect       *api.Event
		expectLog    string
		expectUpdate bool
	}{
		{
			obj:        testRef,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "bar",
					APIVersion: "version",
					FieldPath:  "spec.containers[2]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[2]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: false,
		},
		{
			obj:        testPod,
			reason:     "Killed",
			messageFmt: "some other verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "bar",
					APIVersion: "version",
				},
				Reason:  "Killed",
				Message: "some other verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:""}): reason: 'Killed' some other verbose message: 1`,
			expectUpdate: false,
		},
		{
			obj:        testRef,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "bar",
					APIVersion: "version",
					FieldPath:  "spec.containers[2]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   2,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[2]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: true,
		},
		{
			obj:        testRef2,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "differentUid",
					APIVersion: "version",
					FieldPath:  "spec.containers[3]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"differentUid", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[3]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: false,
		},
		{
			obj:        testRef,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "bar",
					APIVersion: "version",
					FieldPath:  "spec.containers[2]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   3,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[2]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: true,
		},
		{
			obj:        testRef2,
			reason:     "Stopped",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "differentUid",
					APIVersion: "version",
					FieldPath:  "spec.containers[3]",
				},
				Reason:  "Stopped",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"differentUid", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[3]"}): reason: 'Stopped' some verbose message: 1`,
			expectUpdate: false,
		},
		{
			obj:        testRef2,
			reason:     "Stopped",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "differentUid",
					APIVersion: "version",
					FieldPath:  "spec.containers[3]",
				},
				Reason:  "Stopped",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   2,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"differentUid", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[3]"}): reason: 'Stopped' some verbose message: 1`,
			expectUpdate: true,
		},
	}

	testCache := map[string]*api.Event{}
	logCalled := make(chan struct{})
	createEvent := make(chan *api.Event)
	updateEvent := make(chan *api.Event)
	patchEvent := make(chan *api.Event)
	testEvents := testEventSink{
		OnCreate: OnCreateFactory(testCache, createEvent),
		OnUpdate: func(event *api.Event) (*api.Event, error) {
			updateEvent <- event
			return event, nil
		},
		OnPatch: OnPatchFactory(testCache, patchEvent),
	}
	eventBroadcaster := NewBroadcaster()
	sinkWatcher := eventBroadcaster.StartRecordingToSink(&testEvents)

	clock := &util.FakeClock{time.Now()}
	recorder := recorderWithFakeClock(api.EventSource{Component: "eventTest"}, eventBroadcaster, clock)
	for _, item := range table {
		clock.Step(1 * time.Second)
		logWatcher1 := eventBroadcaster.StartLogging(t.Logf) // Prove that it is useful
		logWatcher2 := eventBroadcaster.StartLogging(func(formatter string, args ...interface{}) {
			if e, a := item.expectLog, fmt.Sprintf(formatter, args...); e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			logCalled <- struct{}{}
		})
		recorder.Eventf(item.obj, item.reason, item.messageFmt, item.elements...)

		<-logCalled

		// validate event
		if item.expectUpdate {
			actualEvent := <-patchEvent
			validateEvent(actualEvent, item.expect, t)
		} else {
			actualEvent := <-createEvent
			validateEvent(actualEvent, item.expect, t)
		}
		logWatcher1.Stop()
		logWatcher2.Stop()
	}
	sinkWatcher.Stop()
}

func validateEvent(actualEvent *api.Event, expectedEvent *api.Event, t *testing.T) (*api.Event, error) {
	recvEvent := *actualEvent
	expectCompression := expectedEvent.Count > 1
	t.Logf("expectedEvent.Count is %d\n", expectedEvent.Count)
	// Just check that the timestamp was set.
	if recvEvent.FirstTimestamp.IsZero() || recvEvent.LastTimestamp.IsZero() {
		t.Errorf("timestamp wasn't set: %#v", recvEvent)
	}
	actualFirstTimestamp := recvEvent.FirstTimestamp
	actualLastTimestamp := recvEvent.LastTimestamp
	if actualFirstTimestamp.Equal(actualLastTimestamp) {
		if expectCompression {
			t.Errorf("FirstTimestamp (%q) and LastTimestamp (%q) must be different to indicate event compression happened, but were the same. Actual Event: %#v", actualFirstTimestamp, actualLastTimestamp, recvEvent)
		}
	} else {
		if expectedEvent.Count == 1 {
			t.Errorf("FirstTimestamp (%q) and LastTimestamp (%q) must be equal to indicate only one occurrence of the event, but were different. Actual Event: %#v", actualFirstTimestamp, actualLastTimestamp, recvEvent)
		}
	}
	// Temp clear time stamps for comparison because actual values don't matter for comparison
	recvEvent.FirstTimestamp = expectedEvent.FirstTimestamp
	recvEvent.LastTimestamp = expectedEvent.LastTimestamp
	// Check that name has the right prefix.
	if n, en := recvEvent.Name, expectedEvent.Name; !strings.HasPrefix(n, en) {
		t.Errorf("Name '%v' does not contain prefix '%v'", n, en)
	}
	recvEvent.Name = expectedEvent.Name
	if e, a := expectedEvent, &recvEvent; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectGoPrintDiff(e, a))
	}
	recvEvent.FirstTimestamp = actualFirstTimestamp
	recvEvent.LastTimestamp = actualLastTimestamp
	return actualEvent, nil
}

func recorderWithFakeClock(eventSource api.EventSource, eventBroadcaster EventBroadcaster, clock util.Clock) EventRecorder {
	return &recorderImpl{eventSource, eventBroadcaster.(*eventBroadcasterImpl).Broadcaster, clock}
}

func TestWriteEventError(t *testing.T) {
	ref := &api.ObjectReference{
		Kind:       "Pod",
		Name:       "foo",
		Namespace:  "baz",
		UID:        "bar",
		APIVersion: "version",
	}
	type entry struct {
		timesToSendError int
		attemptsMade     int
		attemptsWanted   int
		err              error
	}
	table := map[string]*entry{
		"giveUp1": {
			timesToSendError: 1000,
			attemptsWanted:   1,
			err:              &client.RequestConstructionError{},
		},
		"giveUp2": {
			timesToSendError: 1000,
			attemptsWanted:   1,
			err:              &errors.StatusError{},
		},
		"retry1": {
			timesToSendError: 1000,
			attemptsWanted:   12,
			err:              &errors.UnexpectedObjectError{},
		},
		"retry2": {
			timesToSendError: 1000,
			attemptsWanted:   12,
			err:              fmt.Errorf("A weird error"),
		},
		"succeedEventually": {
			timesToSendError: 2,
			attemptsWanted:   2,
			err:              fmt.Errorf("A weird error"),
		},
	}
	done := make(chan struct{})

	eventBroadcaster := NewBroadcaster()
	defer eventBroadcaster.StartRecordingToSink(
		&testEventSink{
			OnCreate: func(event *api.Event) (*api.Event, error) {
				if event.Message == "finished" {
					close(done)
					return event, nil
				}
				item, ok := table[event.Message]
				if !ok {
					t.Errorf("Unexpected event: %#v", event)
					return event, nil
				}
				item.attemptsMade++
				if item.attemptsMade < item.timesToSendError {
					return nil, item.err
				}
				return event, nil
			},
		},
	).Stop()
	clock := &util.FakeClock{time.Now()}
	recorder := recorderWithFakeClock(api.EventSource{Component: "eventTest"}, eventBroadcaster, clock)
	for caseName := range table {
		clock.Step(1 * time.Second)
		recorder.Event(ref, "Reason", caseName)
	}
	recorder.Event(ref, "Reason", "finished")
	<-done

	for caseName, item := range table {
		if e, a := item.attemptsWanted, item.attemptsMade; e != a {
			t.Errorf("case %v: wanted %v, got %v attempts", caseName, e, a)
		}
	}
}

func TestLotsOfEvents(t *testing.T) {
	recorderCalled := make(chan struct{})
	loggerCalled := make(chan struct{})

	// Fail each event a few times to ensure there's some load on the tested code.
	var counts [1000]int
	testEvents := testEventSink{
		OnCreate: func(event *api.Event) (*api.Event, error) {
			num, err := strconv.Atoi(event.Message)
			if err != nil {
				t.Error(err)
				return event, nil
			}
			counts[num]++
			if counts[num] < 5 {
				return nil, fmt.Errorf("fake error")
			}
			recorderCalled <- struct{}{}
			return event, nil
		},
	}

	eventBroadcaster := NewBroadcaster()
	sinkWatcher := eventBroadcaster.StartRecordingToSink(&testEvents)
	logWatcher := eventBroadcaster.StartLogging(func(formatter string, args ...interface{}) {
		loggerCalled <- struct{}{}
	})
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "eventTest"})
	ref := &api.ObjectReference{
		Kind:       "Pod",
		Name:       "foo",
		Namespace:  "baz",
		UID:        "bar",
		APIVersion: "version",
	}
	for i := 0; i < maxQueuedEvents; i++ {
		go recorder.Eventf(ref, "Reason", strconv.Itoa(i))
	}
	// Make sure no events were dropped by either of the listeners.
	for i := 0; i < maxQueuedEvents; i++ {
		<-recorderCalled
		<-loggerCalled
	}
	// Make sure that every event was attempted 5 times
	for i := 0; i < maxQueuedEvents; i++ {
		if counts[i] < 5 {
			t.Errorf("Only attempted to record event '%d' %d times.", i, counts[i])
		}
	}
	sinkWatcher.Stop()
	logWatcher.Stop()
}

func TestEventfNoNamespace(t *testing.T) {
	testPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			SelfLink: "/api/version/pods/foo",
			Name:     "foo",
			UID:      "bar",
		},
	}
	testRef, err := api.GetPartialReference(testPod, "spec.containers[2]")
	if err != nil {
		t.Fatal(err)
	}
	table := []struct {
		obj          runtime.Object
		reason       string
		messageFmt   string
		elements     []interface{}
		expect       *api.Event
		expectLog    string
		expectUpdate bool
	}{
		{
			obj:        testRef,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "default",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "",
					UID:        "bar",
					APIVersion: "version",
					FieldPath:  "spec.containers[2]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[2]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: false,
		},
	}

	testCache := map[string]*api.Event{}
	logCalled := make(chan struct{})
	createEvent := make(chan *api.Event)
	updateEvent := make(chan *api.Event)
	patchEvent := make(chan *api.Event)
	testEvents := testEventSink{
		OnCreate: OnCreateFactory(testCache, createEvent),
		OnUpdate: func(event *api.Event) (*api.Event, error) {
			updateEvent <- event
			return event, nil
		},
		OnPatch: OnPatchFactory(testCache, patchEvent),
	}
	eventBroadcaster := NewBroadcaster()
	sinkWatcher := eventBroadcaster.StartRecordingToSink(&testEvents)

	clock := &util.FakeClock{time.Now()}
	recorder := recorderWithFakeClock(api.EventSource{Component: "eventTest"}, eventBroadcaster, clock)

	for _, item := range table {
		clock.Step(1 * time.Second)
		logWatcher1 := eventBroadcaster.StartLogging(t.Logf) // Prove that it is useful
		logWatcher2 := eventBroadcaster.StartLogging(func(formatter string, args ...interface{}) {
			if e, a := item.expectLog, fmt.Sprintf(formatter, args...); e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			logCalled <- struct{}{}
		})
		recorder.Eventf(item.obj, item.reason, item.messageFmt, item.elements...)

		<-logCalled

		// validate event
		if item.expectUpdate {
			actualEvent := <-patchEvent
			validateEvent(actualEvent, item.expect, t)
		} else {
			actualEvent := <-createEvent
			validateEvent(actualEvent, item.expect, t)
		}

		logWatcher1.Stop()
		logWatcher2.Stop()
	}
	sinkWatcher.Stop()
}

func TestMultiSinkCache(t *testing.T) {
	testPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			SelfLink:  "/api/version/pods/foo",
			Name:      "foo",
			Namespace: "baz",
			UID:       "bar",
		},
	}
	testPod2 := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			SelfLink:  "/api/version/pods/foo",
			Name:      "foo",
			Namespace: "baz",
			UID:       "differentUid",
		},
	}
	testRef, err := api.GetPartialReference(testPod, "spec.containers[2]")
	testRef2, err := api.GetPartialReference(testPod2, "spec.containers[3]")
	if err != nil {
		t.Fatal(err)
	}
	table := []struct {
		obj          runtime.Object
		reason       string
		messageFmt   string
		elements     []interface{}
		expect       *api.Event
		expectLog    string
		expectUpdate bool
	}{
		{
			obj:        testRef,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "bar",
					APIVersion: "version",
					FieldPath:  "spec.containers[2]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[2]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: false,
		},
		{
			obj:        testPod,
			reason:     "Killed",
			messageFmt: "some other verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "bar",
					APIVersion: "version",
				},
				Reason:  "Killed",
				Message: "some other verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:""}): reason: 'Killed' some other verbose message: 1`,
			expectUpdate: false,
		},
		{
			obj:        testRef,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "bar",
					APIVersion: "version",
					FieldPath:  "spec.containers[2]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   2,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[2]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: true,
		},
		{
			obj:        testRef2,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "differentUid",
					APIVersion: "version",
					FieldPath:  "spec.containers[3]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"differentUid", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[3]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: false,
		},
		{
			obj:        testRef,
			reason:     "Started",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "bar",
					APIVersion: "version",
					FieldPath:  "spec.containers[2]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   3,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[2]"}): reason: 'Started' some verbose message: 1`,
			expectUpdate: true,
		},
		{
			obj:        testRef2,
			reason:     "Stopped",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "differentUid",
					APIVersion: "version",
					FieldPath:  "spec.containers[3]",
				},
				Reason:  "Stopped",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"differentUid", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[3]"}): reason: 'Stopped' some verbose message: 1`,
			expectUpdate: false,
		},
		{
			obj:        testRef2,
			reason:     "Stopped",
			messageFmt: "some verbose message: %v",
			elements:   []interface{}{1},
			expect: &api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "baz",
				},
				InvolvedObject: api.ObjectReference{
					Kind:       "Pod",
					Name:       "foo",
					Namespace:  "baz",
					UID:        "differentUid",
					APIVersion: "version",
					FieldPath:  "spec.containers[3]",
				},
				Reason:  "Stopped",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   2,
			},
			expectLog:    `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"differentUid", APIVersion:"version", ResourceVersion:"", FieldPath:"spec.containers[3]"}): reason: 'Stopped' some verbose message: 1`,
			expectUpdate: true,
		},
	}

	testCache := map[string]*api.Event{}
	createEvent := make(chan *api.Event)
	updateEvent := make(chan *api.Event)
	patchEvent := make(chan *api.Event)
	testEvents := testEventSink{
		OnCreate: OnCreateFactory(testCache, createEvent),
		OnUpdate: func(event *api.Event) (*api.Event, error) {
			updateEvent <- event
			return event, nil
		},
		OnPatch: OnPatchFactory(testCache, patchEvent),
	}

	testCache2 := map[string]*api.Event{}
	createEvent2 := make(chan *api.Event)
	updateEvent2 := make(chan *api.Event)
	patchEvent2 := make(chan *api.Event)
	testEvents2 := testEventSink{
		OnCreate: OnCreateFactory(testCache2, createEvent2),
		OnUpdate: func(event *api.Event) (*api.Event, error) {
			updateEvent2 <- event
			return event, nil
		},
		OnPatch: OnPatchFactory(testCache2, patchEvent2),
	}

	eventBroadcaster := NewBroadcaster()
	clock := &util.FakeClock{time.Now()}
	recorder := recorderWithFakeClock(api.EventSource{Component: "eventTest"}, eventBroadcaster, clock)

	sinkWatcher := eventBroadcaster.StartRecordingToSink(&testEvents)
	for _, item := range table {
		clock.Step(1 * time.Second)
		recorder.Eventf(item.obj, item.reason, item.messageFmt, item.elements...)

		// validate event
		if item.expectUpdate {
			actualEvent := <-patchEvent
			validateEvent(actualEvent, item.expect, t)
		} else {
			actualEvent := <-createEvent
			validateEvent(actualEvent, item.expect, t)
		}
	}

	// Another StartRecordingToSink call should start to record events with new clean cache.
	sinkWatcher2 := eventBroadcaster.StartRecordingToSink(&testEvents2)
	for _, item := range table {
		clock.Step(1 * time.Second)
		recorder.Eventf(item.obj, item.reason, item.messageFmt, item.elements...)

		// validate event
		if item.expectUpdate {
			actualEvent := <-patchEvent2
			validateEvent(actualEvent, item.expect, t)
		} else {
			actualEvent := <-createEvent2
			validateEvent(actualEvent, item.expect, t)
		}
	}

	sinkWatcher.Stop()
	sinkWatcher2.Stop()
}
