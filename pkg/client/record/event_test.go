/*
Copyright 2014 Google Inc. All rights reserved.

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
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func init() {
	// Don't bother sleeping between retries.
	sleepDuration = 0
}

type testEventRecorder struct {
	OnEvent func(e *api.Event) (*api.Event, error)
}

// CreateEvent records the event for testing.
func (t *testEventRecorder) Create(e *api.Event) (*api.Event, error) {
	if t.OnEvent != nil {
		return t.OnEvent(e)
	}
	return e, nil
}

func (t *testEventRecorder) clearOnEvent() {
	t.OnEvent = nil
}

func TestEventf(t *testing.T) {
	testPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			SelfLink:  "/api/v1beta1/pods/foo",
			Name:      "foo",
			Namespace: "baz",
			UID:       "bar",
		},
	}
	testRef, err := api.GetPartialReference(testPod, "desiredState.manifest.containers[2]")
	if err != nil {
		t.Fatal(err)
	}
	table := []struct {
		obj        runtime.Object
		reason     string
		messageFmt string
		elements   []interface{}
		expect     *api.Event
		expectLog  string
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
					APIVersion: "v1beta1",
					FieldPath:  "desiredState.manifest.containers[2]",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog: `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"v1beta1", ResourceVersion:"", FieldPath:"desiredState.manifest.containers[2]"}): reason: 'Started' some verbose message: 1`,
		},
		{
			obj:        testPod,
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
					APIVersion: "v1beta1",
				},
				Reason:  "Started",
				Message: "some verbose message: 1",
				Source:  api.EventSource{Component: "eventTest"},
				Count:   1,
			},
			expectLog: `Event(api.ObjectReference{Kind:"Pod", Namespace:"baz", Name:"foo", UID:"bar", APIVersion:"v1beta1", ResourceVersion:"", FieldPath:""}): reason: 'Started' some verbose message: 1`,
		},
	}

	for _, item := range table {
		called := make(chan struct{})
		testEvents := testEventRecorder{
			OnEvent: func(event *api.Event) (*api.Event, error) {
				a := *event
				// Just check that the timestamp was set.
				if a.FirstTimestamp.IsZero() || a.LastTimestamp.IsZero() {
					t.Errorf("timestamp wasn't set")
				}
				a.FirstTimestamp = item.expect.FirstTimestamp
				a.LastTimestamp = item.expect.LastTimestamp
				// Check that name has the right prefix.
				if n, en := a.Name, item.expect.Name; !strings.HasPrefix(n, en) {
					t.Errorf("Name '%v' does not contain prefix '%v'", n, en)
				}
				a.Name = item.expect.Name
				if e, a := item.expect, &a; !reflect.DeepEqual(e, a) {
					t.Errorf("diff: %s", util.ObjectDiff(e, a))
				}
				called <- struct{}{}
				return event, nil
			},
		}
		recorder := StartRecording(&testEvents, api.EventSource{Component: "eventTest"})
		logger := StartLogging(t.Logf) // Prove that it is useful
		logger2 := StartLogging(func(formatter string, args ...interface{}) {
			if e, a := item.expectLog, fmt.Sprintf(formatter, args...); e != a {
				t.Errorf("Expected '%v', got '%v'", e, a)
			}
			called <- struct{}{}
		})

		Eventf(item.obj, item.reason, item.messageFmt, item.elements...)

		<-called
		<-called
		recorder.Stop()
		logger.Stop()
		logger2.Stop()
	}
}

func TestWriteEventError(t *testing.T) {
	ref := &api.ObjectReference{
		Kind:       "Pod",
		Name:       "foo",
		Namespace:  "baz",
		UID:        "bar",
		APIVersion: "v1beta1",
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

	defer StartRecording(
		&testEventRecorder{
			OnEvent: func(event *api.Event) (*api.Event, error) {
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
		api.EventSource{Component: "eventTest"},
	).Stop()

	for caseName := range table {
		Event(ref, "Reason", caseName)
	}
	Event(ref, "Reason", "finished")
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
	testEvents := testEventRecorder{
		OnEvent: func(event *api.Event) (*api.Event, error) {
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
	recorder := StartRecording(&testEvents, api.EventSource{Component: "eventTest"})
	logger := StartLogging(func(formatter string, args ...interface{}) {
		loggerCalled <- struct{}{}
	})

	ref := &api.ObjectReference{
		Kind:       "Pod",
		Name:       "foo",
		Namespace:  "baz",
		UID:        "bar",
		APIVersion: "v1beta1",
	}
	for i := 0; i < maxQueuedEvents; i++ {
		go Event(ref, "Reason", strconv.Itoa(i))
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
	recorder.Stop()
	logger.Stop()
}
