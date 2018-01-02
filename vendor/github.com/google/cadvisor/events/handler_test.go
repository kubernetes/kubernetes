// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package events

import (
	"testing"
	"time"

	info "github.com/google/cadvisor/info/v1"

	"github.com/stretchr/testify/assert"
)

func createOldTime(t *testing.T) time.Time {
	const longForm = "Jan 2, 2006 at 3:04pm (MST)"
	linetime, err := time.Parse(longForm, "Feb 3, 2013 at 7:54pm (PST)")
	if err != nil {
		t.Fatalf("could not format time.Time object")
	} else {
		return linetime
	}
	return time.Now()
}

// used to convert an OomInstance to an Event object
func makeEvent(inTime time.Time, containerName string) *info.Event {
	return &info.Event{
		ContainerName: containerName,
		Timestamp:     inTime,
		EventType:     info.EventOom,
	}
}

// returns EventManager and Request to use in tests
func initializeScenario(t *testing.T) (*events, *Request, *info.Event, *info.Event) {
	fakeEvent := makeEvent(createOldTime(t), "/")
	fakeEvent2 := makeEvent(time.Now(), "/")

	return NewEventManager(DefaultStoragePolicy()), NewRequest(), fakeEvent, fakeEvent2
}

func checkNumberOfEvents(t *testing.T, numEventsExpected int, numEventsReceived int) {
	if numEventsReceived != numEventsExpected {
		t.Fatalf("Expected to return %v events but received %v",
			numEventsExpected, numEventsReceived)
	}
}

func ensureProperEventReturned(t *testing.T, expectedEvent *info.Event, eventObjectFound *info.Event) {
	if eventObjectFound != expectedEvent {
		t.Errorf("Expected to find test object %v but found a different object: %v",
			expectedEvent, eventObjectFound)
	}
}

func TestCheckIfIsSubcontainer(t *testing.T) {
	myRequest := NewRequest()
	myRequest.ContainerName = "/root"
	rootRequest := NewRequest()
	rootRequest.ContainerName = "/"

	sameContainerEvent := &info.Event{
		ContainerName: "/root",
	}
	subContainerEvent := &info.Event{
		ContainerName: "/root/subdir",
	}
	differentContainerEvent := &info.Event{
		ContainerName: "/root-completely-different-container",
	}

	if checkIfIsSubcontainer(rootRequest, sameContainerEvent) {
		t.Errorf("should not have found %v to be a subcontainer of %v",
			sameContainerEvent, rootRequest)
	}
	if !checkIfIsSubcontainer(myRequest, sameContainerEvent) {
		t.Errorf("should have found %v and %v had the same container name",
			myRequest, sameContainerEvent)
	}
	if checkIfIsSubcontainer(myRequest, subContainerEvent) {
		t.Errorf("should have found %v and %v had different containers",
			myRequest, subContainerEvent)
	}

	rootRequest.IncludeSubcontainers = true
	myRequest.IncludeSubcontainers = true

	if !checkIfIsSubcontainer(rootRequest, sameContainerEvent) {
		t.Errorf("should have found %v to be a subcontainer of %v",
			sameContainerEvent.ContainerName, rootRequest.ContainerName)
	}
	if !checkIfIsSubcontainer(myRequest, sameContainerEvent) {
		t.Errorf("should have found %v and %v had the same container",
			myRequest.ContainerName, sameContainerEvent.ContainerName)
	}
	if !checkIfIsSubcontainer(myRequest, subContainerEvent) {
		t.Errorf("should have found %v was a subcontainer of %v",
			subContainerEvent.ContainerName, myRequest.ContainerName)
	}
	if checkIfIsSubcontainer(myRequest, differentContainerEvent) {
		t.Errorf("should have found %v and %v had different containers",
			myRequest.ContainerName, differentContainerEvent.ContainerName)
	}
}

func TestWatchEventsDetectsNewEvents(t *testing.T) {
	myEventHolder, myRequest, fakeEvent, fakeEvent2 := initializeScenario(t)
	myRequest.EventType[info.EventOom] = true
	returnEventChannel, err := myEventHolder.WatchEvents(myRequest)
	assert.Nil(t, err)

	myEventHolder.AddEvent(fakeEvent)
	myEventHolder.AddEvent(fakeEvent2)

	startTime := time.Now()
	go func() {
		time.Sleep(5 * time.Second)
		if time.Since(startTime) > (5 * time.Second) {
			t.Errorf("Took too long to receive all the events")
		}
	}()

	eventsFound := 0
	go func() {
		for event := range returnEventChannel.GetChannel() {
			eventsFound += 1
			if eventsFound == 1 {
				ensureProperEventReturned(t, fakeEvent, event)
			} else if eventsFound == 2 {
				ensureProperEventReturned(t, fakeEvent2, event)
				break
			}
		}
	}()
}

func TestAddEventAddsEventsToEventManager(t *testing.T) {
	myEventHolder, _, fakeEvent, _ := initializeScenario(t)

	myEventHolder.AddEvent(fakeEvent)

	checkNumberOfEvents(t, 1, len(myEventHolder.eventStore))
	ensureProperEventReturned(t, fakeEvent, myEventHolder.eventStore[info.EventOom].Get(0).(*info.Event))
}

func TestGetEventsForOneEvent(t *testing.T) {
	myEventHolder, myRequest, fakeEvent, fakeEvent2 := initializeScenario(t)
	myRequest.MaxEventsReturned = 1
	myRequest.EventType[info.EventOom] = true

	myEventHolder.AddEvent(fakeEvent)
	myEventHolder.AddEvent(fakeEvent2)

	receivedEvents, err := myEventHolder.GetEvents(myRequest)
	assert.Nil(t, err)
	checkNumberOfEvents(t, 1, len(receivedEvents))
	ensureProperEventReturned(t, fakeEvent2, receivedEvents[0])
}

func TestGetEventsForTimePeriod(t *testing.T) {
	myEventHolder, myRequest, fakeEvent, fakeEvent2 := initializeScenario(t)
	myRequest.StartTime = time.Now().Add(-1 * time.Second * 10)
	myRequest.EndTime = time.Now().Add(time.Second * 10)
	myRequest.EventType[info.EventOom] = true

	myEventHolder.AddEvent(fakeEvent)
	myEventHolder.AddEvent(fakeEvent2)

	receivedEvents, err := myEventHolder.GetEvents(myRequest)
	assert.Nil(t, err)

	checkNumberOfEvents(t, 1, len(receivedEvents))
	ensureProperEventReturned(t, fakeEvent2, receivedEvents[0])
}

func TestGetEventsForNoTypeRequested(t *testing.T) {
	myEventHolder, myRequest, fakeEvent, fakeEvent2 := initializeScenario(t)

	myEventHolder.AddEvent(fakeEvent)
	myEventHolder.AddEvent(fakeEvent2)

	receivedEvents, err := myEventHolder.GetEvents(myRequest)
	assert.Nil(t, err)
	checkNumberOfEvents(t, 0, len(receivedEvents))
}
