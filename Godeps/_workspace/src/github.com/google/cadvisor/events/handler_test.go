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
func makeEvent(inTime time.Time, containerName string) *Event {
	return &Event{
		ContainerName: containerName,
		Timestamp:     inTime,
		EventType:     TypeOom,
	}
}

// returns EventManager and Request to use in tests
func initializeScenario(t *testing.T) (*events, *Request, *Event, *Event) {
	fakeEvent := makeEvent(createOldTime(t), "/")
	fakeEvent2 := makeEvent(time.Now(), "/")

	return NewEventManager(), NewRequest(), fakeEvent, fakeEvent2
}

func checkNumberOfEvents(t *testing.T, numEventsExpected int, numEventsReceived int) {
	if numEventsReceived != numEventsExpected {
		t.Fatalf("Expected to return %v events but received %v",
			numEventsExpected, numEventsReceived)
	}
}

func ensureProperEventReturned(t *testing.T, expectedEvent *Event, eventObjectFound *Event) {
	if eventObjectFound != expectedEvent {
		t.Errorf("Expected to find test object %v but found a different object: %v",
			expectedEvent, eventObjectFound)
	}
}

func TestCheckIfIsSubcontainer(t *testing.T) {
	myRequest := NewRequest()
	myRequest.ContainerName = "/root"

	sameContainerEvent := &Event{
		ContainerName: "/root",
	}
	subContainerEvent := &Event{
		ContainerName: "/root/subdir",
	}
	differentContainerEvent := &Event{
		ContainerName: "/root-completely-different-container",
	}

	if !checkIfIsSubcontainer(myRequest, sameContainerEvent) {
		t.Errorf("should have found %v and %v had the same container name",
			myRequest, sameContainerEvent)
	}
	if checkIfIsSubcontainer(myRequest, subContainerEvent) {
		t.Errorf("should have found %v and %v had different containers",
			myRequest, subContainerEvent)
	}

	myRequest.IncludeSubcontainers = true

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
	myRequest.EventType[TypeOom] = true
	outChannel := make(chan *Event, 10)
	myEventHolder.WatchEvents(outChannel, myRequest)

	myEventHolder.AddEvent(fakeEvent)
	myEventHolder.AddEvent(fakeEvent2)

	startTime := time.Now()
	go func() {
		time.Sleep(5 * time.Second)
		if time.Since(startTime) > (5 * time.Second) {
			t.Errorf("Took too long to receive all the events")
			close(outChannel)
		}
	}()

	eventsFound := 0
	go func() {
		for event := range outChannel {
			eventsFound += 1
			if eventsFound == 1 {
				ensureProperEventReturned(t, fakeEvent, event)
			} else if eventsFound == 2 {
				ensureProperEventReturned(t, fakeEvent2, event)
				close(outChannel)
				break
			}
		}
	}()
}

func TestAddEventAddsEventsToEventManager(t *testing.T) {
	myEventHolder, _, fakeEvent, _ := initializeScenario(t)

	myEventHolder.AddEvent(fakeEvent)

	checkNumberOfEvents(t, 1, myEventHolder.eventlist.Len())
	ensureProperEventReturned(t, fakeEvent, myEventHolder.eventlist[0])
}

func TestGetEventsForOneEvent(t *testing.T) {
	myEventHolder, myRequest, fakeEvent, fakeEvent2 := initializeScenario(t)
	myRequest.MaxEventsReturned = 1
	myRequest.EventType[TypeOom] = true

	myEventHolder.AddEvent(fakeEvent)
	myEventHolder.AddEvent(fakeEvent2)

	receivedEvents, err := myEventHolder.GetEvents(myRequest)
	if err != nil {
		t.Errorf("Failed to GetEvents: %v", err)
	}
	checkNumberOfEvents(t, 1, receivedEvents.Len())
	ensureProperEventReturned(t, fakeEvent2, receivedEvents[0])
}

func TestGetEventsForTimePeriod(t *testing.T) {
	myEventHolder, myRequest, fakeEvent, fakeEvent2 := initializeScenario(t)
	myRequest.StartTime = createOldTime(t).Add(-1 * time.Second * 10)
	myRequest.EndTime = createOldTime(t).Add(time.Second * 10)
	myRequest.EventType[TypeOom] = true

	myEventHolder.AddEvent(fakeEvent)
	myEventHolder.AddEvent(fakeEvent2)

	receivedEvents, err := myEventHolder.GetEvents(myRequest)
	if err != nil {
		t.Errorf("Failed to GetEvents: %v", err)
	}

	checkNumberOfEvents(t, 1, receivedEvents.Len())
	ensureProperEventReturned(t, fakeEvent, receivedEvents[0])
}

func TestGetEventsForNoTypeRequested(t *testing.T) {
	myEventHolder, myRequest, fakeEvent, fakeEvent2 := initializeScenario(t)

	myEventHolder.AddEvent(fakeEvent)
	myEventHolder.AddEvent(fakeEvent2)

	receivedEvents, err := myEventHolder.GetEvents(myRequest)
	if err != nil {
		t.Errorf("Failed to GetEvents: %v", err)
	}
	checkNumberOfEvents(t, 0, receivedEvents.Len())
}
