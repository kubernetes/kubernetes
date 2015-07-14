/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestAddOrUpdateEventNoExisting(t *testing.T) {
	// Arrange
	eventTime := util.Now()
	event := api.Event{
		Reason:  "my reasons are many",
		Message: "my message is love",
		InvolvedObject: api.ObjectReference{
			Kind:       "Pod",
			Name:       "awesome.name",
			Namespace:  "betterNamespace",
			UID:        "C934D34AFB20242",
			APIVersion: "version",
		},
		Source: api.EventSource{
			Component: "kubelet",
			Host:      "kublet.node1",
		},
		Count:          1,
		FirstTimestamp: eventTime,
		LastTimestamp:  eventTime,
	}

	// Act
	result := addOrUpdateEvent(&event)

	// Assert
	compareEventWithHistoryEntry(&event, &result, t)
}

func TestAddOrUpdateEventExisting(t *testing.T) {
	// Arrange
	event1Time := util.Unix(2324, 2342)
	event2Time := util.Now()
	event1 := api.Event{
		Reason:  "something happened",
		Message: "can you believe it?",
		ObjectMeta: api.ObjectMeta{
			ResourceVersion: "rs1",
		},
		InvolvedObject: api.ObjectReference{
			Kind:       "Scheduler",
			Name:       "anOkName",
			Namespace:  "someNamespace",
			UID:        "C934D3234CD0242",
			APIVersion: "version",
		},
		Source: api.EventSource{
			Component: "kubelet",
			Host:      "kublet.node2",
		},
		Count:          1,
		FirstTimestamp: event1Time,
		LastTimestamp:  event1Time,
	}
	event2 := api.Event{
		Reason:  "something happened",
		Message: "can you believe it?",
		ObjectMeta: api.ObjectMeta{
			ResourceVersion: "rs2",
		},
		InvolvedObject: api.ObjectReference{
			Kind:       "Scheduler",
			Name:       "anOkName",
			Namespace:  "someNamespace",
			UID:        "C934D3234CD0242",
			APIVersion: "version",
		},
		Source: api.EventSource{
			Component: "kubelet",
			Host:      "kublet.node2",
		},
		Count:          3,
		FirstTimestamp: event1Time,
		LastTimestamp:  event2Time,
	}

	// Act
	addOrUpdateEvent(&event1)
	result1 := addOrUpdateEvent(&event2)
	result2 := getEvent(&event1)

	// Assert
	compareEventWithHistoryEntry(&event2, &result1, t)
	compareEventWithHistoryEntry(&event2, &result2, t)
}

func TestGetEventNoExisting(t *testing.T) {
	// Arrange
	event := api.Event{
		Reason:  "to be or not to be",
		Message: "do I exist",
		InvolvedObject: api.ObjectReference{
			Kind:       "Controller",
			Name:       "iAmAController",
			Namespace:  "IHaveANamespace",
			UID:        "9039D34AFBCDA42",
			APIVersion: "version",
		},
		Source: api.EventSource{
			Component: "kubelet",
			Host:      "kublet.node3",
		},
		Count: 1,
	}

	// Act
	existingEvent := getEvent(&event)

	// Assert
	if existingEvent.Count != 0 {
		t.Fatalf("There should be no existing instance of this event in the hash table.")
	}
}

func TestGetEventExisting(t *testing.T) {
	// Arrange
	eventTime := util.Now()
	event := api.Event{
		Reason:  "do I exist",
		Message: "I do, oh my",
		InvolvedObject: api.ObjectReference{
			Kind:       "Pod",
			Name:       "clever.name.here",
			Namespace:  "spaceOfName",
			UID:        "D933D32AFB2A238",
			APIVersion: "version",
		},
		Source: api.EventSource{
			Component: "kubelet",
			Host:      "kublet.node4",
		},
		Count:          1,
		FirstTimestamp: eventTime,
		LastTimestamp:  eventTime,
	}
	addOrUpdateEvent(&event)

	// Act
	existingEvent := getEvent(&event)

	// Assert
	compareEventWithHistoryEntry(&event, &existingEvent, t)
}

func compareEventWithHistoryEntry(expected *api.Event, actual *history, t *testing.T) {

	if actual.Count != expected.Count {
		t.Fatalf("There should be one existing instance of this event in the hash table.")
	}

	if !actual.FirstTimestamp.Equal(expected.FirstTimestamp) {
		t.Fatalf("Unexpected FirstTimestamp. Expected: <%v> Actual: <%v>", expected.FirstTimestamp, actual.FirstTimestamp)
	}

	if actual.Name != expected.Name {
		t.Fatalf("Unexpected Name. Expected: <%v> Actual: <%v>", expected.Name, actual.Name)
	}

	if actual.ResourceVersion != expected.ResourceVersion {
		t.Fatalf("Unexpected ResourceVersion. Expected: <%v> Actual: <%v>", expected.ResourceVersion, actual.ResourceVersion)
	}

}
