/*
Copyright 2015 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"sync"
)

type History struct {
	// The number of times the event has occured since first occurance.
	Count int

	// The time at which the event was first recorded.
	FirstTimestamp util.Time

	// The unique name of the first occurance of this event
	Name string

	// Resource version returned from previous interaction with server
	ResourceVersion string
}

type historyMap struct {
	sync.RWMutex
	table map[string]History
}

var previousEvents = historyMap{table: make(map[string]History)}

// AddOrUpdateEvent creates a new entry for the given event in the previous events hash table if the event
// doesn't already exist, otherwise it updates the existing entry.
func AddOrUpdateEvent(newEvent *api.Event) History {
	key := getEventKey(newEvent)
	previousEvents.Lock()
	defer previousEvents.Unlock()
	previousEvents.table[key] =
		History{
			Count:           newEvent.Count,
			FirstTimestamp:  newEvent.FirstTimestamp,
			Name:            newEvent.Name,
			ResourceVersion: newEvent.ResourceVersion,
		}
	return previousEvents.table[key]
}

// GetEvent returns the entry corresponding to the given event, if one exists, otherwise a History object
// with a count of 1 is returned.
func GetEvent(event *api.Event) History {
	key := getEventKey(event)
	previousEvents.RLock()
	defer previousEvents.RUnlock()
	return previousEvents.table[key]
}

func getEventKey(event *api.Event) string {
	return event.Source.Component +
		event.Source.Host +
		event.InvolvedObject.Kind +
		event.InvolvedObject.Namespace +
		event.InvolvedObject.Name +
		string(event.InvolvedObject.UID) +
		event.InvolvedObject.APIVersion +
		event.Reason +
		event.Message
}
