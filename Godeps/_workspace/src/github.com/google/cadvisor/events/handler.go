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
	"errors"
	"sort"
	"strings"
	"sync"
	"time"
)

// EventManager is implemented by Events. It provides two ways to monitor
// events and one way to add events
type EventManager interface {
	// Watch checks if events fed to it by the caller of AddEvent satisfy the
	// request and if so sends the event back to the caller on outChannel
	WatchEvents(outChannel chan *Event, request *Request) error
	// GetEvents() returns a slice of all events detected that have passed
	// the *Request object parameters to the caller
	GetEvents(request *Request) (EventSlice, error)
	// AddEvent allows the caller to add an event to an EventManager
	// object
	AddEvent(e *Event) error
}

// Events  holds a slice of *Event objects with a potential field
// that caps the number of events held.  It is an implementation of the
// EventManager interface
type events struct {
	// eventlist holds the complete set of events found over an
	// EventManager events instantiation.
	eventlist EventSlice
	// the slice of watch pointers allows the EventManager access to channels
	// linked to different calls of WatchEvents. When new events are found that
	// satisfy the request of a given watch object in watchers, the event
	// is sent over the channel to that caller of WatchEvents
	watchers []*watch
	// lock that blocks eventlist from being accessed until a writer releases it
	eventsLock sync.RWMutex
	// lock that blocks watchers from being accessed until a writer releases it
	watcherLock sync.RWMutex
}

// initialized by a call to WatchEvents(), a watch struct will then be added
// to the events slice of *watch objects. When AddEvent() finds an event that
// satisfies the request parameter of a watch object in events.watchers,
// it will send that event out over the watch object's channel. The caller that
// called WatchEvents will receive the event over the channel provided to
// WatchEvents
type watch struct {
	// request specifies all the parameters that events sent through the
	// channel must satisfy. Specified by the creator of the watch object
	request *Request
	// a channel created by the caller through which events satisfying the
	// request are sent to the caller
	channel chan *Event
}

// typedef of a slice of Event pointers
type EventSlice []*Event

// Event contains information general to events such as the time at which they
// occurred, their specific type, and the actual event. Event types are
// differentiated by the EventType field of Event.
type Event struct {
	// the absolute container name for which the event occurred
	ContainerName string
	// the time at which the event occurred
	Timestamp time.Time
	// the type of event. EventType is an enumerated type
	EventType EventType
	// the original event object and all of its extraneous data, ex. an
	// OomInstance
	EventData EventDataInterface
}

// Request holds a set of parameters by which Event objects may be screened.
// The caller may want events that occurred within a specific timeframe
// or of a certain type, which may be specified in the *Request object
// they pass to an EventManager function
type Request struct {
	// events falling before StartTime do not satisfy the request. StartTime
	// must be left blank in calls to WatchEvents
	StartTime time.Time
	// events falling after EndTime do not satisfy the request. EndTime
	// must be left blank in calls to WatchEvents
	EndTime time.Time
	// EventType is a map that specifies the type(s) of events wanted
	EventType map[EventType]bool
	// allows the caller to put a limit on how many
	// events they receive. If there are more events than MaxEventsReturned
	// then the most chronologically recent events in the time period
	// specified are returned. Must be >= 1
	MaxEventsReturned int
	// the absolute container name for which the event occurred
	ContainerName string
	// if IncludeSubcontainers is false, only events occurring in the specific
	// container, and not the subcontainers, will be returned
	IncludeSubcontainers bool
}

// EventType is an enumerated type which lists the categories under which
// events may fall. The Event field EventType is populated by this enum.
type EventType int

const (
	TypeOom EventType = iota
	TypeContainerCreation
	TypeContainerDeletion
)

// a general interface which populates the Event field EventData. The actual
// object, such as an OomInstance, is set as an Event's EventData
type EventDataInterface interface {
}

// returns a pointer to an initialized Events object
func NewEventManager() *events {
	return &events{
		eventlist: make(EventSlice, 0),
		watchers:  []*watch{},
	}
}

// returns a pointer to an initialized Request object
func NewRequest() *Request {
	return &Request{
		EventType:            map[EventType]bool{},
		IncludeSubcontainers: false,
	}
}

// returns a pointer to an initialized watch object
func newWatch(request *Request, outChannel chan *Event) *watch {
	return &watch{
		request: request,
		channel: outChannel,
	}
}

// function necessary to implement the sort interface on the Events struct
func (e EventSlice) Len() int {
	return len(e)
}

// function necessary to implement the sort interface on the Events struct
func (e EventSlice) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}

// function necessary to implement the sort interface on the Events struct
func (e EventSlice) Less(i, j int) bool {
	return e[i].Timestamp.Before(e[j].Timestamp)
}

// sorts and returns up to the last MaxEventsReturned chronological elements
func getMaxEventsReturned(request *Request, eSlice EventSlice) EventSlice {
	sort.Sort(eSlice)
	n := request.MaxEventsReturned
	if n >= eSlice.Len() || n <= 0 {
		return eSlice
	}
	return eSlice[eSlice.Len()-n:]
}

// If the request wants all subcontainers, this returns if the request's
// container path is a prefix of the event container path.  Otherwise,
// it checks that the container paths of the event and request are
// equivalent
func checkIfIsSubcontainer(request *Request, event *Event) bool {
	if request.IncludeSubcontainers == true {
		return strings.HasPrefix(event.ContainerName+"/", request.ContainerName+"/")
	}
	return event.ContainerName == request.ContainerName
}

// determines if an event occurs within the time set in the request object and is the right type
func checkIfEventSatisfiesRequest(request *Request, event *Event) bool {
	startTime := request.StartTime
	endTime := request.EndTime
	eventTime := event.Timestamp
	if !startTime.IsZero() {
		if startTime.After(eventTime) {
			return false
		}
	}
	if !endTime.IsZero() {
		if endTime.Before(eventTime) {
			return false
		}
	}
	if request.EventType[event.EventType] != true {
		return false
	}
	if request.ContainerName != "" {
		return checkIfIsSubcontainer(request, event)
	}
	return true
}

// method of Events object that screens Event objects found in the eventlist
// attribute and if they fit the parameters passed by the Request object,
// adds it to a slice of *Event objects that is returned. If both MaxEventsReturned
// and StartTime/EndTime are specified in the request object, then only
// up to the most recent MaxEventsReturned events in that time range are returned.
func (self *events) GetEvents(request *Request) (EventSlice, error) {
	returnEventList := EventSlice{}
	self.eventsLock.RLock()
	defer self.eventsLock.RUnlock()
	for _, e := range self.eventlist {
		if checkIfEventSatisfiesRequest(request, e) {
			returnEventList = append(returnEventList, e)
		}
	}
	returnEventList = getMaxEventsReturned(request, returnEventList)
	return returnEventList, nil
}

// method of Events object that maintains an *Event channel passed by the user.
// When an event is added by AddEvents that satisfies the parameters in the passed
// Request object it is fed to the channel. The StartTime and EndTime of the watch
// request should be uninitialized because the purpose is to watch indefinitely
// for events that will happen in the future
func (self *events) WatchEvents(outChannel chan *Event, request *Request) error {
	if !request.StartTime.IsZero() || !request.EndTime.IsZero() {
		return errors.New(
			"for a call to watch, request.StartTime and request.EndTime must be uninitialized")
	}
	newWatcher := newWatch(request, outChannel)
	self.watcherLock.Lock()
	defer self.watcherLock.Unlock()
	self.watchers = append(self.watchers, newWatcher)
	return nil
}

// helper function to update the event manager's eventlist
func (self *events) updateEventList(e *Event) {
	self.eventsLock.Lock()
	defer self.eventsLock.Unlock()
	self.eventlist = append(self.eventlist, e)
}

func (self *events) findValidWatchers(e *Event) []*watch {
	watchesToSend := make([]*watch, 0)
	self.watcherLock.RLock()
	defer self.watcherLock.RUnlock()
	for _, watcher := range self.watchers {
		watchRequest := watcher.request
		if checkIfEventSatisfiesRequest(watchRequest, e) {
			watchesToSend = append(watchesToSend, watcher)
		}
	}
	return watchesToSend
}

// method of Events object that adds the argument Event object to the
// eventlist. It also feeds the event to a set of watch channels
// held by the manager if it satisfies the request keys of the channels
func (self *events) AddEvent(e *Event) error {
	self.updateEventList(e)
	watchesToSend := self.findValidWatchers(e)
	for _, watchObject := range watchesToSend {
		watchObject.channel <- e
	}
	return nil
}
