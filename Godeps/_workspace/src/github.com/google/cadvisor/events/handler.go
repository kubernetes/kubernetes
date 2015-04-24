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

	"github.com/golang/glog"
	info "github.com/google/cadvisor/info/v1"
)

type EventSlice []*info.Event

// functions necessary to implement the sort interface on the Events struct
func (e EventSlice) Len() int {
	return len(e)
}

func (e EventSlice) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}

func (e EventSlice) Less(i, j int) bool {
	return e[i].Timestamp.Before(e[j].Timestamp)
}

type EventChannel struct {
	// Watch ID. Can be used by the caller to request cancellation of watch events.
	watchId int
	// Channel on which the caller can receive watch events.
	channel chan *info.Event
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
	EventType map[info.EventType]bool
	// allows the caller to put a limit on how many
	// events to receive. If there are more events than MaxEventsReturned
	// then the most chronologically recent events in the time period
	// specified are returned. Must be >= 1
	MaxEventsReturned int
	// the absolute container name for which the event occurred
	ContainerName string
	// if IncludeSubcontainers is false, only events occurring in the specific
	// container, and not the subcontainers, will be returned
	IncludeSubcontainers bool
}

// EventManager is implemented by Events. It provides two ways to monitor
// events and one way to add events
type EventManager interface {
	// WatchEvents() allows a caller to register for receiving events based on the specified request.
	// On successful registration, an EventChannel object is returned.
	WatchEvents(request *Request) (*EventChannel, error)
	// GetEvents() returns all detected events based on the filters specified in request.
	GetEvents(request *Request) (EventSlice, error)
	// AddEvent allows the caller to add an event to an EventManager
	// object
	AddEvent(e *info.Event) error
	// Cancels a previously requested watch event.
	StopWatch(watch_id int)
}

// events provides an implementation for the EventManager interface.
type events struct {
	// eventList holds the complete set of events found over an
	// EventManager events instantiation.
	eventList EventSlice
	// map of registered watchers keyed by watch id.
	watchers map[int]*watch
	// lock guarding the eventList.
	eventsLock sync.RWMutex
	// lock guarding watchers.
	watcherLock sync.RWMutex
	// last allocated watch id.
	lastId int
}

// initialized by a call to WatchEvents(), a watch struct will then be added
// to the events slice of *watch objects. When AddEvent() finds an event that
// satisfies the request parameter of a watch object in events.watchers,
// it will send that event out over the watch object's channel. The caller that
// called WatchEvents will receive the event over the channel provided to
// WatchEvents
type watch struct {
	// request parameters passed in by the caller of WatchEvents()
	request *Request
	// a channel used to send event back to the caller.
	eventChannel *EventChannel
	// unique identifier of a watch that is used as a key in events' watchers
	// map
	id int
}

func NewEventChannel(watchId int) *EventChannel {
	return &EventChannel{
		watchId: watchId,
		channel: make(chan *info.Event, 10),
	}
}

// returns a pointer to an initialized Events object
func NewEventManager() *events {
	return &events{
		eventList: make(EventSlice, 0),
		watchers:  make(map[int]*watch),
	}
}

// returns a pointer to an initialized Request object
func NewRequest() *Request {
	return &Request{
		EventType:            map[info.EventType]bool{},
		IncludeSubcontainers: false,
		MaxEventsReturned:    10,
	}
}

// returns a pointer to an initialized watch object
func newWatch(request *Request, eventChannel *EventChannel) *watch {
	return &watch{
		request:      request,
		eventChannel: eventChannel,
	}
}

func (self *EventChannel) GetChannel() chan *info.Event {
	return self.channel
}

func (self *EventChannel) GetWatchId() int {
	return self.watchId
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
func checkIfIsSubcontainer(request *Request, event *info.Event) bool {
	if request.IncludeSubcontainers == true {
		return request.ContainerName == "/" || strings.HasPrefix(event.ContainerName+"/", request.ContainerName+"/")
	}
	return event.ContainerName == request.ContainerName
}

// determines if an event occurs within the time set in the request object and is the right type
func checkIfEventSatisfiesRequest(request *Request, event *info.Event) bool {
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

// method of Events object that screens Event objects found in the eventList
// attribute and if they fit the parameters passed by the Request object,
// adds it to a slice of *Event objects that is returned. If both MaxEventsReturned
// and StartTime/EndTime are specified in the request object, then only
// up to the most recent MaxEventsReturned events in that time range are returned.
func (self *events) GetEvents(request *Request) (EventSlice, error) {
	returnEventList := EventSlice{}
	self.eventsLock.RLock()
	defer self.eventsLock.RUnlock()
	for _, e := range self.eventList {
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
func (self *events) WatchEvents(request *Request) (*EventChannel, error) {
	if !request.StartTime.IsZero() || !request.EndTime.IsZero() {
		return nil, errors.New(
			"for a call to watch, request.StartTime and request.EndTime must be uninitialized")
	}
	self.watcherLock.Lock()
	defer self.watcherLock.Unlock()
	new_id := self.lastId + 1
	returnEventChannel := NewEventChannel(new_id)
	newWatcher := newWatch(request, returnEventChannel)
	self.watchers[new_id] = newWatcher
	self.lastId = new_id
	return returnEventChannel, nil
}

// helper function to update the event manager's eventList
func (self *events) updateEventList(e *info.Event) {
	self.eventsLock.Lock()
	defer self.eventsLock.Unlock()
	self.eventList = append(self.eventList, e)
}

func (self *events) findValidWatchers(e *info.Event) []*watch {
	watchesToSend := make([]*watch, 0)
	for _, watcher := range self.watchers {
		watchRequest := watcher.request
		if checkIfEventSatisfiesRequest(watchRequest, e) {
			watchesToSend = append(watchesToSend, watcher)
		}
	}
	return watchesToSend
}

// method of Events object that adds the argument Event object to the
// eventList. It also feeds the event to a set of watch channels
// held by the manager if it satisfies the request keys of the channels
func (self *events) AddEvent(e *info.Event) error {
	self.updateEventList(e)
	self.watcherLock.RLock()
	defer self.watcherLock.RUnlock()
	watchesToSend := self.findValidWatchers(e)
	for _, watchObject := range watchesToSend {
		watchObject.eventChannel.GetChannel() <- e
	}
	glog.V(4).Infof("Added event %v", e)
	return nil
}

// Removes a watch instance from the EventManager's watchers map
func (self *events) StopWatch(watchId int) {
	self.watcherLock.Lock()
	defer self.watcherLock.Unlock()
	_, ok := self.watchers[watchId]
	if !ok {
		glog.Errorf("Could not find watcher instance %v", watchId)
	}
	close(self.watchers[watchId].eventChannel.GetChannel())
	delete(self.watchers, watchId)
}
