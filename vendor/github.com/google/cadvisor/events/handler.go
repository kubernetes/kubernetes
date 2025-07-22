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

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils"

	"k8s.io/klog/v2"
)

type byTimestamp []*info.Event

// functions necessary to implement the sort interface on the Events struct
func (e byTimestamp) Len() int {
	return len(e)
}

func (e byTimestamp) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}

func (e byTimestamp) Less(i, j int) bool {
	return e[i].Timestamp.Before(e[j].Timestamp)
}

type EventChannel struct {
	// Watch ID. Can be used by the caller to request cancellation of watch events.
	watchID int
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
	GetEvents(request *Request) ([]*info.Event, error)
	// AddEvent allows the caller to add an event to an EventManager
	// object
	AddEvent(event *info.Event) error
	// Cancels a previously requested watch event.
	StopWatch(watchID int)
}

// events provides an implementation for the EventManager interface.
type events struct {
	// eventStore holds the events by event type.
	eventStore map[info.EventType]*utils.TimedStore
	// map of registered watchers keyed by watch id.
	watchers map[int]*watch
	// lock guarding the eventStore.
	eventsLock sync.RWMutex
	// lock guarding watchers.
	watcherLock sync.RWMutex
	// last allocated watch id.
	lastID int
	// Event storage policy.
	storagePolicy StoragePolicy
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
}

func NewEventChannel(watchID int) *EventChannel {
	return &EventChannel{
		watchID: watchID,
		channel: make(chan *info.Event, 10),
	}
}

// Policy specifying how many events to store.
// MaxAge is the max duration for which to keep events.
// MaxNumEvents is the max number of events to keep (-1 for no limit).
type StoragePolicy struct {
	// Defaults limites, used if a per-event limit is not set.
	DefaultMaxAge       time.Duration
	DefaultMaxNumEvents int

	// Per-event type limits.
	PerTypeMaxAge       map[info.EventType]time.Duration
	PerTypeMaxNumEvents map[info.EventType]int
}

func DefaultStoragePolicy() StoragePolicy {
	return StoragePolicy{
		DefaultMaxAge:       24 * time.Hour,
		DefaultMaxNumEvents: 100000,
		PerTypeMaxAge:       make(map[info.EventType]time.Duration),
		PerTypeMaxNumEvents: make(map[info.EventType]int),
	}
}

// returns a pointer to an initialized Events object.
func NewEventManager(storagePolicy StoragePolicy) EventManager {
	return &events{
		eventStore:    make(map[info.EventType]*utils.TimedStore),
		watchers:      make(map[int]*watch),
		storagePolicy: storagePolicy,
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

func (ch *EventChannel) GetChannel() chan *info.Event {
	return ch.channel
}

func (ch *EventChannel) GetWatchId() int {
	return ch.watchID
}

// sorts and returns up to the last MaxEventsReturned chronological elements
func getMaxEventsReturned(request *Request, eSlice []*info.Event) []*info.Event {
	sort.Sort(byTimestamp(eSlice))
	n := request.MaxEventsReturned
	if n >= len(eSlice) || n <= 0 {
		return eSlice
	}
	return eSlice[len(eSlice)-n:]
}

// If the request wants all subcontainers, this returns if the request's
// container path is a prefix of the event container path.  Otherwise,
// it checks that the container paths of the event and request are
// equivalent
func isSubcontainer(request *Request, event *info.Event) bool {
	if request.IncludeSubcontainers {
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
	if !request.EventType[event.EventType] {
		return false
	}
	if request.ContainerName != "" {
		return isSubcontainer(request, event)
	}
	return true
}

// method of Events object that screens Event objects found in the eventStore
// attribute and if they fit the parameters passed by the Request object,
// adds it to a slice of *Event objects that is returned. If both MaxEventsReturned
// and StartTime/EndTime are specified in the request object, then only
// up to the most recent MaxEventsReturned events in that time range are returned.
func (e *events) GetEvents(request *Request) ([]*info.Event, error) {
	returnEventList := []*info.Event{}
	e.eventsLock.RLock()
	defer e.eventsLock.RUnlock()
	for eventType, fetch := range request.EventType {
		if !fetch {
			continue
		}
		evs, ok := e.eventStore[eventType]
		if !ok {
			continue
		}

		res := evs.InTimeRange(request.StartTime, request.EndTime, request.MaxEventsReturned)
		for _, in := range res {
			e := in.(*info.Event)
			if checkIfEventSatisfiesRequest(request, e) {
				returnEventList = append(returnEventList, e)
			}
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
func (e *events) WatchEvents(request *Request) (*EventChannel, error) {
	if !request.StartTime.IsZero() || !request.EndTime.IsZero() {
		return nil, errors.New(
			"for a call to watch, request.StartTime and request.EndTime must be uninitialized")
	}
	e.watcherLock.Lock()
	defer e.watcherLock.Unlock()
	newID := e.lastID + 1
	returnEventChannel := NewEventChannel(newID)
	newWatcher := newWatch(request, returnEventChannel)
	e.watchers[newID] = newWatcher
	e.lastID = newID
	return returnEventChannel, nil
}

// helper function to update the event manager's eventStore
func (e *events) updateEventStore(event *info.Event) {
	e.eventsLock.Lock()
	defer e.eventsLock.Unlock()
	if _, ok := e.eventStore[event.EventType]; !ok {
		maxNumEvents := e.storagePolicy.DefaultMaxNumEvents
		if numEvents, ok := e.storagePolicy.PerTypeMaxNumEvents[event.EventType]; ok {
			maxNumEvents = numEvents
		}
		if maxNumEvents == 0 {
			// Event storage is disabled for event.EventType
			return
		}

		maxAge := e.storagePolicy.DefaultMaxAge
		if age, ok := e.storagePolicy.PerTypeMaxAge[event.EventType]; ok {
			maxAge = age
		}

		e.eventStore[event.EventType] = utils.NewTimedStore(maxAge, maxNumEvents)
	}
	e.eventStore[event.EventType].Add(event.Timestamp, event)
}

func (e *events) findValidWatchers(event *info.Event) []*watch {
	watchesToSend := make([]*watch, 0)
	for _, watcher := range e.watchers {
		watchRequest := watcher.request
		if checkIfEventSatisfiesRequest(watchRequest, event) {
			watchesToSend = append(watchesToSend, watcher)
		}
	}
	return watchesToSend
}

// method of Events object that adds the argument Event object to the
// eventStore. It also feeds the event to a set of watch channels
// held by the manager if it satisfies the request keys of the channels
func (e *events) AddEvent(event *info.Event) error {
	e.updateEventStore(event)
	e.watcherLock.RLock()
	defer e.watcherLock.RUnlock()
	watchesToSend := e.findValidWatchers(event)
	for _, watchObject := range watchesToSend {
		watchObject.eventChannel.GetChannel() <- event
	}
	klog.V(4).Infof("Added event %v", event)
	return nil
}

// Removes a watch instance from the EventManager's watchers map
func (e *events) StopWatch(watchID int) {
	e.watcherLock.Lock()
	defer e.watcherLock.Unlock()
	_, ok := e.watchers[watchID]
	if !ok {
		klog.Errorf("Could not find watcher instance %v", watchID)
	}
	close(e.watchers[watchID].eventChannel.GetChannel())
	delete(e.watchers, watchID)
}
