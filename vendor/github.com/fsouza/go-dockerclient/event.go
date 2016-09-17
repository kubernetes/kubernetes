// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httputil"
	"sync"
	"sync/atomic"
	"time"
)

// APIEvents represents events coming from the Docker API
// The fields in the Docker API changed in API version 1.22, and
// events for more than images and containers are now fired off.
// To maintain forward and backward compatibility, go-dockerclient
// replicates the event in both the new and old format as faithfully as possible.
//
// For events that only exist in 1.22 in later, `Status` is filled in as
// `"Type:Action"` instead of just `Action` to allow for older clients to
// differentiate and not break if they rely on the pre-1.22 Status types.
//
// The transformEvent method can be consulted for more information about how
// events are translated from new/old API formats
type APIEvents struct {
	// New API Fields in 1.22
	Action string   `json:"action,omitempty"`
	Type   string   `json:"type,omitempty"`
	Actor  APIActor `json:"actor,omitempty"`

	// Old API fields for < 1.22
	Status string `json:"status,omitempty"`
	ID     string `json:"id,omitempty"`
	From   string `json:"from,omitempty"`

	// Fields in both
	Time     int64 `json:"time,omitempty"`
	TimeNano int64 `json:"timeNano,omitempty"`
}

// APIActor represents an actor that accomplishes something for an event
type APIActor struct {
	ID         string            `json:"id,omitempty"`
	Attributes map[string]string `json:"attributes,omitempty"`
}

type eventMonitoringState struct {
	// `sync/atomic` expects the first word in an allocated struct to be 64-bit
	// aligned on both ARM and x86-32. See https://goo.gl/zW7dgq for more details.
	lastSeen int64
	sync.RWMutex
	sync.WaitGroup
	enabled   bool
	C         chan *APIEvents
	errC      chan error
	listeners []chan<- *APIEvents
}

const (
	maxMonitorConnRetries = 5
	retryInitialWaitTime  = 10.
)

var (
	// ErrNoListeners is the error returned when no listeners are available
	// to receive an event.
	ErrNoListeners = errors.New("no listeners present to receive event")

	// ErrListenerAlreadyExists is the error returned when the listerner already
	// exists.
	ErrListenerAlreadyExists = errors.New("listener already exists for docker events")

	// EOFEvent is sent when the event listener receives an EOF error.
	EOFEvent = &APIEvents{
		Type:   "EOF",
		Status: "EOF",
	}
)

// AddEventListener adds a new listener to container events in the Docker API.
//
// The parameter is a channel through which events will be sent.
func (c *Client) AddEventListener(listener chan<- *APIEvents) error {
	var err error
	if !c.eventMonitor.isEnabled() {
		err = c.eventMonitor.enableEventMonitoring(c)
		if err != nil {
			return err
		}
	}
	err = c.eventMonitor.addListener(listener)
	if err != nil {
		return err
	}
	return nil
}

// RemoveEventListener removes a listener from the monitor.
func (c *Client) RemoveEventListener(listener chan *APIEvents) error {
	err := c.eventMonitor.removeListener(listener)
	if err != nil {
		return err
	}
	if c.eventMonitor.listernersCount() == 0 {
		c.eventMonitor.disableEventMonitoring()
	}
	return nil
}

func (eventState *eventMonitoringState) addListener(listener chan<- *APIEvents) error {
	eventState.Lock()
	defer eventState.Unlock()
	if listenerExists(listener, &eventState.listeners) {
		return ErrListenerAlreadyExists
	}
	eventState.Add(1)
	eventState.listeners = append(eventState.listeners, listener)
	return nil
}

func (eventState *eventMonitoringState) removeListener(listener chan<- *APIEvents) error {
	eventState.Lock()
	defer eventState.Unlock()
	if listenerExists(listener, &eventState.listeners) {
		var newListeners []chan<- *APIEvents
		for _, l := range eventState.listeners {
			if l != listener {
				newListeners = append(newListeners, l)
			}
		}
		eventState.listeners = newListeners
		eventState.Add(-1)
	}
	return nil
}

func (eventState *eventMonitoringState) closeListeners() {
	for _, l := range eventState.listeners {
		close(l)
		eventState.Add(-1)
	}
	eventState.listeners = nil
}

func (eventState *eventMonitoringState) listernersCount() int {
	eventState.RLock()
	defer eventState.RUnlock()
	return len(eventState.listeners)
}

func listenerExists(a chan<- *APIEvents, list *[]chan<- *APIEvents) bool {
	for _, b := range *list {
		if b == a {
			return true
		}
	}
	return false
}

func (eventState *eventMonitoringState) enableEventMonitoring(c *Client) error {
	eventState.Lock()
	defer eventState.Unlock()
	if !eventState.enabled {
		eventState.enabled = true
		atomic.StoreInt64(&eventState.lastSeen, 0)
		eventState.C = make(chan *APIEvents, 100)
		eventState.errC = make(chan error, 1)
		go eventState.monitorEvents(c)
	}
	return nil
}

func (eventState *eventMonitoringState) disableEventMonitoring() error {
	eventState.Lock()
	defer eventState.Unlock()

	eventState.closeListeners()

	eventState.Wait()

	if eventState.enabled {
		eventState.enabled = false
		close(eventState.C)
		close(eventState.errC)
	}
	return nil
}

func (eventState *eventMonitoringState) monitorEvents(c *Client) {
	var err error
	for eventState.noListeners() {
		time.Sleep(10 * time.Millisecond)
	}
	if err = eventState.connectWithRetry(c); err != nil {
		// terminate if connect failed
		eventState.disableEventMonitoring()
		return
	}
	for eventState.isEnabled() {
		timeout := time.After(100 * time.Millisecond)
		select {
		case ev, ok := <-eventState.C:
			if !ok {
				return
			}
			if ev == EOFEvent {
				eventState.disableEventMonitoring()
				return
			}
			eventState.updateLastSeen(ev)
			go eventState.sendEvent(ev)
		case err = <-eventState.errC:
			if err == ErrNoListeners {
				eventState.disableEventMonitoring()
				return
			} else if err != nil {
				defer func() { go eventState.monitorEvents(c) }()
				return
			}
		case <-timeout:
			continue
		}
	}
}

func (eventState *eventMonitoringState) connectWithRetry(c *Client) error {
	var retries int
	eventState.RLock()
	eventChan := eventState.C
	errChan := eventState.errC
	eventState.RUnlock()
	err := c.eventHijack(atomic.LoadInt64(&eventState.lastSeen), eventChan, errChan)
	for ; err != nil && retries < maxMonitorConnRetries; retries++ {
		waitTime := int64(retryInitialWaitTime * math.Pow(2, float64(retries)))
		time.Sleep(time.Duration(waitTime) * time.Millisecond)
		eventState.RLock()
		eventChan = eventState.C
		errChan = eventState.errC
		eventState.RUnlock()
		err = c.eventHijack(atomic.LoadInt64(&eventState.lastSeen), eventChan, errChan)
	}
	return err
}

func (eventState *eventMonitoringState) noListeners() bool {
	eventState.RLock()
	defer eventState.RUnlock()
	return len(eventState.listeners) == 0
}

func (eventState *eventMonitoringState) isEnabled() bool {
	eventState.RLock()
	defer eventState.RUnlock()
	return eventState.enabled
}

func (eventState *eventMonitoringState) sendEvent(event *APIEvents) {
	eventState.RLock()
	defer eventState.RUnlock()
	eventState.Add(1)
	defer eventState.Done()
	if eventState.enabled {
		if len(eventState.listeners) == 0 {
			eventState.errC <- ErrNoListeners
			return
		}

		for _, listener := range eventState.listeners {
			listener <- event
		}
	}
}

func (eventState *eventMonitoringState) updateLastSeen(e *APIEvents) {
	eventState.Lock()
	defer eventState.Unlock()
	if atomic.LoadInt64(&eventState.lastSeen) < e.Time {
		atomic.StoreInt64(&eventState.lastSeen, e.Time)
	}
}

func (c *Client) eventHijack(startTime int64, eventChan chan *APIEvents, errChan chan error) error {
	uri := "/events"
	if startTime != 0 {
		uri += fmt.Sprintf("?since=%d", startTime)
	}
	protocol := c.endpointURL.Scheme
	address := c.endpointURL.Path
	if protocol != "unix" {
		protocol = "tcp"
		address = c.endpointURL.Host
	}
	var dial net.Conn
	var err error
	if c.TLSConfig == nil {
		dial, err = c.Dialer.Dial(protocol, address)
	} else {
		dial, err = tlsDialWithDialer(c.Dialer, protocol, address, c.TLSConfig)
	}
	if err != nil {
		return err
	}
	conn := httputil.NewClientConn(dial, nil)
	req, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		return err
	}
	res, err := conn.Do(req)
	if err != nil {
		return err
	}
	go func(res *http.Response, conn *httputil.ClientConn) {
		defer conn.Close()
		defer res.Body.Close()
		decoder := json.NewDecoder(res.Body)
		for {
			var event APIEvents
			if err = decoder.Decode(&event); err != nil {
				if err == io.EOF || err == io.ErrUnexpectedEOF {
					c.eventMonitor.RLock()
					if c.eventMonitor.enabled && c.eventMonitor.C == eventChan {
						// Signal that we're exiting.
						eventChan <- EOFEvent
					}
					c.eventMonitor.RUnlock()
					break
				}
				errChan <- err
			}
			if event.Time == 0 {
				continue
			}
			if !c.eventMonitor.isEnabled() || c.eventMonitor.C != eventChan {
				return
			}
			transformEvent(&event)
			eventChan <- &event
		}
	}(res, conn)
	return nil
}

// transformEvent takes an event and determines what version it is from
// then populates both versions of the event
func transformEvent(event *APIEvents) {
	// if event version is <= 1.21 there will be no Action and no Type
	if event.Action == "" && event.Type == "" {
		event.Action = event.Status
		event.Actor.ID = event.ID
		event.Actor.Attributes = map[string]string{}
		switch event.Status {
		case "delete", "import", "pull", "push", "tag", "untag":
			event.Type = "image"
		default:
			event.Type = "container"
			if event.From != "" {
				event.Actor.Attributes["image"] = event.From
			}
		}
	} else {
		if event.Status == "" {
			if event.Type == "image" || event.Type == "container" {
				event.Status = event.Action
			} else {
				// Because just the Status has been overloaded with different Types
				// if an event is not for an image or a container, we prepend the type
				// to avoid problems for people relying on actions being only for
				// images and containers
				event.Status = event.Type + ":" + event.Action
			}
		}
		if event.ID == "" {
			event.ID = event.Actor.ID
		}
		if event.From == "" {
			event.From = event.Actor.Attributes["image"]
		}
	}
}
