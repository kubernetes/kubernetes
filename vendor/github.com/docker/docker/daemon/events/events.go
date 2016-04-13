package events

import (
	"sync"
	"time"

	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/docker/docker/pkg/pubsub"
)

const eventsLimit = 64

// Events is pubsub channel for *jsonmessage.JSONMessage
type Events struct {
	mu     sync.Mutex
	events []*jsonmessage.JSONMessage
	pub    *pubsub.Publisher
}

// New returns new *Events instance
func New() *Events {
	return &Events{
		events: make([]*jsonmessage.JSONMessage, 0, eventsLimit),
		pub:    pubsub.NewPublisher(100*time.Millisecond, 1024),
	}
}

// Subscribe adds new listener to events, returns slice of 64 stored last events
// channel in which you can expect new events in form of interface{}, so you
// need type assertion.
func (e *Events) Subscribe() ([]*jsonmessage.JSONMessage, chan interface{}) {
	e.mu.Lock()
	current := make([]*jsonmessage.JSONMessage, len(e.events))
	copy(current, e.events)
	l := e.pub.Subscribe()
	e.mu.Unlock()
	return current, l
}

// Evict evicts listener from pubsub
func (e *Events) Evict(l chan interface{}) {
	e.pub.Evict(l)
}

// Log broadcasts event to listeners. Each listener has 100 millisecond for
// receiving event or it will be skipped.
func (e *Events) Log(action, id, from string) {
	go func() {
		e.mu.Lock()
		jm := &jsonmessage.JSONMessage{Status: action, ID: id, From: from, Time: time.Now().UTC().Unix()}
		if len(e.events) == cap(e.events) {
			// discard oldest event
			copy(e.events, e.events[1:])
			e.events[len(e.events)-1] = jm
		} else {
			e.events = append(e.events, jm)
		}
		e.mu.Unlock()
		e.pub.Publish(jm)
	}()
}

// SubscribersCount returns number of event listeners
func (e *Events) SubscribersCount() int {
	return e.pub.Len()
}
