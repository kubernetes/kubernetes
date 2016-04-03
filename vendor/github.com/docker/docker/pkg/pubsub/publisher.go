package pubsub

import (
	"sync"
	"time"
)

// NewPublisher creates a new pub/sub publisher to broadcast messages.
// The duration is used as the send timeout as to not block the publisher publishing
// messages to other clients if one client is slow or unresponsive.
// The buffer is used when creating new channels for subscribers.
func NewPublisher(publishTimeout time.Duration, buffer int) *Publisher {
	return &Publisher{
		buffer:      buffer,
		timeout:     publishTimeout,
		subscribers: make(map[subscriber]struct{}),
	}
}

type subscriber chan interface{}

// Publisher is basic pub/sub structure. Allows to send events and subscribe
// to them. Can be safely used from multiple goroutines.
type Publisher struct {
	m           sync.RWMutex
	buffer      int
	timeout     time.Duration
	subscribers map[subscriber]struct{}
}

// Len returns the number of subscribers for the publisher
func (p *Publisher) Len() int {
	p.m.RLock()
	i := len(p.subscribers)
	p.m.RUnlock()
	return i
}

// Subscribe adds a new subscriber to the publisher returning the channel.
func (p *Publisher) Subscribe() chan interface{} {
	ch := make(chan interface{}, p.buffer)
	p.m.Lock()
	p.subscribers[ch] = struct{}{}
	p.m.Unlock()
	return ch
}

// Evict removes the specified subscriber from receiving any more messages.
func (p *Publisher) Evict(sub chan interface{}) {
	p.m.Lock()
	delete(p.subscribers, sub)
	close(sub)
	p.m.Unlock()
}

// Publish sends the data in v to all subscribers currently registered with the publisher.
func (p *Publisher) Publish(v interface{}) {
	p.m.RLock()
	for sub := range p.subscribers {
		// send under a select as to not block if the receiver is unavailable
		if p.timeout > 0 {
			select {
			case sub <- v:
			case <-time.After(p.timeout):
			}
			continue
		}
		select {
		case sub <- v:
		default:
		}
	}
	p.m.RUnlock()
}

// Close closes the channels to all subscribers registered with the publisher.
func (p *Publisher) Close() {
	p.m.Lock()
	for sub := range p.subscribers {
		delete(p.subscribers, sub)
		close(sub)
	}
	p.m.Unlock()
}
