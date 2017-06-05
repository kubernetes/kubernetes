package notifications

import (
	"container/list"
	"fmt"
	"sync"
	"time"

	"github.com/Sirupsen/logrus"
)

// NOTE(stevvooe): This file contains definitions for several utility sinks.
// Typically, the broadcaster is the only sink that should be required
// externally, but others are suitable for export if the need arises. Albeit,
// the tight integration with endpoint metrics should be removed.

// Broadcaster sends events to multiple, reliable Sinks. The goal of this
// component is to dispatch events to configured endpoints. Reliability can be
// provided by wrapping incoming sinks.
type Broadcaster struct {
	sinks  []Sink
	events chan []Event
	closed chan chan struct{}
}

// NewBroadcaster ...
// Add appends one or more sinks to the list of sinks. The broadcaster
// behavior will be affected by the properties of the sink. Generally, the
// sink should accept all messages and deal with reliability on its own. Use
// of EventQueue and RetryingSink should be used here.
func NewBroadcaster(sinks ...Sink) *Broadcaster {
	b := Broadcaster{
		sinks:  sinks,
		events: make(chan []Event),
		closed: make(chan chan struct{}),
	}

	// Start the broadcaster
	go b.run()

	return &b
}

// Write accepts a block of events to be dispatched to all sinks. This method
// will never fail and should never block (hopefully!). The caller cedes the
// slice memory to the broadcaster and should not modify it after calling
// write.
func (b *Broadcaster) Write(events ...Event) error {
	select {
	case b.events <- events:
	case <-b.closed:
		return ErrSinkClosed
	}
	return nil
}

// Close the broadcaster, ensuring that all messages are flushed to the
// underlying sink before returning.
func (b *Broadcaster) Close() error {
	logrus.Infof("broadcaster: closing")
	select {
	case <-b.closed:
		// already closed
		return fmt.Errorf("broadcaster: already closed")
	default:
		// do a little chan handoff dance to synchronize closing
		closed := make(chan struct{})
		b.closed <- closed
		close(b.closed)
		<-closed
		return nil
	}
}

// run is the main broadcast loop, started when the broadcaster is created.
// Under normal conditions, it waits for events on the event channel. After
// Close is called, this goroutine will exit.
func (b *Broadcaster) run() {
	for {
		select {
		case block := <-b.events:
			for _, sink := range b.sinks {
				if err := sink.Write(block...); err != nil {
					logrus.Errorf("broadcaster: error writing events to %v, these events will be lost: %v", sink, err)
				}
			}
		case closing := <-b.closed:

			// close all the underlying sinks
			for _, sink := range b.sinks {
				if err := sink.Close(); err != nil {
					logrus.Errorf("broadcaster: error closing sink %v: %v", sink, err)
				}
			}
			closing <- struct{}{}

			logrus.Debugf("broadcaster: closed")
			return
		}
	}
}

// eventQueue accepts all messages into a queue for asynchronous consumption
// by a sink. It is unbounded and thread safe but the sink must be reliable or
// events will be dropped.
type eventQueue struct {
	sink      Sink
	events    *list.List
	listeners []eventQueueListener
	cond      *sync.Cond
	mu        sync.Mutex
	closed    bool
}

// eventQueueListener is called when various events happen on the queue.
type eventQueueListener interface {
	ingress(events ...Event)
	egress(events ...Event)
}

// newEventQueue returns a queue to the provided sink. If the updater is non-
// nil, it will be called to update pending metrics on ingress and egress.
func newEventQueue(sink Sink, listeners ...eventQueueListener) *eventQueue {
	eq := eventQueue{
		sink:      sink,
		events:    list.New(),
		listeners: listeners,
	}

	eq.cond = sync.NewCond(&eq.mu)
	go eq.run()
	return &eq
}

// Write accepts the events into the queue, only failing if the queue has
// beend closed.
func (eq *eventQueue) Write(events ...Event) error {
	eq.mu.Lock()
	defer eq.mu.Unlock()

	if eq.closed {
		return ErrSinkClosed
	}

	for _, listener := range eq.listeners {
		listener.ingress(events...)
	}
	eq.events.PushBack(events)
	eq.cond.Signal() // signal waiters

	return nil
}

// Close shutsdown the event queue, flushing
func (eq *eventQueue) Close() error {
	eq.mu.Lock()
	defer eq.mu.Unlock()

	if eq.closed {
		return fmt.Errorf("eventqueue: already closed")
	}

	// set closed flag
	eq.closed = true
	eq.cond.Signal() // signal flushes queue
	eq.cond.Wait()   // wait for signal from last flush

	return eq.sink.Close()
}

// run is the main goroutine to flush events to the target sink.
func (eq *eventQueue) run() {
	for {
		block := eq.next()

		if block == nil {
			return // nil block means event queue is closed.
		}

		if err := eq.sink.Write(block...); err != nil {
			logrus.Warnf("eventqueue: error writing events to %v, these events will be lost: %v", eq.sink, err)
		}

		for _, listener := range eq.listeners {
			listener.egress(block...)
		}
	}
}

// next encompasses the critical section of the run loop. When the queue is
// empty, it will block on the condition. If new data arrives, it will wake
// and return a block. When closed, a nil slice will be returned.
func (eq *eventQueue) next() []Event {
	eq.mu.Lock()
	defer eq.mu.Unlock()

	for eq.events.Len() < 1 {
		if eq.closed {
			eq.cond.Broadcast()
			return nil
		}

		eq.cond.Wait()
	}

	front := eq.events.Front()
	block := front.Value.([]Event)
	eq.events.Remove(front)

	return block
}

// retryingSink retries the write until success or an ErrSinkClosed is
// returned. Underlying sink must have p > 0 of succeeding or the sink will
// block. Internally, it is a circuit breaker retries to manage reset.
// Concurrent calls to a retrying sink are serialized through the sink,
// meaning that if one is in-flight, another will not proceed.
type retryingSink struct {
	mu     sync.Mutex
	sink   Sink
	closed bool

	// circuit breaker heuristics
	failures struct {
		threshold int
		recent    int
		last      time.Time
		backoff   time.Duration // time after which we retry after failure.
	}
}

type retryingSinkListener interface {
	active(events ...Event)
	retry(events ...Event)
}

// TODO(stevvooe): We are using circuit break here, which actually doesn't
// make a whole lot of sense for this use case, since we always retry. Move
// this to use bounded exponential backoff.

// newRetryingSink returns a sink that will retry writes to a sink, backing
// off on failure. Parameters threshold and backoff adjust the behavior of the
// circuit breaker.
func newRetryingSink(sink Sink, threshold int, backoff time.Duration) *retryingSink {
	rs := &retryingSink{
		sink: sink,
	}
	rs.failures.threshold = threshold
	rs.failures.backoff = backoff

	return rs
}

// Write attempts to flush the events to the downstream sink until it succeeds
// or the sink is closed.
func (rs *retryingSink) Write(events ...Event) error {
	rs.mu.Lock()
	defer rs.mu.Unlock()

retry:

	if rs.closed {
		return ErrSinkClosed
	}

	if !rs.proceed() {
		logrus.Warnf("%v encountered too many errors, backing off", rs.sink)
		rs.wait(rs.failures.backoff)
		goto retry
	}

	if err := rs.write(events...); err != nil {
		if err == ErrSinkClosed {
			// terminal!
			return err
		}

		logrus.Errorf("retryingsink: error writing events: %v, retrying", err)
		goto retry
	}

	return nil
}

// Close closes the sink and the underlying sink.
func (rs *retryingSink) Close() error {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	if rs.closed {
		return fmt.Errorf("retryingsink: already closed")
	}

	rs.closed = true
	return rs.sink.Close()
}

// write provides a helper that dispatches failure and success properly. Used
// by write as the single-flight write call.
func (rs *retryingSink) write(events ...Event) error {
	if err := rs.sink.Write(events...); err != nil {
		rs.failure()
		return err
	}

	rs.reset()
	return nil
}

// wait backoff time against the sink, unlocking so others can proceed. Should
// only be called by methods that currently have the mutex.
func (rs *retryingSink) wait(backoff time.Duration) {
	rs.mu.Unlock()
	defer rs.mu.Lock()

	// backoff here
	time.Sleep(backoff)
}

// reset marks a successful call.
func (rs *retryingSink) reset() {
	rs.failures.recent = 0
	rs.failures.last = time.Time{}
}

// failure records a failure.
func (rs *retryingSink) failure() {
	rs.failures.recent++
	rs.failures.last = time.Now().UTC()
}

// proceed returns true if the call should proceed based on circuit breaker
// heuristics.
func (rs *retryingSink) proceed() bool {
	return rs.failures.recent < rs.failures.threshold ||
		time.Now().UTC().After(rs.failures.last.Add(rs.failures.backoff))
}
