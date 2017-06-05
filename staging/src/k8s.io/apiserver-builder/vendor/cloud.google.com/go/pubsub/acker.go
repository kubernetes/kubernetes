// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pubsub

import (
	"sync"
	"time"

	"golang.org/x/net/context"
)

// ackBuffer stores the pending ack IDs and notifies the Dirty channel when it becomes non-empty.
type ackBuffer struct {
	Dirty chan struct{}
	// Close done when ackBuffer is no longer needed.
	Done chan struct{}

	mu      sync.Mutex
	pending []string
	send    bool
}

// Add adds ackID to the buffer.
func (buf *ackBuffer) Add(ackID string) {
	buf.mu.Lock()
	defer buf.mu.Unlock()
	buf.pending = append(buf.pending, ackID)

	// If we are transitioning into a non-empty notification state.
	if buf.send && len(buf.pending) == 1 {
		buf.notify()
	}
}

// RemoveAll removes all ackIDs from the buffer and returns them.
func (buf *ackBuffer) RemoveAll() []string {
	buf.mu.Lock()
	defer buf.mu.Unlock()

	ret := buf.pending
	buf.pending = nil
	return ret
}

// SendNotifications enables sending dirty notification on empty -> non-empty transitions.
// If the buffer is already non-empty, a notification will be sent immediately.
func (buf *ackBuffer) SendNotifications() {
	buf.mu.Lock()
	defer buf.mu.Unlock()

	buf.send = true
	// If we are transitioning into a non-empty notification state.
	if len(buf.pending) > 0 {
		buf.notify()
	}
}

func (buf *ackBuffer) notify() {
	go func() {
		select {
		case buf.Dirty <- struct{}{}:
		case <-buf.Done:
		}
	}()
}

// acker acks messages in batches.
type acker struct {
	s       service
	Ctx     context.Context  // The context to use when acknowledging messages.
	Sub     string           // The full name of the subscription.
	AckTick <-chan time.Time // AckTick supplies the frequency with which to make ack requests.

	// Notify is called with an ack ID after the message with that ack ID
	// has been processed.  An ackID is considered to have been processed
	// if at least one attempt has been made to acknowledge it.
	Notify func(string)

	ackBuffer

	wg   sync.WaitGroup
	done chan struct{}
}

// Start intiates processing of ackIDs which are added via Add.
// Notify is called with each ackID once it has been processed.
func (a *acker) Start() {
	a.done = make(chan struct{})
	a.ackBuffer.Dirty = make(chan struct{})
	a.ackBuffer.Done = a.done

	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		for {
			select {
			case <-a.ackBuffer.Dirty:
				a.ack(a.ackBuffer.RemoveAll())
			case <-a.AckTick:
				a.ack(a.ackBuffer.RemoveAll())
			case <-a.done:
				return
			}
		}

	}()
}

// Ack adds an ack id to be acked in the next batch.
func (a *acker) Ack(ackID string) {
	a.ackBuffer.Add(ackID)
}

// FastMode switches acker into a mode which acks messages as they arrive, rather than waiting
// for a.AckTick.
func (a *acker) FastMode() {
	a.ackBuffer.SendNotifications()
}

// Stop drops all pending messages, and releases resources before returning.
func (a *acker) Stop() {
	close(a.done)
	a.wg.Wait()
}

const maxAckAttempts = 2

// ack acknowledges the supplied ackIDs.
// After the acknowledgement request has completed (regardless of its success
// or failure), ids will be passed to a.Notify.
func (a *acker) ack(ids []string) {
	head, tail := a.s.splitAckIDs(ids)
	for len(head) > 0 {
		for i := 0; i < maxAckAttempts; i++ {
			if a.s.acknowledge(a.Ctx, a.Sub, head) == nil {
				break
			}
		}
		// NOTE: if retry gives up and returns an error, we simply drop
		// those ack IDs. The messages will be redelivered and this is
		// a documented behaviour of the API.
		head, tail = a.s.splitAckIDs(tail)
	}
	for _, id := range ids {
		a.Notify(id)
	}
}
