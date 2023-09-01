/*
 *
 * Copyright 2023 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpcsync

import (
	"context"
	"sync"
)

// Subscriber represents an entity that is subscribed to messages published on
// a PubSub. It wraps the callback to be invoked by the PubSub when a new
// message is published.
type Subscriber interface {
	// OnMessage is invoked when a new message is published. Implementations
	// must not block in this method.
	OnMessage(msg interface{})
}

// PubSub is a simple one-to-many publish-subscribe system that supports
// messages of arbitrary type. It guarantees that messages are delivered in
// the same order in which they were published.
//
// Publisher invokes the Publish() method to publish new messages, while
// subscribers interested in receiving these messages register a callback
// via the Subscribe() method.
//
// Once a PubSub is stopped, no more messages can be published, and
// it is guaranteed that no more subscriber callback will be invoked.
type PubSub struct {
	cs     *CallbackSerializer
	cancel context.CancelFunc

	// Access to the below fields are guarded by this mutex.
	mu          sync.Mutex
	msg         interface{}
	subscribers map[Subscriber]bool
	stopped     bool
}

// NewPubSub returns a new PubSub instance.
func NewPubSub() *PubSub {
	ctx, cancel := context.WithCancel(context.Background())
	return &PubSub{
		cs:          NewCallbackSerializer(ctx),
		cancel:      cancel,
		subscribers: map[Subscriber]bool{},
	}
}

// Subscribe registers the provided Subscriber to the PubSub.
//
// If the PubSub contains a previously published message, the Subscriber's
// OnMessage() callback will be invoked asynchronously with the existing
// message to begin with, and subsequently for every newly published message.
//
// The caller is responsible for invoking the returned cancel function to
// unsubscribe itself from the PubSub.
func (ps *PubSub) Subscribe(sub Subscriber) (cancel func()) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	if ps.stopped {
		return func() {}
	}

	ps.subscribers[sub] = true

	if ps.msg != nil {
		msg := ps.msg
		ps.cs.Schedule(func(context.Context) {
			ps.mu.Lock()
			defer ps.mu.Unlock()
			if !ps.subscribers[sub] {
				return
			}
			sub.OnMessage(msg)
		})
	}

	return func() {
		ps.mu.Lock()
		defer ps.mu.Unlock()
		delete(ps.subscribers, sub)
	}
}

// Publish publishes the provided message to the PubSub, and invokes
// callbacks registered by subscribers asynchronously.
func (ps *PubSub) Publish(msg interface{}) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	if ps.stopped {
		return
	}

	ps.msg = msg
	for sub := range ps.subscribers {
		s := sub
		ps.cs.Schedule(func(context.Context) {
			ps.mu.Lock()
			defer ps.mu.Unlock()
			if !ps.subscribers[s] {
				return
			}
			s.OnMessage(msg)
		})
	}
}

// Stop shuts down the PubSub and releases any resources allocated by it.
// It is guaranteed that no subscriber callbacks would be invoked once this
// method returns.
func (ps *PubSub) Stop() {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.stopped = true

	ps.cancel()
}
