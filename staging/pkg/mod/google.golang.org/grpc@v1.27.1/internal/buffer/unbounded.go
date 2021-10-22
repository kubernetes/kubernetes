/*
 * Copyright 2019 gRPC authors.
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

// Package buffer provides an implementation of an unbounded buffer.
package buffer

import "sync"

// Unbounded is an implementation of an unbounded buffer which does not use
// extra goroutines. This is typically used for passing updates from one entity
// to another within gRPC.
//
// All methods on this type are thread-safe and don't block on anything except
// the underlying mutex used for synchronization.
//
// Unbounded supports values of any type to be stored in it by using a channel
// of `interface{}`. This means that a call to Put() incurs an extra memory
// allocation, and also that users need a type assertion while reading. For
// performance critical code paths, using Unbounded is strongly discouraged and
// defining a new type specific implementation of this buffer is preferred. See
// internal/transport/transport.go for an example of this.
type Unbounded struct {
	c       chan interface{}
	mu      sync.Mutex
	backlog []interface{}
}

// NewUnbounded returns a new instance of Unbounded.
func NewUnbounded() *Unbounded {
	return &Unbounded{c: make(chan interface{}, 1)}
}

// Put adds t to the unbounded buffer.
func (b *Unbounded) Put(t interface{}) {
	b.mu.Lock()
	if len(b.backlog) == 0 {
		select {
		case b.c <- t:
			b.mu.Unlock()
			return
		default:
		}
	}
	b.backlog = append(b.backlog, t)
	b.mu.Unlock()
}

// Load sends the earliest buffered data, if any, onto the read channel
// returned by Get(). Users are expected to call this every time they read a
// value from the read channel.
func (b *Unbounded) Load() {
	b.mu.Lock()
	if len(b.backlog) > 0 {
		select {
		case b.c <- b.backlog[0]:
			b.backlog[0] = nil
			b.backlog = b.backlog[1:]
		default:
		}
	}
	b.mu.Unlock()
}

// Get returns a read channel on which values added to the buffer, via Put(),
// are sent on.
//
// Upon reading a value from this channel, users are expected to call Load() to
// send the next buffered value onto the channel if there is any.
func (b *Unbounded) Get() <-chan interface{} {
	return b.c
}
