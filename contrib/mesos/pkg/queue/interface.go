/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package queue

import (
	"time"

	"k8s.io/kubernetes/pkg/client/cache"
)

type EventType int

const (
	ADD_EVENT EventType = 1 << iota
	UPDATE_EVENT
	DELETE_EVENT
	POP_EVENT
)

type Entry interface {
	Copyable
	Value() UniqueCopyable
	// types is a logically OR'd combination of EventType, e.g. ADD_EVENT|UPDATE_EVENT
	Is(types EventType) bool
}

type Copyable interface {
	// return an independent copy (deep clone) of the current object
	Copy() Copyable
}

type UniqueID interface {
	GetUID() string
}

type UniqueCopyable interface {
	Copyable
	UniqueID
}

type FIFO interface {
	cache.Store

	// Pop waits until an item is ready and returns it. If multiple items are
	// ready, they are returned in the order in which they were added/updated.
	// The item is removed from the queue (and the store) before it is returned,
	// so if you don't successfully process it, you need to add it back with Add().
	Pop(cancel <-chan struct{}) interface{}

	// Await attempts to Pop within the given interval; upon success the non-nil
	// item is returned, otherwise nil
	Await(timeout time.Duration) interface{}

	// Is there an entry for the id that matches the event mask?
	Poll(id string, types EventType) bool
}

type Delayed interface {
	// return the remaining delay; a non-positive value indicates no delay
	GetDelay() time.Duration
}

type Deadlined interface {
	// when ok, returns the time when this object should be activated/executed/evaluated
	Deadline() (deadline time.Time, ok bool)
}

// No objects are ever expected to be sent over this channel. References to BreakChan
// instances may be nil (always blocking). Signalling over this channel is performed by
// closing the channel. As such there can only ever be a single signal sent over the
// lifetime of the channel.
type BreakChan <-chan struct{}

// an optional interface to be implemented by Delayed objects; returning a nil
// channel from Breaker() results in waiting the full delay duration
type Breakout interface {
	// return a channel that signals early departure from a blocking delay
	Breaker() BreakChan
}

type UniqueDelayed interface {
	UniqueID
	Delayed
}

type UniqueDeadlined interface {
	UniqueID
	Deadlined
}

// WithoutCancel returns a chan that may never be closed and always blocks
func WithoutCancel() <-chan struct{} { return nil }
