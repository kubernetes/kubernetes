/*
Copyright 2024 The Kubernetes Authors.

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

package controller

import (
	"slices"
	"time"

	"k8s.io/client-go/util/workqueue"
)

// TODO (pohly): move this to k8s.io/client-go/util/workqueue/workqueue.go
// if it turns out to be generally useful. Doc comments are already written
// as if the code was there.

// MockQueue is an implementation of [TypedRateLimitingInterface] which
// can be used to test a function which pulls work items out of a queue
// and processes them.
//
// A null instance is directly usable. The usual usage is:
//
//	var m workqueue.Mock[string]
//	m.SyncOne("some-item", func(queue workqueue.TypedRateLimitingInterface[string]) { ... } )
//	if diff := cmp.Diff(workqueue.Mock[string]{}, m); diff != "" {
//	    t.Errorf("unexpected state of mock work queue after sync (-want, +got):\n%s", diff)
//	}
//
// All slices get reset to nil when they become empty, so there are no spurious
// differences because of the nil vs. empty slice.
type Mock[T comparable] struct {
	// Ready contains the items which are ready for processing.
	Ready []T

	// InFlight contains the items which are currently being processed (= Get
	// was called, Done not yet).
	InFlight []T

	// MismatchedDone contains the items for which Done was called without
	// a matching Get.
	MismatchedDone []T

	// Later contains the items which are meant to be added to the queue after
	// a certain delay (= AddAfter was called for them).
	Later []MockDelayedItem[T]

	// Failures contains the items and their retry count which failed to be
	// processed (AddRateLimited called at least once, Forget not yet).
	// The retry count is always larger than zero.
	Failures map[T]int

	// ShutDownCalled tracks how often ShutDown got called.
	ShutDownCalled int

	// ShutDownWithDrainCalled tracks how often ShutDownWithDrain got called.
	ShutDownWithDrainCalled int
}

// MockDelayedItem is an item which was queue for later processing.
type MockDelayedItem[T comparable] struct {
	Item     T
	Duration time.Duration
}

// SyncOne adds the item to the work queue and calls sync.
// That sync function can pull one or more items from the work
// queue until the queue is empty. Then it is told that the queue
// is shutting down, which must cause it to return.
//
// The test can then retrieve the state of the queue to check the result.
func (m *Mock[T]) SyncOne(item T, sync func(workqueue.TypedRateLimitingInterface[T])) {
	m.Ready = append(m.Ready, item)
	sync(m)
}

// Add implements [TypedInterface].
func (m *Mock[T]) Add(item T) {
	m.Ready = append(m.Ready, item)
}

// Len implements [TypedInterface].
func (m *Mock[T]) Len() int {
	return len(m.Ready)
}

// Get implements [TypedInterface].
func (m *Mock[T]) Get() (item T, shutdown bool) {
	if len(m.Ready) == 0 {
		shutdown = true
		return
	}
	item = m.Ready[0]
	m.Ready = m.Ready[1:]
	if len(m.Ready) == 0 {
		m.Ready = nil
	}
	m.InFlight = append(m.InFlight, item)
	return item, false
}

// Done implements [TypedInterface].
func (m *Mock[T]) Done(item T) {
	index := slices.Index(m.InFlight, item)
	if index < 0 {
		m.MismatchedDone = append(m.MismatchedDone, item)
	}
	m.InFlight = slices.Delete(m.InFlight, index, index+1)
	if len(m.InFlight) == 0 {
		m.InFlight = nil
	}
}

// ShutDown implements [TypedInterface].
func (m *Mock[T]) ShutDown() {
	m.ShutDownCalled++
}

// ShutDownWithDrain implements [TypedInterface].
func (m *Mock[T]) ShutDownWithDrain() {
	m.ShutDownWithDrainCalled++
}

// ShuttingDown implements [TypedInterface].
func (m *Mock[T]) ShuttingDown() bool {
	return m.ShutDownCalled > 0 || m.ShutDownWithDrainCalled > 0
}

// AddAfter implements [TypedDelayingInterface.AddAfter]
func (m *Mock[T]) AddAfter(item T, duration time.Duration) {
	m.Later = append(m.Later, MockDelayedItem[T]{Item: item, Duration: duration})
}

// AddRateLimited implements [TypedRateLimitingInterface.AddRateLimited].
func (m *Mock[T]) AddRateLimited(item T) {
	if m.Failures == nil {
		m.Failures = make(map[T]int)
	}
	m.Failures[item]++
}

// Forget implements [TypedRateLimitingInterface.Forget].
func (m *Mock[T]) Forget(item T) {
	if m.Failures == nil {
		return
	}
	delete(m.Failures, item)
}

// NumRequeues implements [TypedRateLimitingInterface.NumRequeues].
func (m *Mock[T]) NumRequeues(item T) int {
	return m.Failures[item]
}
