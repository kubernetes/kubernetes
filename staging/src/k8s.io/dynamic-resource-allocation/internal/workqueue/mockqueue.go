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

package workqueue

import (
	"maps"
	"slices"
	"sync"
	"time"

	"k8s.io/client-go/util/workqueue"
)

// TODO (pohly): move this to k8s.io/client-go/util/workqueue/mockqueue.go
// if it turns out to be generally useful. Doc comments are already written
// as if the code was there.

// MockQueue is an implementation of [TypedRateLimitingInterface] which
// can be used to test a function which pulls work items out of a queue
// and processes them. It is thread-safe.
//
// A null instance is directly usable. The usual usage is:
//
//	var m workqueue.Mock[string]
//	m.SyncOne("some-item", func(queue workqueue.TypedRateLimitingInterface[string]) { ... } )
//	if diff := cmp.Diff(workqueue.MockState[string]{}, m.State()); diff != "" {
//	    t.Errorf("unexpected state of mock work queue after sync (-want, +got):\n%s", diff)
//	}
//
// All slices get reset to nil when they become empty, so there are no spurious
// differences because of nil vs. empty slice.
type Mock[T comparable] struct {
	mutex sync.Mutex
	state MockState[T]
}

type MockState[T comparable] struct {
	// Ready contains the items which are ready for processing.
	Ready []T

	// InFlight contains the items which are currently being processed (= Get
	// was called, Done not yet).
	InFlight []T

	// MismatchedDone contains the items for which Done was called without
	// a matching Get.
	MismatchedDone []T

	// Later contains the items which are meant to be added to the queue after
	// a certain delay (= AddAfter was called for them). They appear in the
	// order in which AddAfter got called.
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

// DeepCopy takes a snapshot of all slices. It cannot do a deep copy of the items in those slices,
// but typically those keys are immutable.
func (m MockState[T]) DeepCopy() *MockState[T] {
	m.Ready = slices.Clone(m.Ready)
	m.InFlight = slices.Clone(m.InFlight)
	m.MismatchedDone = slices.Clone(m.MismatchedDone)
	m.Later = slices.Clone(m.Later)
	m.Failures = maps.Clone(m.Failures)
	return &m
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
	// sync must run with the mutex not locked.
	defer sync(m)
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.state.Ready = append(m.state.Ready, item)
}

// State returns the current state of the queue.
func (m *Mock[T]) State() MockState[T] {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	return *m.state.DeepCopy()
}

// Add implements [TypedInterface].
func (m *Mock[T]) Add(item T) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !slices.Contains(m.state.Ready, item) {
		m.state.Ready = append(m.state.Ready, item)
	}
}

// Len implements [TypedInterface].
func (m *Mock[T]) Len() int {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	return len(m.state.Ready)
}

// Get implements [TypedInterface].
func (m *Mock[T]) Get() (item T, shutdown bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if len(m.state.Ready) == 0 {
		shutdown = true
		return
	}
	item = m.state.Ready[0]
	m.state.Ready = m.state.Ready[1:]
	if len(m.state.Ready) == 0 {
		m.state.Ready = nil
	}
	m.state.InFlight = append(m.state.InFlight, item)
	return item, false
}

// Done implements [TypedInterface].
func (m *Mock[T]) Done(item T) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	index := slices.Index(m.state.InFlight, item)
	if index < 0 {
		m.state.MismatchedDone = append(m.state.MismatchedDone, item)
	}
	m.state.InFlight = slices.Delete(m.state.InFlight, index, index+1)
	if len(m.state.InFlight) == 0 {
		m.state.InFlight = nil
	}
}

// ShutDown implements [TypedInterface].
func (m *Mock[T]) ShutDown() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.state.ShutDownCalled++
}

// ShutDownWithDrain implements [TypedInterface].
func (m *Mock[T]) ShutDownWithDrain() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.state.ShutDownWithDrainCalled++
}

// ShuttingDown implements [TypedInterface].
func (m *Mock[T]) ShuttingDown() bool {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	return m.state.ShutDownCalled > 0 || m.state.ShutDownWithDrainCalled > 0
}

// AddAfter implements [TypedDelayingInterface.AddAfter]
func (m *Mock[T]) AddAfter(item T, duration time.Duration) {
	if duration == 0 {
		m.Add(item)
		return
	}

	m.mutex.Lock()
	defer m.mutex.Unlock()

	for i := range m.state.Later {
		if m.state.Later[i].Item == item {
			// https://github.com/kubernetes/client-go/blob/270e5ab1714527c455865953da8ceba2810dbb50/util/workqueue/delaying_queue.go#L340-L349
			// only shortens the delay for an existing item. It does not make it longer.
			if m.state.Later[i].Duration > duration {
				m.state.Later[i].Duration = duration
			}
			return
		}
	}

	m.state.Later = append(m.state.Later, MockDelayedItem[T]{Item: item, Duration: duration})
}

// AddRateLimited implements [TypedRateLimitingInterface.AddRateLimited].
func (m *Mock[T]) AddRateLimited(item T) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.state.Failures == nil {
		m.state.Failures = make(map[T]int)
	}
	m.state.Failures[item]++
}

// Forget implements [TypedRateLimitingInterface.Forget].
func (m *Mock[T]) Forget(item T) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.state.Failures == nil {
		return
	}
	delete(m.state.Failures, item)
}

// NumRequeues implements [TypedRateLimitingInterface.NumRequeues].
func (m *Mock[T]) NumRequeues(item T) int {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	return m.state.Failures[item]
}
