/*
Copyright 2025 The Kubernetes Authors.

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
	"context"
	"time"
)

var _ TypedInterruptibleInterface[any] = &interruptibleType[any]{}

// TypedInterruptibleInterface is an Interface that can interruptible from Get operation.
// TypedInterruptibleInterface is an Interface that can be interrupted during Get operations.
type TypedInterruptibleInterface[T comparable] interface {
	TypedInterface[T]
	GetWithContext(ctx context.Context) (item T, shutdown bool, err error)
}

// NewTypedInterruptibleQueue constructs a new workqueue with interruptible ability.
// NewTypedInterruptibleQueue does not emit metrics. For use with a MetricsProvider, please use
// NewTypedInterruptibleQueueWithConfig instead and specify a name.
func NewTypedInterruptibleQueue[T comparable]() TypedInterruptibleInterface[T] {
	return NewTypedInterruptibleQueueWithConfig(TypedQueueConfig[T]{})
}

// NewTypedInterruptibleQueueWithConfig constructs a new workqueue with options to
// customize different properties.
func NewTypedInterruptibleQueueWithConfig[T comparable](config TypedQueueConfig[T]) TypedInterruptibleInterface[T] {
	return newInterruptibleQueue(config, defaultUnfinishedWorkUpdatePeriod)
}

func newInterruptibleQueue[T comparable](config TypedQueueConfig[T], updatePeriod time.Duration) *interruptibleType[T] {
	return &interruptibleType[T]{
		newQueueWithConfig(config, defaultUnfinishedWorkUpdatePeriod),
	}
}

type interruptibleType[T comparable] struct {
	*Typed[T]
}

// GetWithContext tries to get an item to process, but allows context cancellation to
// interrupt the blocking. It returns the item, a bool indicating if the queue is
// shutting down, and an error which is set to ctx.Err() if the context was cancelled.
func (q *interruptibleType[T]) GetWithContext(ctx context.Context) (item T, shutdown bool, err error) {
	// Fast path for cancelled context
	if ctx.Err() != nil {
		return *new(T), false, ctx.Err()
	}

	q.cond.L.Lock()

	// Fast path for non-empty queue
	if q.queue.Len() > 0 {
		item = q.queue.Pop()
		q.metrics.get(item)
		q.processing.insert(item)
		q.dirty.delete(item)
		q.cond.L.Unlock()
		return item, false, nil
	}

	if q.shuttingDown {
		q.cond.L.Unlock()
		return *new(T), true, nil
	}

	// Set up for waiting
	waitCh := make(chan struct{})
	go func() {
		q.cond.L.Lock()
		for q.queue.Len() == 0 && !q.shuttingDown {
			q.cond.Wait()
		}
		close(waitCh)
		q.cond.L.Unlock()
	}()

	q.cond.L.Unlock()

	// Wait for either context cancellation or condition signaling
	select {
	case <-ctx.Done():
		// Need to reacquire lock to ensure proper state
		q.cond.L.Lock()
		defer q.cond.L.Unlock()

		// Double check if queue got items or shutdown while we were waiting for context
		if q.queue.Len() > 0 {
			item = q.queue.Pop()
			q.metrics.get(item)
			q.processing.insert(item)
			q.dirty.delete(item)
			return item, false, nil
		}

		if q.shuttingDown {
			return *new(T), true, nil
		}

		return *new(T), false, ctx.Err()

	case <-waitCh:
		// Condition was signaled, get the item
		q.cond.L.Lock()
		defer q.cond.L.Unlock()

		if q.queue.Len() == 0 {
			// Queue is empty, must be shutting down
			return *new(T), true, nil
		}

		item = q.queue.Pop()
		q.metrics.get(item)
		q.processing.insert(item)
		q.dirty.delete(item)
		return item, false, nil
	}
}
