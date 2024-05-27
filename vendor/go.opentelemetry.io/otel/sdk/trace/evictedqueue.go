// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package trace // import "go.opentelemetry.io/otel/sdk/trace"

// evictedQueue is a FIFO queue with a configurable capacity.
type evictedQueue struct {
	queue        []interface{}
	capacity     int
	droppedCount int
}

func newEvictedQueue(capacity int) evictedQueue {
	// Do not pre-allocate queue, do this lazily.
	return evictedQueue{capacity: capacity}
}

// add adds value to the evictedQueue eq. If eq is at capacity, the oldest
// queued value will be discarded and the drop count incremented.
func (eq *evictedQueue) add(value interface{}) {
	if eq.capacity == 0 {
		eq.droppedCount++
		return
	}

	if eq.capacity > 0 && len(eq.queue) == eq.capacity {
		// Drop first-in while avoiding allocating more capacity to eq.queue.
		copy(eq.queue[:eq.capacity-1], eq.queue[1:])
		eq.queue = eq.queue[:eq.capacity-1]
		eq.droppedCount++
	}
	eq.queue = append(eq.queue, value)
}
