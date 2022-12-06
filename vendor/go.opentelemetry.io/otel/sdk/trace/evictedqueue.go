// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
