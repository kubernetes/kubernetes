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

package queue

// FIFO implements a first-in-first-out queue with unbounded size.
// The null FIFO is a valid empty queue.
//
// Access must be protected by the caller when used concurrently by
// differerent goroutines, the queue itself implements no locking.
type FIFO[T any] struct {
	// elements contains a buffer for elements which have been
	// pushed and not popped yet. Two scenarios are possible:
	// - one chunk in the middle (start <= end)
	// - one chunk at the end, followed by one chunk at the
	//   beginning (end <= start)
	//
	// start == end can be either an empty queue or a completely
	// full one (with two chunks).
	elements []T

	// len counts the number of elements which have been pushed and
	// not popped yet.
	len int

	// start is the index of the first valid element.
	start int

	// end is the index after the last valid element.
	end int
}

func (q *FIFO[T]) Len() int {
	return q.len
}

func (q *FIFO[T]) Push(element T) {
	size := len(q.elements)
	if q.len == size {
		// Need larger buffer.
		newSize := size * 2
		if newSize == 0 {
			newSize = 4
		}
		elements := make([]T, newSize)
		if q.start == 0 {
			copy(elements, q.elements)
		} else {
			copy(elements, q.elements[q.start:])
			copy(elements[len(q.elements)-q.start:], q.elements[0:q.end])
		}
		q.start = 0
		q.end = q.len
		q.elements = elements
		size = newSize
	}
	if q.end == size {
		// Wrap around.
		q.elements[0] = element
		q.end = 1
		q.len++
		return
	}
	q.elements[q.end] = element
	q.end++
	q.len++
}

func (q *FIFO[T]) Pop() (element T, ok bool) {
	if q.len == 0 {
		return
	}
	element = q.elements[q.start]
	q.start++
	if q.start == len(q.elements) {
		// Wrap around.
		q.start = 0
	}
	q.len--
	ok = true
	return
}
