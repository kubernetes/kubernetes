/*
Copyright 2021 The Kubernetes Authors.

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

package queueset

import (
	"container/list"
)

// removeFunc removes a designated element from the list.
// The complexity of the runtime cost is O(1)
type removeFunc func()

// internal interface to abstract out the implementation details
// of the underlying list used.
type fifo interface {
	// Enqueue enqueues the specified request into the list and
	// returns a removeFunc function that can be used to remove the
	// request from the list
	Enqueue(request *request) removeFunc

	// Dequeue pulls out the oldest request from the list.
	Dequeue() (*request, bool)

	// Length returns the number of requests in the list.
	Length() int

	// ForEach iterates through the list and executes the specified
	// function for each request.
	// The order is least recent to most recent.
	ForEach(func(r *request))
}

type requestFIFO struct {
	*list.List
}

func newRequestFIFO() fifo {
	return &requestFIFO{
		list.New(),
	}
}

func (l *requestFIFO) Length() int {
	return l.Len()
}

func (l *requestFIFO) Enqueue(request *request) removeFunc {
	e := l.PushBack(request)
	return func() { l.Remove(e) }
}

func (l *requestFIFO) Dequeue() (*request, bool) {
	e := l.Front()
	if e == nil {
		return nil, false
	}

	request, ok := e.Value.(*request)
	if !ok {
		// we should never be here.
		return nil, false
	}

	return request, true
}

func (l *requestFIFO) ForEach(f func(r *request)) {
	for current := l.Front(); current != nil; current = current.Next() {
		if r, ok := current.Value.(*request); ok {
			f(r)
		}
	}
}
