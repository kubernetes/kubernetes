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

// removeFromFIFOFunc removes a designated element from the list.
// The complexity of the runtime cost is O(1)
// It returns the request removed from the list.
type removeFromFIFOFunc func() *request

// walkFunc is called for each request in the list in the
// oldest -> newest order.
// ok: if walkFunc returns false then the iteration stops immediately.
type walkFunc func(*request) (ok bool)

// Internal interface to abstract out the implementation details
// of the underlying list used to maintain the requests.
//
// Note that the FIFO list is not safe for concurrent use by multiple
// goroutines without additional locking or coordination. It rests with
// the user to ensure that the FIFO list is used with proper locking.
type fifo interface {
	// Enqueue enqueues the specified request into the list and
	// returns a removeFromFIFOFunc function that can be used to remove the
	// request from the list
	Enqueue(*request) removeFromFIFOFunc

	// Dequeue pulls out the oldest request from the list.
	Dequeue() (*request, bool)

	// Length returns the number of requests in the list.
	Length() int

	// Walk iterates through the list in order of oldest -> newest
	// and executes the specified walkFunc for each request in that order.
	//
	// if the specified walkFunc returns false the Walk function
	// stops the walk an returns immediately.
	Walk(walkFunc)
}

// the FIFO list implementation is not safe for concurrent use by multiple
// goroutines without additional locking or coordination.
type requestFIFO struct {
	*list.List
}

func newRequestFIFO() fifo {
	return &requestFIFO{
		List: list.New(),
	}
}

func (l *requestFIFO) Length() int {
	return l.Len()
}

func (l *requestFIFO) Enqueue(req *request) removeFromFIFOFunc {
	e := l.PushBack(req)
	return func() *request {
		l.Remove(e)
		return req
	}
}

func (l *requestFIFO) Dequeue() (*request, bool) {
	e := l.Front()
	if e == nil {
		return nil, false
	}
	defer l.Remove(e)

	request, ok := e.Value.(*request)
	return request, ok
}

func (l *requestFIFO) Walk(f walkFunc) {
	for current := l.Front(); current != nil; current = current.Next() {
		if r, ok := current.Value.(*request); ok {
			if !f(r) {
				return
			}
		}
	}
}
