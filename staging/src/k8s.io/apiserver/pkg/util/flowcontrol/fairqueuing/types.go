/*
Copyright 2016 The Kubernetes Authors.

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

package fairqueuing

import (
	"time"
)

// Request is a temporary container for "requests" with additional tracking fields
// required for the functionality FQScheduler
type Request struct {
	//TODO(aaron-prindle) seq is only used for testing, this was abstracted
	// via an interface before, keeping this for now
	QueueIdx int

	Queue           *Queue
	StartTime       time.Time
	DequeueChannel  chan bool
	RealEnqueueTime time.Time
	Enqueued        bool
}

// Queue is an array of requests with additional metadata required for
// the FQScheduler
type Queue struct {
	Requests          []*Request
	VirtualStart      float64
	RequestsExecuting int
	Index             int
}

// Enqueue enqueues a request into the queue
func (q *Queue) Enqueue(request *Request) {
	request.Enqueued = true
	q.Requests = append(q.Requests, request)
}

// Dequeue dequeues a request from the queue
func (q *Queue) Dequeue() (*Request, bool) {
	if len(q.Requests) == 0 {
		return nil, false
	}
	request := q.Requests[0]
	q.Requests = q.Requests[1:]

	request.Enqueued = false
	return request, true
}

// GetVirtualFinish returns the expected virtual finish time of the request at
// index J in the queue with estimated finish time G
func (q *Queue) GetVirtualFinish(J int, G float64) float64 {
	// The virtual finish time of request number J in the queue
	// (counting from J=1 for the head) is J * G + (virtual start time).

	// counting from J=1 for the head (eg: queue.Requests[0] -> J=1) - J+1
	jg := float64(J+1) * float64(G)
	return jg + q.VirtualStart
}
