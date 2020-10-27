/*
Copyright 2019 The Kubernetes Authors.

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
	"context"
	"time"

	genericrequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/util/flowcontrol/debug"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise"
)

// request is a temporary container for "requests" with additional
// tracking fields required for the functionality FQScheduler
type request struct {
	ctx context.Context

	qs *queueSet

	flowDistinguisher string
	fsName            string

	// The relevant queue.  Is nil if this request did not go through
	// a queue.
	queue *queue

	// startTime is the real time when the request began executing
	startTime time.Time

	// decision gets set to a `requestDecision` indicating what to do
	// with this request.  It gets set exactly once, when the request
	// is removed from its queue.  The value will be decisionReject,
	// decisionCancel, or decisionExecute; decisionTryAnother never
	// appears here.
	decision promise.LockingWriteOnce

	// arrivalTime is the real time when the request entered this system
	arrivalTime time.Time

	// descr1 and descr2 are not used in any logic but they appear in
	// log messages
	descr1, descr2 interface{}

	// Indicates whether client has called Request::Wait()
	waitStarted bool

	queueNoteFn fq.QueueNoteFn
}

// queue is an array of requests with additional metadata required for
// the FQScheduler
type queue struct {
	requests []*request

	// virtualStart is the virtual time (virtual seconds since process
	// startup) when the oldest request in the queue (if there is any)
	// started virtually executing
	virtualStart float64

	requestsExecuting int
	index             int
}

// Enqueue enqueues a request into the queue
func (q *queue) Enqueue(request *request) {
	q.requests = append(q.requests, request)
}

// Dequeue dequeues a request from the queue
func (q *queue) Dequeue() (*request, bool) {
	if len(q.requests) == 0 {
		return nil, false
	}
	request := q.requests[0]
	q.requests = q.requests[1:]
	return request, true
}

// GetVirtualFinish returns the expected virtual finish time of the request at
// index J in the queue with estimated finish time G
func (q *queue) GetVirtualFinish(J int, G float64) float64 {
	// The virtual finish time of request number J in the queue
	// (counting from J=1 for the head) is J * G + (virtual start time).

	// counting from J=1 for the head (eg: queue.requests[0] -> J=1) - J+1
	jg := float64(J+1) * float64(G)
	return jg + q.virtualStart
}

func (q *queue) dump(includeDetails bool) debug.QueueDump {
	digest := make([]debug.RequestDump, len(q.requests))
	for i, r := range q.requests {
		// dump requests.
		digest[i].MatchedFlowSchema = r.fsName
		digest[i].FlowDistinguisher = r.flowDistinguisher
		digest[i].ArriveTime = r.arrivalTime
		digest[i].StartTime = r.startTime
		if includeDetails {
			userInfo, _ := genericrequest.UserFrom(r.ctx)
			digest[i].UserName = userInfo.GetName()
			requestInfo, ok := genericrequest.RequestInfoFrom(r.ctx)
			if ok {
				digest[i].RequestInfo = *requestInfo
			}
		}
	}
	return debug.QueueDump{
		VirtualStart:      q.virtualStart,
		Requests:          digest,
		ExecutingRequests: q.requestsExecuting,
	}
}
