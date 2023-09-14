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

	"k8s.io/apimachinery/pkg/util/sets"
	genericrequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/util/flowcontrol/debug"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise"
	fcrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
)

// request is a temporary container for "requests" with additional
// tracking fields required for QueueSet functionality.
type request struct {
	ctx context.Context

	qs *queueSet

	flowDistinguisher string
	fsName            string

	// The relevant queue.  Is nil if this request did not go through
	// a queue.
	queue *queue

	// estimated amount of work of the request
	workEstimate completedWorkEstimate

	// decision gets set to a `requestDecision` indicating what to do
	// with this request.  It gets set exactly once, when the request
	// is removed from its queue.  The value will be decisionReject,
	// decisionCancel, or decisionExecute.
	//
	// decision.Set is called with the queueSet locked.
	// decision.Get is called without the queueSet locked.
	decision promise.WriteOnce

	// arrivalTime is the real time when the request entered this system
	arrivalTime time.Time

	// descr1 and descr2 are not used in any logic but they appear in
	// log messages
	descr1, descr2 interface{}

	queueNoteFn fq.QueueNoteFn

	// The preceding fields are filled in at creation and not modified since;
	// the following fields may be modified later and must only be accessed while
	// holding the queueSet's lock.

	// Removes this request from its queue. If the request is not put into a
	// a queue it will be nil.
	removeFromQueueLocked removeFromFIFOFunc

	// arrivalR is R(arrivalTime).  R is, confusingly, also called "virtual time".
	// This field is meaningful only while the request is waiting in the virtual world.
	arrivalR fcrequest.SeatSeconds

	// startTime is the real time when the request began executing
	startTime time.Time

	// Indicates whether client has called Request::Wait()
	waitStarted bool
}

type completedWorkEstimate struct {
	fcrequest.WorkEstimate
	totalWork fcrequest.SeatSeconds // initial plus final work
	finalWork fcrequest.SeatSeconds // only final work
}

// queue is a sequence of requests that have arrived but not yet finished
// execution in both the real and virtual worlds.
type queue struct {
	// The requestsWaiting not yet executing in the real world are stored in a FIFO list.
	requestsWaiting fifo

	// nextDispatchR is the R progress meter reading at
	// which the next request will be dispatched in the virtual world.
	nextDispatchR fcrequest.SeatSeconds

	// requestsExecuting is the set of requests executing in the real world.
	requestsExecuting sets.Set[*request]

	// index is the position of this queue among those in its queueSet.
	index int

	// seatsInUse is the total number of "seats" currently occupied
	// by all the requests that are currently executing in this queue.
	seatsInUse int
}

// queueSum tracks the sum of initial seats, max seats, and
// totalWork from all requests in a given queue
type queueSum struct {
	// InitialSeatsSum is the sum of InitialSeats
	// associated with all requests in a given queue.
	InitialSeatsSum int

	// MaxSeatsSum is the sum of MaxSeats
	// associated with all requests in a given queue.
	MaxSeatsSum int

	// TotalWorkSum is the sum of totalWork of the waiting requests
	TotalWorkSum fcrequest.SeatSeconds
}

func (req *request) totalWork() fcrequest.SeatSeconds {
	return req.workEstimate.totalWork
}

func (qs *queueSet) completeWorkEstimate(we *fcrequest.WorkEstimate) completedWorkEstimate {
	finalWork := qs.computeFinalWork(we)
	return completedWorkEstimate{
		WorkEstimate: *we,
		totalWork:    qs.computeInitialWork(we) + finalWork,
		finalWork:    finalWork,
	}
}

func (qs *queueSet) computeInitialWork(we *fcrequest.WorkEstimate) fcrequest.SeatSeconds {
	return fcrequest.SeatsTimesDuration(float64(we.InitialSeats), qs.estimatedServiceDuration)
}

func (qs *queueSet) computeFinalWork(we *fcrequest.WorkEstimate) fcrequest.SeatSeconds {
	return fcrequest.SeatsTimesDuration(float64(we.FinalSeats), we.AdditionalLatency)
}

func (q *queue) dumpLocked(includeDetails bool) debug.QueueDump {
	waitingDigest := make([]debug.RequestDump, 0, q.requestsWaiting.Length())
	q.requestsWaiting.Walk(func(r *request) bool {
		waitingDigest = append(waitingDigest, dumpRequest(includeDetails)(r))
		return true
	})
	executingDigest := SetMapReduce(dumpRequest(includeDetails), append1[debug.RequestDump])(q.requestsExecuting)

	sum := q.requestsWaiting.QueueSum()
	queueSum := debug.QueueSum{
		InitialSeatsSum: sum.InitialSeatsSum,
		MaxSeatsSum:     sum.MaxSeatsSum,
		TotalWorkSum:    sum.TotalWorkSum.String(),
	}

	return debug.QueueDump{
		NextDispatchR:     q.nextDispatchR.String(),
		Requests:          waitingDigest,
		RequestsExecuting: executingDigest,
		ExecutingRequests: q.requestsExecuting.Len(),
		SeatsInUse:        q.seatsInUse,
		QueueSum:          queueSum,
	}
}

func dumpRequest(includeDetails bool) func(*request) debug.RequestDump {
	return func(r *request) debug.RequestDump {
		ans := debug.RequestDump{
			MatchedFlowSchema: r.fsName,
			FlowDistinguisher: r.flowDistinguisher,
			ArriveTime:        r.arrivalTime,
			StartTime:         r.startTime,
			WorkEstimate:      r.workEstimate.WorkEstimate,
		}
		if includeDetails {
			userInfo, _ := genericrequest.UserFrom(r.ctx)
			ans.UserName = userInfo.GetName()
			requestInfo, ok := genericrequest.RequestInfoFrom(r.ctx)
			if ok {
				ans.RequestInfo = *requestInfo
			}
		}
		return ans
	}
}

// SetMapReduce is map-reduce starting from a set type in the sets package.
func SetMapReduce[Elt comparable, Result, Accumulator any](mapFn func(Elt) Result, reduceFn func(Accumulator, Result) Accumulator) func(map[Elt]sets.Empty) Accumulator {
	return func(set map[Elt]sets.Empty) Accumulator {
		var ans Accumulator
		for elt := range set {
			ans = reduceFn(ans, mapFn(elt))
		}
		return ans
	}
}

// SliceMapReduce is map-reduce starting from a slice.
func SliceMapReduce[Elt, Result, Accumulator any](mapFn func(Elt) Result, reduceFn func(Accumulator, Result) Accumulator) func([]Elt) Accumulator {
	return func(slice []Elt) Accumulator {
		var ans Accumulator
		for _, elt := range slice {
			ans = reduceFn(ans, mapFn(elt))
		}
		return ans
	}
}

func or(x, y bool) bool { return x || y }

func append1[Elt any](slice []Elt, next Elt) []Elt { return append(slice, next) }
