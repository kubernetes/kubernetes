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

package fairqueuing

import (
	"time"
)

// QueueSetFactory knows how to make QueueSet objects.  The request
// management filter makes a QueueSet for each priority level.  The
// clock, and any other test-facilitating infrastructure, to use in
// each QueueSet is known to the factory, the client does not need to
// pass this stuff in each call to NewQueueSet.
type QueueSetFactory interface {
	NewQueueSet(name string, concurrencyLimit, numQueues, queueLengthLimit int, requestWaitLimit time.Duration) QueueSet
}

// QueueSet is the abstraction for the queuing and dispatching
// functionality of one non-exempt priority level.  It covers the
// functionality described in the "Assignment to a Queue", "Queuing",
// and "Dispatching" sections of
// https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md
// .  Some day we may have connections between priority levels, but
// today is not that day.
type QueueSet interface {
	// SetConfiguration updates the configuration
	SetConfiguration(concurrencyLimit, desiredNumQueues, queueLengthLimit int, requestWaitLimit time.Duration)

	// Quiesce controls whether this system is quiescing.  Passing a
	// non-nil handler means the system should become quiescent, a nil
	// handler means the system should become non-quiescent.  A call
	// to Wait while the system is quiescent will be rebuffed by
	// returning `quiescent=true`.  If all the queues have no requests
	// waiting nor executing while the system is quiescent then the
	// handler will eventually be called with no locks held (even if
	// the system becomes non-quiescent between the triggering state
	// and the required call).
	//
	// The filter uses this for a priority level that has become
	// undesired, setting a handler that will cause the priority level
	// to eventually be removed from the filter if the filter still
	// wants that.  If the filter later changes its mind and wants to
	// preserve the priority level then the filter can use this to
	// cancel the handler registration.
	Quiesce(EmptyHandler)

	// Wait, in the happy case, shuffle shards the given request into
	// a queue and eventually dispatches the request from that queue.
	// Dispatching means to return with `quiescent==false` and
	// `execute==true`.  In one unhappy case the request is
	// immediately rebuffed with `quiescent==true` (which tells the
	// filter that there has been a timing splinter and the filter
	// re-calcuates the priority level to use); in all other cases
	// `quiescent` will be returned `false` (even if the system is
	// quiescent by then).  In the non-quiescent unhappy cases the
	// request is eventually rejected, which means to return with
	// `execute=false`.  In the happy case the caller is required to
	// invoke the returned `afterExecution` after the request is done
	// executing.  The hash value and hand size are used to do the
	// shuffle sharding.
	Wait(hashValue uint64, handSize int32) (quiescent, execute bool, afterExecution func())
}

// EmptyHandler can be notified that something is empty
type EmptyHandler interface {
	// HandleEmpty is called to deliver the notification
	HandleEmpty()
}
