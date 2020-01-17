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
	"context"
	"time"
)

// QueueSetFactory is used to create QueueSet objects.
type QueueSetFactory interface {
	NewQueueSet(config QueueSetConfig) (QueueSet, error)
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
	SetConfiguration(QueueSetConfig) error

	// Quiesce controls whether the QueueSet is operating normally or is quiescing.
	// A quiescing QueueSet drains as normal but does not admit any
	// new requests. Passing a non-nil handler means the system should
	// be quiescing, a nil handler means the system should operate
	// normally. A call to Wait while the system is quiescing
	// will be rebuffed by returning tryAnother=true. If all the
	// queues have no requests waiting nor executing while the system
	// is quiescing then the handler will eventually be called with no
	// locks held (even if the system becomes non-quiescing between the
	// triggering state and the required call).
	Quiesce(EmptyHandler)

	// Wait uses the given hashValue as the source of entropy as it
	// shuffle-shards a request into a queue and waits for a decision
	// on what to do with that request.  The descr1 and descr2 values
	// play no role in the logic but appear in log messages.  If
	// tryAnother==true at return then the QueueSet has become
	// undesirable and the client should try to find a different
	// QueueSet to use; execute and afterExecution are irrelevant in
	// this case.  Otherwise, if execute then the client should start
	// executing the request and, once the request finishes execution
	// or is canceled, call afterExecution().  Otherwise the client
	// should not execute the request and afterExecution is
	// irrelevant.
	Wait(ctx context.Context, hashValue uint64, descr1, descr2 interface{}) (tryAnother, execute bool, afterExecution func())
}

// QueueSetConfig defines the configuration of a QueueSet.
type QueueSetConfig struct {
	// Name is used to identify a queue set, allowing for descriptive information about its intended use
	Name string
	// ConcurrencyLimit is the maximum number of requests of this QueueSet that may be executing at a time
	ConcurrencyLimit int
	// DesiredNumQueues is the number of queues that the API says
	// should exist now.  This may be zero, in which case
	// QueueLengthLimit, HandSize, and RequestWaitLimit are ignored.
	DesiredNumQueues int
	// QueueLengthLimit is the maximum number of requests that may be waiting in a given queue at a time
	QueueLengthLimit int
	// HandSize is a parameter of shuffle sharding.  Upon arrival of a request, a queue is chosen by randomly
	// dealing a "hand" of this many queues and then picking one of minimum length.
	HandSize int
	// RequestWaitLimit is the maximum amount of time that a request may wait in a queue.
	// If, by the end of that time, the request has not been dispatched then it is rejected.
	RequestWaitLimit time.Duration
}

// EmptyHandler is used to notify the callee when all the queues
// of a QueueSet have been drained.
type EmptyHandler interface {
	// HandleEmpty is called to deliver the notification
	HandleEmpty()
}
