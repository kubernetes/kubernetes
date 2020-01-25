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

// QueueSetFactory is used to create QueueSet objects.  Creation, like
// config update, is done in two phases: the first phase consumes the
// QueuingConfig and the second consumes the DispatchingConfig.  They
// are separated so that errors from the first phase can be found
// before committing to a concurrency allotment for the second.
type QueueSetFactory interface {
	// BeginConstruction does the first phase of creating a QueueSet
	BeginConstruction(QueuingConfig) (QueueSetCompleter, error)
}

// QueueSetCompleter finishes the two-step process of creating or
// reconfiguring a QueueSet
type QueueSetCompleter interface {
	// Complete returns a QueueSet configured by the given
	// dispatching configuration.
	Complete(DispatchingConfig) QueueSet
}

// QueueSet is the abstraction for the queuing and dispatching
// functionality of one non-exempt priority level.  It covers the
// functionality described in the "Assignment to a Queue", "Queuing",
// and "Dispatching" sections of
// https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md
// .  Some day we may have connections between priority levels, but
// today is not that day.
type QueueSet interface {
	// BeginConfigChange starts the two-step process of updating
	// the configuration.  No change is made until Complete is
	// called.  If `C := X.BeginConstruction(q)` then
	// `C.Complete(d)` returns the same value `X`.  If the
	// QueuingConfig's DesiredNumQueues field is zero then the other
	// queuing-specific config parameters are not changed, so that the
	// queues continue draining as before.
	BeginConfigChange(QueuingConfig) (QueueSetCompleter, error)

	// Quiesce controls whether the QueueSet is operating normally or
	// is quiescing.  A quiescing QueueSet drains as normal but does
	// not admit any new requests. Passing a non-nil handler means the
	// system should be quiescing, a nil handler means the system
	// should operate normally. A call to Wait while the system is
	// quiescing will be rebuffed by returning tryAnother=true. If all
	// the queues have no requests waiting nor executing while the
	// system is quiescing then the handler will eventually be called
	// with no locks held (even if the system becomes non-quiescing
	// between the triggering state and the required call).  In Go
	// Memory Model terms, the triggering state happens before the
	// call to the EmptyHandler.
	Quiesce(EmptyHandler)

	// Wait uses the given hashValue as the source of entropy as it
	// shuffle-shards a request into a queue and waits for a decision
	// on what to do with that request.  The descr1 and descr2 values
	// play no role in the logic but appear in log messages.  If
	// tryAnother==true at return then the QueueSet has become
	// undesirable and the client should try to find a different
	// QueueSet to use; execute and afterExecution are irrelevant in
	// this case.  In the terms of the Go Memory Model, there was a
	// call to Quiesce with a non-nil handler that happened before
	// this return from Wait.  Otherwise, if execute then the client
	// should start executing the request and, once the request
	// finishes execution or is canceled, call afterExecution().
	// Otherwise the client should not execute the request and
	// afterExecution is irrelevant.  Canceling the context while the
	// request is waiting in its queue will cut short that wait and
	// cause a return with tryAnother and execute both false; later
	// cancellations are the caller's problem.
	Wait(ctx context.Context, hashValue uint64, descr1, descr2 interface{}) (tryAnother, execute bool, afterExecution func())
}

// QueuingConfig defines the configuration of the queuing aspect of a QueueSet.
type QueuingConfig struct {
	// Name is used to identify a queue set, allowing for descriptive information about its intended use
	Name string

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

// DispatchingConfig defines the configuration of the dispatching aspect of a QueueSet.
type DispatchingConfig struct {
	// ConcurrencyLimit is the maximum number of requests of this QueueSet that may be executing at a time
	ConcurrencyLimit int
}

// EmptyHandler is used to notify the callee when all the queues
// of a QueueSet have been drained.
type EmptyHandler interface {
	// HandleEmpty is called to deliver the notification
	HandleEmpty()
}
