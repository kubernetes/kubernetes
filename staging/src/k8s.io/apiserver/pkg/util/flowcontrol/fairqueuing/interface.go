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
	// BeginConfigChange starts the two-step process of updating the
	// configuration.  No change is made until Complete is called.  If
	// `C := X.BeginConstruction(q)` then `C.Complete(d)` returns the
	// same value `X`.  If the QueuingConfig's DesiredNumQueues field
	// is zero then the other queuing-specific config parameters are
	// not changed, so that the queues continue draining as before.
	// In any case, reconfiguration does not discard any queue unless
	// and until it is undesired and empty.
	BeginConfigChange(QueuingConfig) (QueueSetCompleter, error)

	// IsIdle returns a bool indicating whether the QueueSet was idle
	// at the moment of the return.  Idle means the QueueSet has zero
	// requests queued and zero executing.  This bit can change only
	// (1) during a call to StartRequest and (2) during a call to
	// Request::Finish.  In the latter case idleness can only change
	// from false to true.
	IsIdle() bool

	// StartRequest begins the process of handling a request.  If the
	// request gets queued and the number of queues is greater than
	// 1 then Wait uses the given hashValue as the source of entropy
	// as it shuffle-shards the request into a queue.  The descr1 and
	// descr2 values play no role in the logic but appear in log
	// messages.  This method always returns quickly (without waiting
	// for the request to be dequeued).  If this method returns a nil
	// Request value then caller should reject the request and the
	// returned bool indicates whether the QueueSet was idle at the
	// moment of the return.  Otherwise idle==false and the client
	// must call the Wait method of the Request exactly once.
	StartRequest(ctx context.Context, hashValue uint64, fsName string, descr1, descr2 interface{}) (req Request, idle bool)
}

// Request represents the remainder of the handling of one request
type Request interface {
	// Finish determines whether to execute or reject the request and
	// invokes `execute` if the decision is to execute the request.
	// The returned `idle bool` value indicates whether the QueueSet
	// was idle when the value was calculated, but might no longer be
	// accurate by the time the client examines that value.
	Finish(execute func()) (idle bool)
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
