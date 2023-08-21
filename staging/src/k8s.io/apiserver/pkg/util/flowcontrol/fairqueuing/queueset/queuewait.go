/*
Copyright 2023 The Kubernetes Authors.

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

	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/eventclock"
)

// newQueueWaitContextFactory returns a new instance of queueWaitContextFactory
func newQueueWaitContextFactory(clock eventclock.Interface) *queueWaitFactory {
	return &queueWaitFactory{
		clock:    clock,
		newCtxFn: context.WithTimeout,
	}
}

type queueWaitFactory struct {
	clock eventclock.Interface
	// comes handy to write unit test
	newCtxFn func(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc)
}

// queueWaitContextFactory returns a new context with a deadline of how
// long the request is allowed to wait before it is removed from its
// queue and rejcted.
// The context.CancelFunc returned must never be nil and the caller is
// responsible for calling the CancelFunc function for cleanup.
//   - ctx: the context associated with the request (it may or may
//     not have a deadline).
//   - defaultQueueWaitTime: the default wait duration that is used if
//     the request context does not have any deadline.
func (qw *queueWaitFactory) GetQueueWaitContext(parent context.Context, defaultQueueWaitTime time.Duration) (context.Context, context.CancelFunc) {
	if parent.Err() != nil {
		return parent, func() {}
	}

	if deadline, ok := parent.Deadline(); ok {
		// we will allow the request to wait in the queue for one fourth of the time
		// of its remaining deadline. (This is a new behavior).
		// if the request context does not have any deadline then we default to
		// 'defaultQueueWaitTime' which is 15s today. (This is in keeping with
		// the current behavior)
		queueWaitTime := deadline.Sub(qw.clock.Now()) / 4
		if queueWaitTime <= 0 {
			return parent, func() {}
		}
		return qw.newCtxFn(parent, queueWaitTime)
	}

	return qw.newCtxFn(parent, defaultQueueWaitTime)
}
