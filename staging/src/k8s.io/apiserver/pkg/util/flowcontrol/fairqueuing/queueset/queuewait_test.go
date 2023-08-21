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
	"testing"
	"time"

	testeventclock "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/eventclock"
)

func TestGetQueueWaitContext(t *testing.T) {
	tests := []struct {
		name                      string
		defaultQueueWaitTime      time.Duration
		parent                    func(t time.Time) (context.Context, context.CancelFunc)
		queueWaitCtxExpected      bool
		queueWaitDeadlineExpected time.Duration
	}{
		{
			name:                 "parent context has already expired",
			defaultQueueWaitTime: 15 * time.Second,
			parent: func(_ time.Time) (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				return ctx, cancel
			},
			queueWaitCtxExpected: false,
		},
		{
			name:                 "parent context has a deadline, queue wait deadline should be approximately one fourth",
			defaultQueueWaitTime: 30 * time.Second,
			parent: func(t time.Time) (context.Context, context.CancelFunc) {
				return context.WithDeadline(context.Background(), t.Add(40*time.Second))
			},
			queueWaitCtxExpected:      true,
			queueWaitDeadlineExpected: 10 * time.Second,
		},
		{
			name:                 "parent does not have any deadline",
			defaultQueueWaitTime: 15 * time.Second,
			parent: func(_ time.Time) (context.Context, context.CancelFunc) {
				return context.WithCancel(context.Background())
			},
			queueWaitCtxExpected:      true,
			queueWaitDeadlineExpected: 15 * time.Second,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			now := time.Now()
			parent, cancel := test.parent(now)
			defer cancel()

			clock, _ := testeventclock.NewFake(now, 0, nil)
			qw := newQueueWaitContextFactory(clock)

			var queueWaitTimeGot time.Duration
			var parentGot context.Context
			delegatedCtxFn := qw.newCtxFn
			qw.newCtxFn = func(parent context.Context, queueWaitTime time.Duration) (context.Context, context.CancelFunc) {
				parentGot = parent
				queueWaitTimeGot = queueWaitTime
				return delegatedCtxFn(parent, queueWaitTime)
			}
			queueWaitCtxGot, cancelGot := qw.GetQueueWaitContext(parent, test.defaultQueueWaitTime)
			if cancelGot == nil {
				t.Errorf("Expected a non nil context.CancelFunc")
				return
			}
			defer cancelGot()

			switch {
			case test.queueWaitCtxExpected:
				if _, ok := queueWaitCtxGot.Deadline(); !ok {
					t.Errorf("Expected the context to have a deadline")
				}
				if test.queueWaitDeadlineExpected != queueWaitTimeGot {
					t.Errorf("Expected the context with a deadline of %s, but got: %s", test.queueWaitDeadlineExpected, queueWaitTimeGot)
				}
				if parent != parentGot {
					t.Errorf("Expected the parent context to be used")
				}

			default:
				if queueWaitCtxGot != parent {
					t.Errorf("Expected the parent context to be returned")
				}
				if parentGot != nil {
					t.Errorf("Did not expect the new context func to be invoked")
				}
			}
		})
	}
}
