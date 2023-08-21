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

package testing

import (
	"context"
	"time"

	testingpromise "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/promise"
)

func NewFakeQueueWait() fakeQueueWait {
	return fakeQueueWait{}
}

type fakeQueueWait struct{}

func (qw fakeQueueWait) GetQueueWaitContext(parent context.Context, defaultQueueWaitTime time.Duration) (context.Context, context.CancelFunc) {
	// if the context already has a queue wait time associated then
	// it takes precedence
	if _, ok := testingpromise.GetQueueWaitTimeFromContext(parent); ok {
		return parent, func() {}
	}
	// the tests use context.Background() as the context associated with the requests
	if defaultQueueWaitTime == 0 {
		// the test does not want to set a context with queue wait time
		return parent, func() {}
	}

	ctx := testingpromise.NewQueueWaitTimeWithContext(parent, defaultQueueWaitTime)
	return ctx, func() {}
}
