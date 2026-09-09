/*
Copyright The Kubernetes Authors.

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

package framework

import (
	"context"
	"errors"
	"testing"
	"time"

	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestStartTestServerTearDownCauseOnTestCompletion verifies that when a test
// simply ends without calling the returned TearDownFunc explicitly, the
// server's internal context gets canceled (with context.Canceled as its
// Err()) and with a cause that indicates normal test completion.
func TestStartTestServerTearDownCauseOnTestCompletion(t *testing.T) {
	var observedErr, observedCause error

	// Run in a sub-test so that its t.Cleanup callbacks (including the
	// one registered by StartTestServer) run to completion before t.Run
	// returns here, without any extra synchronization.
	//
	// context.Background() (not a TContext derived from t) avoids racing
	// the auto-cancellation path against the "test has completed" trigger
	// this test verifies.
	t.Run("server", func(t *testing.T) {
		StartTestServer(context.Background(), t, TestServerSetup{
			observeTearDown: func(ctx context.Context) {
				observedErr = ctx.Err()
				observedCause = context.Cause(ctx)
			},
		})
	})

	if !errors.Is(observedErr, context.Canceled) {
		t.Errorf("expected the internal context to be canceled with context.Canceled, got: %v", observedErr)
	}
	if observedCause == nil || observedCause.Error() != "test has completed" {
		t.Errorf("expected cause to indicate test completion, got: %v", observedCause)
	}
}

// TestStartTestServerTearDownCauseOnExplicitTearDown verifies that calling
// the returned TearDownFunc explicitly cancels the server's internal context
// (with context.Canceled as its Err()) with a cause that indicates an
// explicit tear-down request.
func TestStartTestServerTearDownCauseOnExplicitTearDown(t *testing.T) {
	var observedErr, observedCause error

	tCtx := ktesting.Init(t)
	_, _, tearDownFn := StartTestServer(tCtx, t, TestServerSetup{
		observeTearDown: func(ctx context.Context) {
			observedErr = ctx.Err()
			observedCause = context.Cause(ctx)
		},
	})
	tearDownFn()

	if !errors.Is(observedErr, context.Canceled) {
		t.Errorf("expected the internal context to be canceled with context.Canceled, got: %v", observedErr)
	}
	if observedCause == nil || observedCause.Error() != "tear-down requested" {
		t.Errorf("expected cause to indicate an explicit tear-down request, got: %v", observedCause)
	}
}

// TestStartTestServerTearDownCauseOnParentCancellation is a regression test
// for https://github.com/kubernetes/kubernetes/pull/141545#discussion_r3901822871:
// when the context passed into StartTestServer gets canceled with some
// application-specific cause *after* the server has started, that same cause
// must be the one which ends up canceling the server's internal context
// (whose Err() must still be context.Canceled). The cause must not get
// replaced by a generic, hard-coded error.
func TestStartTestServerTearDownCauseOnParentCancellation(t *testing.T) {
	const sentinel = "sentinel: parent context canceled after start"
	type observation struct {
		err   error
		cause error
	}
	observationCh := make(chan observation, 1)

	tCtx := ktesting.Init(t)

	_, _, _ = StartTestServer(tCtx, t, TestServerSetup{
		observeTearDown: func(ctx context.Context) {
			select {
			case observationCh <- observation{err: ctx.Err(), cause: context.Cause(ctx)}:
			default:
			}
		},
	})

	// Cancellation is propagated asynchronously (context.AfterFunc), so
	// tear-down may not have completed by the time this call returns.
	tCtx.Cancel(sentinel)

	select {
	case obs := <-observationCh:
		if !errors.Is(obs.err, context.Canceled) {
			t.Errorf("expected the internal context to be canceled with context.Canceled, got: %v", obs.err)
		}
		if obs.cause == nil || obs.cause.Error() != sentinel {
			t.Errorf("expected the sentinel error to be preserved as the tear-down cause, got: %v", obs.cause)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("tear-down was not observed after canceling the parent context")
	}
}
