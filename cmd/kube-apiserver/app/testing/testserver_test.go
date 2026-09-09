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

package testing

import (
	"context"
	"errors"
	"testing"

	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestStartTestServerTearDownCauseOnTestCompletion verifies that when a test
// simply ends without calling the returned TearDownFn explicitly, the
// server's internal context gets canceled (with context.Canceled as its
// Err()) and with a cause that indicates normal test completion.
func TestStartTestServerTearDownCauseOnTestCompletion(t *testing.T) {
	var observedErr, observedCause error

	// Run in a sub-test so that its t.Cleanup callbacks (including the
	// one registered by StartTestServer) run to completion before t.Run
	// returns here, without any extra synchronization.
	//
	// WithoutCancel avoids racing ktesting's auto-cancellation against the
	// "test has completed" trigger this test verifies.
	t.Run("server", func(t *testing.T) {
		tCtx := ktesting.Init(t).WithoutCancel()
		_, storageConfig := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)

		instanceOptions := NewDefaultTestServerOptions()
		instanceOptions.observeTearDown = func(ctx context.Context) {
			observedErr = ctx.Err()
			observedCause = context.Cause(ctx)
		}
		if _, err := StartTestServer(tCtx, instanceOptions, nil, storageConfig); err != nil {
			t.Fatalf("failed to start test server: %v", err)
		}
	})

	if !errors.Is(observedErr, context.Canceled) {
		t.Errorf("expected the internal context to be canceled with context.Canceled, got: %v", observedErr)
	}
	if observedCause == nil || observedCause.Error() != "test has completed" {
		t.Errorf("expected cause to indicate test completion, got: %v", observedCause)
	}
}

// TestStartTestServerTearDownCauseOnExplicitTearDown verifies that calling
// the returned TearDownFn explicitly cancels the server's internal context
// (with context.Canceled as its Err()) with a cause that indicates an
// explicit tear-down request.
func TestStartTestServerTearDownCauseOnExplicitTearDown(t *testing.T) {
	var observedErr, observedCause error

	_, storageConfig := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)

	instanceOptions := NewDefaultTestServerOptions()
	instanceOptions.observeTearDown = func(ctx context.Context) {
		observedErr = ctx.Err()
		observedCause = context.Cause(ctx)
	}
	result, err := StartTestServer(t, instanceOptions, nil, storageConfig)
	if err != nil {
		t.Fatalf("failed to start test server: %v", err)
	}
	result.TearDownFn()

	if !errors.Is(observedErr, context.Canceled) {
		t.Errorf("expected the internal context to be canceled with context.Canceled, got: %v", observedErr)
	}
	if observedCause == nil || observedCause.Error() != "tear-down requested" {
		t.Errorf("expected cause to indicate an explicit tear-down request, got: %v", observedCause)
	}
}
