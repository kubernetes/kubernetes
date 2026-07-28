/*
Copyright 2026 The Kubernetes Authors.

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

package nestedpendingoperations

import (
	"errors"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// Demonstrates kubernetes/kubernetes#136075 behavior pre-fix and validates fix:
// When exponentialBackOffOnError is enabled and an operation errors, the entry
// remains but is not pending. Wait() must not block in that state.
func TestWaitReturnsWhenNoPendingOperationsRemainAfterError(t *testing.T) {
	g := NewNestedPendingOperations(true)

	op := volumetypes.GeneratedOperations{
		OperationName: "fail-once",
		OperationFunc: func() volumetypes.OperationContext {
			return volumetypes.NewOperationContext(nil, errors.New("boom"), false)
		},
	}

	if err := g.Run(v1.UniqueVolumeName("vol"), volumetypes.UniquePodName("pod"), types.NodeName("node"), op); err != nil {
		t.Fatalf("Run() unexpected error: %v", err)
	}

	// Wait until the goroutine marks operation as non-pending but entry remains.
	deadline := time.Now().Add(3 * time.Second)
	for {
		if time.Now().After(deadline) {
			t.Fatalf("timeout waiting for operation state to settle")
		}
		npo := g.(*nestedPendingOperations)
		npo.lock.RLock()
		ok := len(npo.operations) > 0 && !npo.operations[0].operationPending
		npo.lock.RUnlock()
		if ok {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	done := make(chan struct{})
	go func() {
		g.Wait()
		close(done)
	}()

	select {
	case <-done:
		// expected: Wait should not block when no pending operations remain
	case <-time.After(1 * time.Second):
		t.Fatalf("Wait() blocked while no pending operations remained")
	}
}
