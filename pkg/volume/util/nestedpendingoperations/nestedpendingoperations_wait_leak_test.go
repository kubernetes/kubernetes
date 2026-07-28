package nestedpendingoperations

import (
	"errors"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// This test demonstrates the bug described in kubernetes/kubernetes#136075:
// With exponentialBackOffOnError enabled, an operation that returns an error
// leaves a non-pending entry in the operations slice. Wait() loops on
// len(operations) > 0 and therefore blocks indefinitely, even though there is
// no pending work.
func TestWaitBlocksOnNonPendingOperationAfterError(t *testing.T) {
	g := NewNestedPendingOperations(true)

	op := volumetypes.GeneratedOperations{
		OperationName: "fail-once",
		Run: func() (eventErr, detailedErr error) {
			return nil, errors.New("boom")
		},
	}

	if err := g.Run(v1.UniqueVolumeName("vol"), volumetypes.UniquePodName("pod"), types.NodeName("node"), op); err != nil {
		t.Fatalf("Run() unexpected error: %v", err)
	}

	// Wait until the spawned operation goroutine has updated state: entry present and not pending.
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

	// Now call Wait() in a separate goroutine; it should block because
	// len(operations) > 0, even though no operation is pending.
	done := make(chan struct{})
	go func() {
		g.Wait()
		close(done)
	}()

	select {
	case <-done:
		// If this returns, behavior has changed; make the test fail to capture it.
		t.Fatalf("Wait() returned but expected it to block due to retained non-pending operation")
	case <-time.After(500 * time.Millisecond):
		// Demonstrates the bug: Wait() is blocked. Fail the test to mark the regression.
		t.Fatalf("Wait() blocked due to retained non-pending operation; demonstrates kubernetes#136075")
	}
}
