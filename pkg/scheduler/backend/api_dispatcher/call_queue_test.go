/*
Copyright 2025 The Kubernetes Authors.

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

package apidispatcher

import (
	"errors"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-base/metrics/testutil"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

var queuedAPICallCmpOpts = []cmp.Option{
	cmp.AllowUnexported(queuedAPICall{}, mockAPICall{}),
	// Comparison of function fields is not reliable, so they are ignored.
	cmpopts.IgnoreFields(mockAPICall{}, "executeFn", "mergeFn", "syncFn"),
}

// verifyQueueState is a test helper to check both the queue length and pending call metrics.
func verifyQueueState(t *testing.T, cq *callQueue, expectedPendingCalls map[string]int) {
	t.Helper()

	expectedQueueLen := 0
	// Check pending call metrics
	for callType, expected := range expectedPendingCalls {
		expectedQueueLen += expected

		value, err := testutil.GetGaugeMetricValue(metrics.AsyncAPIPendingCalls.WithLabelValues(callType))
		if err != nil {
			t.Errorf("Failed to get pending calls metric for %s: %v", callType, err)
		} else if int(value) != expected {
			t.Errorf("Expected pending calls metric for %s to be %d, got %d", callType, expected, int(value))
		}
	}

	if got := cq.callsQueue.Len(); got != expectedQueueLen {
		t.Errorf("Expected callsQueue to have %d item(s), but has %d", expectedQueueLen, got)
	}
}

// verifyCalls is a test helper to check the content of the apiCalls.
func verifyCalls(t *testing.T, cq *callQueue, calls ...*queuedAPICall) {
	t.Helper()

	if got := len(cq.apiCalls); got != len(calls) {
		t.Errorf("Expected apiCalls to have %d item(s), but has %d item(s)", len(calls), got)
	}
	for _, call := range calls {
		if diff := cmp.Diff(call, cq.apiCalls[call.UID()], queuedAPICallCmpOpts...); diff != "" {
			t.Errorf("API call from apiCalls does not match %v (-want +got):\n%s", call.CallType(), diff)
		}
	}
}

// verifyInFlight is a test helper to check the content of the inFlightEntities.
func verifyInFlight(t *testing.T, cq *callQueue, uids ...types.UID) {
	t.Helper()

	if got := cq.inFlightEntities.Len(); got != len(uids) {
		t.Errorf("Expected inFlightEntities to have %d item(s), but has %d item(s)", len(uids), got)
	}
	for _, uid := range uids {
		if !cq.inFlightEntities.Has(uid) {
			t.Errorf("Expected %v in inFlightEntities", uid)
		}
	}
}

// expectOnFinish is a test helper that waits for an error on a channel and checks if it matches the expected error.
func expectOnFinish(t *testing.T, onFinish <-chan error, expectedErr error) {
	t.Helper()

	select {
	case err := <-onFinish:
		if !errors.Is(err, expectedErr) {
			t.Errorf("Expected call to return %q, but got %v", expectedErr, err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected error on onFinish, but got none")
	}
}

func TestCallQueueAdd(t *testing.T) {
	uid1 := types.UID("uid1")
	uid2 := types.UID("uid2")

	t.Run("First call is added without collision", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
			},
		}

		if err := cq.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"low": 1})
		verifyCalls(t, cq, call)
	})

	t.Run("No-op call is skipped", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		onFinishCh := make(chan error, 1)
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				isNoOpFn: func() bool {
					return true
				},
			},
			onFinish: onFinishCh,
		}

		err := cq.add(call)
		if !errors.Is(err, fwk.ErrCallSkipped) {
			t.Fatalf("Expected call to be skipped, but got %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"low": 0})
		verifyCalls(t, cq)
		expectOnFinish(t, onFinishCh, fwk.ErrCallSkipped)
	})

	t.Run("Two calls for different objects don't collide", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		call1 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
			},
		}
		call2 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid2,
				callType: mockCallTypeHigh,
			},
		}

		if err := cq.add(call1); err != nil {
			t.Fatalf("Unexpected error while adding call1: %v", err)
		}
		if err := cq.add(call2); err != nil {
			t.Fatalf("Unexpected error while adding call2: %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"low": 1, "high": 1})
		verifyCalls(t, cq, call1, call2)
	})

	t.Run("New call overwrites less relevant call", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		onFinishCh := make(chan error, 1)
		callLow := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
			},
			onFinish: onFinishCh,
		}
		callHigh := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeHigh,
			},
		}

		if err := cq.add(callLow); err != nil {
			t.Fatalf("Unexpected error while adding callLow: %v", err)
		}
		if err := cq.add(callHigh); err != nil {
			t.Fatalf("Unexpected error while adding callHigh: %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"high": 1})
		verifyCalls(t, cq, callHigh)
		expectOnFinish(t, onFinishCh, fwk.ErrCallOverwritten)
	})

	t.Run("New call is skipped if less relevant", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		onFinishCh := make(chan error, 1)
		callLow := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
			},
			onFinish: onFinishCh,
		}
		callHigh := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeHigh,
			},
		}

		if err := cq.add(callHigh); err != nil {
			t.Fatalf("Unexpected error while adding callHigh: %v", err)
		}
		err := cq.add(callLow)
		if !errors.Is(err, fwk.ErrCallSkipped) {
			t.Fatalf("Expected callLow to be skipped, but got %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"high": 1})
		verifyCalls(t, cq, callHigh)
		expectOnFinish(t, onFinishCh, fwk.ErrCallSkipped)
	})

	t.Run("New call merges with old call and skips if no-op", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		onFinishCh1 := make(chan error, 1)
		call1 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				mergeFn: func(fwk.APICall) error {
					return nil
				},
			},
			onFinish: onFinishCh1,
			callID:   0,
		}
		onFinishCh2 := make(chan error, 1)
		call2 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				mergeFn: func(fwk.APICall) error {
					return nil
				},
			},
			onFinish: onFinishCh2,
			callID:   1,
		}
		onFinishCh3 := make(chan error, 1)
		isNoOp := false
		call3 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				mergeFn: func(fwk.APICall) error {
					isNoOp = true
					return nil
				},
				isNoOpFn: func() bool {
					return isNoOp
				},
			},
			onFinish: onFinishCh3,
			callID:   2,
		}

		if err := cq.add(call1); err != nil {
			t.Fatalf("Unexpected error while adding call1: %v", err)
		}
		if err := cq.add(call2); err != nil {
			t.Fatalf("Unexpected error while adding call2: %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"low": 1})
		verifyCalls(t, cq, call2)
		expectOnFinish(t, onFinishCh1, fwk.ErrCallOverwritten)

		err := cq.add(call3)
		if !errors.Is(err, fwk.ErrCallSkipped) {
			t.Fatalf("Expected call3 to be skipped, but got %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"low": 0})
		verifyCalls(t, cq)
		expectOnFinish(t, onFinishCh2, fwk.ErrCallOverwritten)
		expectOnFinish(t, onFinishCh3, fwk.ErrCallSkipped)
	})
}

func TestCallQueuePop(t *testing.T) {
	uid1 := types.UID("uid1")
	uid2 := types.UID("uid2")

	t.Run("Calls are popped from the queue in FIFO order", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		call1 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
			},
		}
		call2 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid2,
				callType: mockCallTypeLow,
			},
		}
		if err := cq.add(call1); err != nil {
			t.Fatalf("Unexpected error while adding call1: %v", err)
		}
		if err := cq.add(call2); err != nil {
			t.Fatalf("Unexpected error while adding call2: %v", err)
		}

		// Verify pending calls after adding
		verifyQueueState(t, cq, map[string]int{"low": 2})

		poppedCall, err := cq.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call1: %v", err)
		}
		if diff := cmp.Diff(call1, poppedCall, queuedAPICallCmpOpts...); diff != "" {
			t.Errorf("First popped call does not patch call1 (-want +got):\n%s", diff)
		}
		verifyQueueState(t, cq, map[string]int{"low": 1})
		verifyCalls(t, cq, call1, call2)
		verifyInFlight(t, cq, uid1)

		poppedCall, err = cq.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call2: %v", err)
		}
		if diff := cmp.Diff(call2, poppedCall, queuedAPICallCmpOpts...); diff != "" {
			t.Errorf("Second popped call does not match call2 (-want +got):\n%s", diff)
		}
		verifyQueueState(t, cq, map[string]int{"low": 0})
		verifyCalls(t, cq, call1, call2)
		verifyInFlight(t, cq, uid1, uid2)
	})

	t.Run("Pop blocks when queue is empty and unblocks after add", func(t *testing.T) {
		cq := newCallQueue(mockRelevances)
		poppedCallCh := make(chan *queuedAPICall)

		go func() {
			poppedCall, popErr := cq.pop()
			if popErr != nil {
				t.Errorf("Unexpected error while popping call: %v", popErr)
			}
			poppedCallCh <- poppedCall
		}()

		time.Sleep(100 * time.Millisecond)

		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
			},
		}
		if err := cq.add(call); err != nil {
			t.Errorf("Unexpected error while adding call: %v", err)
		}

		select {
		case poppedCall := <-poppedCallCh:
			if diff := cmp.Diff(call, poppedCall, queuedAPICallCmpOpts...); diff != "" {
				t.Errorf("Popped call does not match added call (-want +got):\n%s", diff)
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatal("Pop() should have returned an added call, but it timed out")
		}
	})
}

func TestCallQueueFinalize(t *testing.T) {
	uid := types.UID("uid")

	t.Run("Call details are cleared if there is no waiting call", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid,
				callType: mockCallTypeLow,
			},
			callID: 0,
		}
		if err := cq.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}
		poppedCall, err := cq.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call: %v", err)
		}

		cq.finalize(poppedCall)
		verifyQueueState(t, cq, map[string]int{"low": 0})
		verifyCalls(t, cq)
		verifyInFlight(t, cq)
	})

	t.Run("UID is re-queued if a new call arrived while one was in-flight", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		call1 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid,
				callType: mockCallTypeLow,
			},
			callID: 0,
		}
		if err := cq.add(call1); err != nil {
			t.Fatalf("Unexpected error while adding call1: %v", err)
		}
		poppedCall, err := cq.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call1: %v", err)
		}

		call2 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid,
				callType: mockCallTypeLow,
			},
			callID: 1,
		}
		if err := cq.add(call2); err != nil {
			t.Fatalf("Unexpected error while adding call2: %v", err)
		}

		cq.finalize(poppedCall)

		verifyQueueState(t, cq, map[string]int{"low": 1})
		verifyCalls(t, cq, call2)
		verifyInFlight(t, cq)
	})
}

func TestCallQueueSyncObject(t *testing.T) {
	uid1 := types.UID("uid1")
	uid2 := types.UID("uid2")

	t.Run("Object is synced with pending call details", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		obj := &metav1.ObjectMeta{
			UID: uid1,
		}
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				syncFn: func(obj metav1.Object) (metav1.Object, error) {
					obj.SetAnnotations(map[string]string{"synced": "true"})
					return obj, nil
				},
			},
		}
		if err := cq.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}

		syncedObj, err := cq.syncObject(obj)
		if err != nil {
			t.Fatalf("Unexpected error while syncing object: %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"low": 1})
		verifyCalls(t, cq, call)
		verifyInFlight(t, cq)

		wantAnnotations := map[string]string{"synced": "true"}
		if diff := cmp.Diff(wantAnnotations, syncedObj.GetAnnotations()); diff != "" {
			t.Errorf("Unexpected annotations (-want +got):\n%s", diff)
		}
	})

	t.Run("Pending call is canceled if sync results in no-op", func(t *testing.T) {
		registerAndResetMetrics(t)

		cq := newCallQueue(mockRelevances)
		obj := &metav1.ObjectMeta{UID: uid1}
		onFinishCh := make(chan error, 1)
		isNoOp := false
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				syncFn: func(o metav1.Object) (metav1.Object, error) {
					isNoOp = true
					return o, nil
				},
				isNoOpFn: func() bool {
					return isNoOp
				},
			},
			onFinish: onFinishCh,
		}
		if err := cq.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}

		_, err := cq.syncObject(obj)
		if err != nil {
			t.Fatalf("Unexpected error while syncing object: %v", err)
		}
		verifyQueueState(t, cq, map[string]int{"low": 0})
		verifyCalls(t, cq)
		verifyInFlight(t, cq)
		expectOnFinish(t, onFinishCh, fwk.ErrCallSkipped)
	})

	t.Run("Object is not synced if UID is not present in the controller", func(t *testing.T) {
		cq := newCallQueue(mockRelevances)
		obj := &metav1.ObjectMeta{UID: uid2}
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				syncFn: func(o metav1.Object) (metav1.Object, error) {
					obj.SetAnnotations(map[string]string{"synced": "true"})
					return obj, nil
				},
			},
		}
		if err := cq.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}

		newObj, err := cq.syncObject(obj)
		if err != nil {
			t.Fatalf("Unexpected error while syncing object: %v", err)
		}
		if diff := cmp.Diff(obj, newObj); diff != "" {
			t.Errorf("Expected object not to be synced (-want +got):\n%s", diff)
		}
	})
}

func TestCallQueueClose(t *testing.T) {
	t.Run("Pop returns nil when controller is closed", func(t *testing.T) {
		cq := newCallQueue(mockRelevances)
		cq.close()
		poppedCall, err := cq.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call: %v", err)
		}
		if poppedCall != nil {
			t.Errorf("Expected popped call to be nil, but got %v", poppedCall)
		}
	})

	t.Run("Pop unblocks and returns nil when controller is closed", func(t *testing.T) {
		cq := newCallQueue(mockRelevances)
		poppedCallCh := make(chan *queuedAPICall)

		go func() {
			poppedCall, popErr := cq.pop()
			if popErr != nil {
				t.Errorf("Unexpected error while popping call: %v", popErr)
			}
			poppedCallCh <- poppedCall
		}()

		time.Sleep(100 * time.Millisecond)

		cq.close()

		select {
		case poppedCall := <-poppedCallCh:
			if poppedCall != nil {
				t.Errorf("Expected popped call to be nil, but got %v", poppedCall)
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatal("Pop() should have been unblocked by close(), but it remained blocked")
		}
	})
}
