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
	fwk "k8s.io/kube-scheduler/framework"
)

var queuedAPICallCmpOpts = []cmp.Option{
	cmp.AllowUnexported(queuedAPICall{}, mockAPICall{}),
	// Comparison of function fields is not reliable, so they are ignored.
	cmpopts.IgnoreFields(mockAPICall{}, "executeFn", "mergeFn", "updateFn"),
}

// verifyQueueLen is a test helper to check the length of the callsQueue.
func verifyQueueLen(t *testing.T, cc *callController, len int) {
	t.Helper()

	if got := cc.callsQueue.Len(); got != len {
		t.Errorf("Expected callsQueue to have %d item(s), but has %d", len, got)
	}
}

// verifyCalls is a test helper to check the content of the apiCalls.
func verifyCalls(t *testing.T, cc *callController, calls ...*queuedAPICall) {
	t.Helper()

	if got := len(cc.apiCalls); got != len(calls) {
		t.Errorf("Expected apiCalls to have %d item(s), but has %d item(s)", len(calls), got)
	}
	for _, call := range calls {
		if diff := cmp.Diff(call, cc.apiCalls[call.UID()], queuedAPICallCmpOpts...); diff != "" {
			t.Errorf("API call from apiCalls does not match %v (-want +got):\n%s", call.CallType(), diff)
		}
	}
}

// verifyInFlight is a test helper to check the content of the inFlightEntities.
func verifyInFlight(t *testing.T, cc *callController, uids ...types.UID) {
	t.Helper()

	if got := cc.inFlightEntities.Len(); got != len(uids) {
		t.Errorf("Expected inFlightEntities to have %d item(s), but has %d item(s)", len(uids), got)
	}
	for _, uid := range uids {
		if !cc.inFlightEntities.Has(uid) {
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

func TestCallControllerAdd(t *testing.T) {
	uid1 := types.UID("uid1")
	uid2 := types.UID("uid2")

	t.Run("First call is added without collision", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
			},
		}

		if err := cc.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}
		verifyQueueLen(t, cc, 1)
		verifyCalls(t, cc, call)
	})

	t.Run("Two calls for different objects don't collide", func(t *testing.T) {
		cc := newCallController(mockRelevances)
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

		if err := cc.add(call1); err != nil {
			t.Fatalf("Unexpected error while adding call1: %v", err)
		}
		if err := cc.add(call2); err != nil {
			t.Fatalf("Unexpected error while adding call2: %v", err)
		}
		verifyQueueLen(t, cc, 2)
		verifyCalls(t, cc, call1, call2)
	})

	t.Run("New call overwrites less relevant call", func(t *testing.T) {
		cc := newCallController(mockRelevances)
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

		if err := cc.add(callLow); err != nil {
			t.Fatalf("Unexpected error while adding callLow: %v", err)
		}
		if err := cc.add(callHigh); err != nil {
			t.Fatalf("Unexpected error while adding callHigh: %v", err)
		}
		verifyQueueLen(t, cc, 1)
		verifyCalls(t, cc, callHigh)
		expectOnFinish(t, onFinishCh, fwk.ErrCallOverwritten)
	})

	t.Run("New call is skipped if less relevant", func(t *testing.T) {
		cc := newCallController(mockRelevances)
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

		if err := cc.add(callHigh); err != nil {
			t.Fatalf("Unexpected error while adding callHigh: %v", err)
		}
		err := cc.add(callLow)
		if !errors.Is(err, fwk.ErrCallSkipped) {
			t.Fatalf("Expected callLow to be skipped, but got %v", err)
		}
		verifyQueueLen(t, cc, 1)
		verifyCalls(t, cc, callHigh)
		expectOnFinish(t, onFinishCh, fwk.ErrCallSkipped)
	})

	t.Run("New call merges with old call and skips if no-op", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		onFinishCh1 := make(chan error, 1)
		call1 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				mergeFn: func(fwk.APICall) (bool, error) {
					return true, nil
				},
			},
			onFinish:  onFinishCh1,
			timestamp: time.Now(),
		}
		onFinishCh2 := make(chan error, 1)
		call2 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				mergeFn: func(fwk.APICall) (bool, error) {
					return true, nil
				},
			},
			onFinish:  onFinishCh2,
			timestamp: time.Now().Add(time.Second),
		}
		onFinishCh3 := make(chan error, 1)
		call3 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				mergeFn: func(fwk.APICall) (bool, error) {
					return false, nil // No-op
				},
			},
			onFinish:  onFinishCh3,
			timestamp: time.Now().Add(2 * time.Second),
		}

		if err := cc.add(call1); err != nil {
			t.Fatalf("Unexpected error while adding call1: %v", err)
		}
		if err := cc.add(call2); err != nil {
			t.Fatalf("Unexpected error while adding call2: %v", err)
		}
		verifyQueueLen(t, cc, 1)
		verifyCalls(t, cc, call2)
		expectOnFinish(t, onFinishCh1, fwk.ErrCallOverwritten)

		err := cc.add(call3)
		if !errors.Is(err, fwk.ErrCallSkipped) {
			t.Fatalf("Expected call3 to be skipped, but got %v", err)
		}
		verifyQueueLen(t, cc, 0)
		verifyCalls(t, cc)
		expectOnFinish(t, onFinishCh2, fwk.ErrCallOverwritten)
		expectOnFinish(t, onFinishCh3, fwk.ErrCallSkipped)
	})

	t.Run("Adding a call with equal relevance but different type results in an error", func(t *testing.T) {
		invalidRelevances := fwk.APICallRelevances{
			mockCallTypeLow:  1,
			mockCallTypeHigh: 1,
		}
		cc := newCallController(invalidRelevances)
		callLow := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
			},
		}
		callHigh := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeHigh,
			},
		}

		if err := cc.add(callLow); err != nil {
			t.Fatalf("Unexpected error while adding callLow: %v", err)
		}
		if err := cc.add(callHigh); err == nil {
			t.Fatalf("Expected 'relevance misconfiguration' error while adding callHigh, but got nil")
		}
	})
}

func TestCallControllerPop(t *testing.T) {
	uid1 := types.UID("uid1")
	uid2 := types.UID("uid2")

	t.Run("Calls are popped from the queue in FIFO order", func(t *testing.T) {
		cc := newCallController(mockRelevances)
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
		if err := cc.add(call1); err != nil {
			t.Fatalf("Unexpected error while adding call1: %v", err)
		}
		if err := cc.add(call2); err != nil {
			t.Fatalf("Unexpected error while adding call2: %v", err)
		}

		poppedCall, err := cc.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call1: %v", err)
		}
		if diff := cmp.Diff(call1, poppedCall, queuedAPICallCmpOpts...); diff != "" {
			t.Errorf("First popped call does not patch call1 (-want +got):\n%s", diff)
		}
		verifyQueueLen(t, cc, 1)
		verifyCalls(t, cc, call1, call2)
		verifyInFlight(t, cc, uid1)

		poppedCall, err = cc.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call2: %v", err)
		}
		if diff := cmp.Diff(call2, poppedCall, queuedAPICallCmpOpts...); diff != "" {
			t.Errorf("Second popped call does not match call2 (-want +got):\n%s", diff)
		}
		verifyQueueLen(t, cc, 0)
		verifyCalls(t, cc, call1, call2)
		verifyInFlight(t, cc, uid1, uid2)
	})

	t.Run("Pop blocks when queue is empty and unblocks after add", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		poppedCallCh := make(chan *queuedAPICall)

		go func() {
			poppedCall, popErr := cc.pop()
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
		if err := cc.add(call); err != nil {
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

func TestCallControllerFinalize(t *testing.T) {
	uid := types.UID("uid")

	t.Run("Call details are cleared if there is no waiting call", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid,
				callType: mockCallTypeLow,
			},
			timestamp: time.Now(),
		}
		if err := cc.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}
		poppedCall, err := cc.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call: %v", err)
		}

		cc.finalize(poppedCall)

		verifyQueueLen(t, cc, 0)
		verifyCalls(t, cc)
		verifyInFlight(t, cc)
	})

	t.Run("UID is re-queued if a new call arrived while one was in-flight", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		call1 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid,
				callType: mockCallTypeLow,
			},
			timestamp: time.Now(),
		}
		if err := cc.add(call1); err != nil {
			t.Fatalf("Unexpected error while adding call1: %v", err)
		}
		poppedCall, err := cc.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call1: %v", err)
		}

		call2 := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid,
				callType: mockCallTypeLow,
			},
			timestamp: time.Now().Add(time.Second),
		}
		if err := cc.add(call2); err != nil {
			t.Fatalf("Unexpected error while adding call2: %v", err)
		}

		cc.finalize(poppedCall)

		verifyQueueLen(t, cc, 1)
		verifyCalls(t, cc, call2)
		verifyInFlight(t, cc)
	})
}

func TestCallControllerUpdateObject(t *testing.T) {
	uid1 := types.UID("uid1")
	uid2 := types.UID("uid2")

	t.Run("Object is updated with pending call details", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		obj := &metav1.ObjectMeta{
			UID: uid1,
		}
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				updateFn: func(obj metav1.Object) (bool, metav1.Object, error) {
					obj.SetAnnotations(map[string]string{"updated": "true"})
					return true, obj, nil
				},
			},
		}
		if err := cc.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}

		updatedObj, err := cc.updateObject(obj)
		if err != nil {
			t.Fatalf("Unexpected error while updating object: %v", err)
		}
		verifyQueueLen(t, cc, 1)
		verifyCalls(t, cc, call)
		verifyInFlight(t, cc)

		wantAnnotations := map[string]string{"updated": "true"}
		if diff := cmp.Diff(wantAnnotations, updatedObj.GetAnnotations()); diff != "" {
			t.Errorf("Unexpected annotations (-want +got):\n%s", diff)
		}
	})

	t.Run("Pending call is canceled if update results in no-op", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		obj := &metav1.ObjectMeta{UID: uid1}
		onFinishCh := make(chan error, 1)
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				updateFn: func(o metav1.Object) (bool, metav1.Object, error) {
					return false, o, nil // No-op
				},
			},
			onFinish: onFinishCh,
		}
		if err := cc.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}

		_, err := cc.updateObject(obj)
		if err != nil {
			t.Fatalf("Unexpected error while updating object: %v", err)
		}
		verifyQueueLen(t, cc, 0)
		verifyCalls(t, cc)
		verifyInFlight(t, cc)
		expectOnFinish(t, onFinishCh, fwk.ErrCallSkipped)
	})

	t.Run("Object is not updated if UID is not present in the controller", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		obj := &metav1.ObjectMeta{UID: uid2}
		call := &queuedAPICall{
			APICall: &mockAPICall{
				uid:      uid1,
				callType: mockCallTypeLow,
				updateFn: func(o metav1.Object) (bool, metav1.Object, error) {
					obj.SetAnnotations(map[string]string{"updated": "true"})
					return true, obj, nil
				},
			},
		}
		if err := cc.add(call); err != nil {
			t.Fatalf("Unexpected error while adding call: %v", err)
		}

		newObj, err := cc.updateObject(obj)
		if err != nil {
			t.Fatalf("Unexpected error while updating object: %v", err)
		}
		if diff := cmp.Diff(obj, newObj); diff != "" {
			t.Errorf("Expected object not to be updated (-want +got):\n%s", diff)
		}
	})
}

func TestCallControllerClose(t *testing.T) {
	t.Run("Pop returns nil when controller is closed", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		cc.close()
		poppedCall, err := cc.pop()
		if err != nil {
			t.Fatalf("Unexpected error while popping call: %v", err)
		}
		if poppedCall != nil {
			t.Errorf("Expected popped call to be nil, but got %v", poppedCall)
		}
	})

	t.Run("Pop unblocks and returns nil when controller is closed", func(t *testing.T) {
		cc := newCallController(mockRelevances)
		poppedCallCh := make(chan *queuedAPICall)

		go func() {
			poppedCall, popErr := cc.pop()
			if popErr != nil {
				t.Errorf("Unexpected error while popping call: %v", popErr)
			}
			poppedCallCh <- poppedCall
		}()

		time.Sleep(100 * time.Millisecond)

		cc.close()

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
