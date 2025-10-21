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
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

func registerAndResetMetrics(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerAsyncAPICalls, true)
	metrics.Register()

	metrics.AsyncAPICallsTotal.Reset()
	metrics.AsyncAPICallDuration.Reset()
	metrics.AsyncAPIPendingCalls.Reset()
}

const (
	mockCallTypeLow  fwk.APICallType = "low"
	mockCallTypeHigh fwk.APICallType = "high"
)

// mockRelevances defines a standard set of relevances for testing purposes.
var mockRelevances = fwk.APICallRelevances{
	mockCallTypeLow:  1,
	mockCallTypeHigh: 2,
}

// mockAPICall is a mock implementation of fwk.APICall for detailed testing.
type mockAPICall struct {
	callType  fwk.APICallType
	uid       types.UID
	executeFn func(ctx context.Context, client clientset.Interface) error
	mergeFn   func(oldCall fwk.APICall) error
	syncFn    func(obj metav1.Object) (metav1.Object, error)
	isNoOpFn  func() bool
}

func (mac *mockAPICall) CallType() fwk.APICallType {
	return mac.callType
}

func (mac *mockAPICall) UID() types.UID {
	return mac.uid
}

func (mac *mockAPICall) Execute(ctx context.Context, client clientset.Interface) error {
	if mac.executeFn != nil {
		return mac.executeFn(ctx, client)
	}
	return nil
}

func (mac *mockAPICall) Merge(oldCall fwk.APICall) error {
	if mac.mergeFn != nil {
		return mac.mergeFn(oldCall)
	}
	return nil
}

func (mac *mockAPICall) Sync(obj metav1.Object) (metav1.Object, error) {
	if mac.syncFn != nil {
		return mac.syncFn(obj)
	}
	return obj, nil
}

func (mac *mockAPICall) IsNoOp() bool {
	if mac.isNoOpFn != nil {
		return mac.isNoOpFn()
	}
	// A default return value indicating the call is still needed.
	return false
}

func TestAPIDispatcherLifecycle(t *testing.T) {
	// Reset all async API metrics
	registerAndResetMetrics(t)

	logger, _ := ktesting.NewTestContext(t)

	uid := types.UID("uid")
	dispatcher := New(fake.NewClientset(), 1, mockRelevances)

	call1 := &mockAPICall{
		uid:      uid,
		callType: mockCallTypeLow,
	}
	onFinishCh1 := make(chan error, 1)
	opts1 := fwk.APICallOptions{
		OnFinish: onFinishCh1,
	}
	var executeCalls, mergeCalls, syncCalls, isNoOpCalls int
	call2 := &mockAPICall{
		uid:      uid,
		callType: mockCallTypeLow,
		executeFn: func(_ context.Context, _ clientset.Interface) error {
			executeCalls++
			return nil
		},
		mergeFn: func(oldCall fwk.APICall) error {
			mergeCalls++
			return nil
		},
		syncFn: func(obj metav1.Object) (metav1.Object, error) {
			syncCalls++
			return obj, nil
		},
		isNoOpFn: func() bool {
			isNoOpCalls++
			return false
		},
	}
	onFinishCh2 := make(chan error, 1)
	opts2 := fwk.APICallOptions{
		OnFinish: onFinishCh2,
	}

	if err := dispatcher.Add(call1, opts1); err != nil {
		t.Fatalf("Unexpected error while adding a call1: %v", err)
	}
	if err := dispatcher.Add(call2, opts2); err != nil {
		t.Fatalf("Unexpected error while adding a call2: %v", err)
	}
	if mergeCalls != 1 {
		t.Errorf("Expected call2's Merge() to be called once, but was %v times", mergeCalls)
	}
	expectOnFinish(t, onFinishCh1, fwk.ErrCallOverwritten)

	obj := &metav1.ObjectMeta{
		UID: uid,
	}
	_, err := dispatcher.SyncObject(obj)
	if err != nil {
		t.Fatalf("Unexpected error while syncing an object: %v", err)
	}
	if syncCalls != 1 {
		t.Errorf("Expected call2's sync to be called once, but was %v times", mergeCalls)
	}

	// Run should be started earlier, but was delayed on purpose to verify the pre-run state (merging and syncing).
	dispatcher.Run(logger)
	defer dispatcher.Close()

	expectOnFinish(t, onFinishCh2, nil)
	if executeCalls != 1 {
		t.Errorf("Expected call2's Execute() to be called once, but was %v times", executeCalls)
	}
	if isNoOpCalls != 2 {
		t.Errorf("Expected call2's IsNoOp() to be called two times, but was %v times", executeCalls)
	}

	// Verify execution metrics
	testutil.AssertVectorCount(t, "scheduler_async_api_call_execution_total", map[string]string{"call_type": "low", "result": "success"}, 1)
	testutil.AssertHistogramTotalCount(t, "scheduler_async_api_call_execution_duration_seconds", map[string]string{"call_type": "low", "result": "success"}, 1)
}
