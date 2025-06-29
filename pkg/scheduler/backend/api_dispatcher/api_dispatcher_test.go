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
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
)

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
	mergeFn   func(oldCall fwk.APICall) (needsCall bool, err error)
	updateFn  func(obj metav1.Object) (needsCall bool, updatedObj metav1.Object, err error)
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

func (mac *mockAPICall) Merge(oldCall fwk.APICall) (needsCall bool, err error) {
	if mac.mergeFn != nil {
		return mac.mergeFn(oldCall)
	}
	// A default return value indicating the call is still needed after merge.
	return true, nil
}

func (mac *mockAPICall) Update(obj metav1.Object) (needsCall bool, updatedObj metav1.Object, err error) {
	if mac.updateFn != nil {
		return mac.updateFn(obj)
	}
	// A default return value indicating the call is still needed after update.
	return true, obj, nil
}

func TestAPIDispatcherLifecycle(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

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
	var executed, merged, updated bool
	call2 := &mockAPICall{
		uid:      uid,
		callType: mockCallTypeLow,
		executeFn: func(_ context.Context, _ clientset.Interface) error {
			executed = true
			return nil
		},
		mergeFn: func(oldCall fwk.APICall) (bool, error) {
			merged = true
			return true, nil
		},
		updateFn: func(obj metav1.Object) (bool, metav1.Object, error) {
			updated = true
			return true, obj, nil
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
	if !merged {
		t.Errorf("Expected call2's Merge() to be called")
	}
	expectOnFinish(t, onFinishCh1, fwk.ErrCallOverwritten)

	obj := &metav1.ObjectMeta{
		UID: uid,
	}
	_, err := dispatcher.UpdateObject(obj)
	if err != nil {
		t.Fatalf("Unexpected error while updating an object: %v", err)
	}
	if !updated {
		t.Errorf("Expected call2's update to be called")
	}

	// Run should be started earlier, but was delayed on purpose to verify the pre-run state (merging and updating).
	dispatcher.Run(ctx)
	defer dispatcher.Close()

	expectOnFinish(t, onFinishCh2, nil)
	if !executed {
		t.Errorf("Expected call2's Execute() to be called")
	}
}
