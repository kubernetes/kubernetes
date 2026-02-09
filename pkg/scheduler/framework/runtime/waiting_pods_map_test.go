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

package runtime

import (
	"testing"
	"time"

	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestWaitingPodReturnFalseOnAllowed(t *testing.T) {
	tests := []struct {
		name       string
		action     func(wp *waitingPod) bool
		actionName string
	}{
		{
			name: "Preempt returns false on allowed pod",
			action: func(wp *waitingPod) bool {
				return wp.Preempt("preemption-plugin", "preempted")
			},
			actionName: "Preempt",
		},
		{
			name: "Reject returns false on allowed pod",
			action: func(wp *waitingPod) bool {
				return wp.Reject("reject-plugin", "rejected")
			},
			actionName: "Reject",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := st.MakePod().Name("test-pod").UID("test-uid").Obj()
			wp := newWaitingPod(pod, map[string]time.Duration{"plugin1": 10 * time.Second})

			// 1. Simulate Allow from Plugin
			wp.Allow("plugin1")

			// 2. Perform Action
			if success := tt.action(wp); success {
				t.Errorf("Expected %s to return false (failed), but it returned true (success)", tt.actionName)
			}

			// 3. Check what signal the pod received (should be Success from Allow)
			select {
			case status := <-wp.s:
				if !status.IsSuccess() {
					t.Errorf("Expected Pod to stay Allowed (Success), but got status: %v", status)
				}
			default:
				t.Fatal("No status received")
			}
		})
	}
}

func TestWaitingPodMultipleActions(t *testing.T) {
	pod := st.MakePod().Name("test-pod").UID("test-uid").Obj()
	wp := newWaitingPod(pod, map[string]time.Duration{"plugin1": 10 * time.Second})
	startWaitOnPermit := make(chan struct{})
	endWaitOnPermit := make(chan struct{})

	// Simulate WaitOnPermit
	go func() {
		close(startWaitOnPermit)
		<-wp.s
		close(endWaitOnPermit)
	}()

	<-startWaitOnPermit
	// 1. Simulate Allow from Plugin, it should be consumed by WaitOnPermit
	wp.Allow("plugin1")
	<-endWaitOnPermit

	// 2. Simulate Rejection from Plugin
	res := wp.Reject("plugin2", "rejected")
	if res {
		t.Fatalf("Expected reject to fail, but it succeeded")
	}

	// 3. Simulate Preempt from Plugin
	res = wp.Preempt("preemption-plugin", "preempted")
	if res {
		t.Fatalf("Expected preempt to fail, but it succeeded")
	}
}
