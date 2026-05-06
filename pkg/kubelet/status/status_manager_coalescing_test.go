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

package status

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientgofake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

func TestInitContainerStatusCoalescing(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	fakeClient := &clientgofake.Clientset{}
	manager := newTestManager(fakeClient)

	pod := getTestPod()
	pod.Spec.InitContainers = []v1.Container{
		{Name: "init1"},
		{Name: "init2"},
	}
	pod.Spec.Containers = []v1.Container{
		{Name: "main"},
	}

	// 1. Initial status - Pending
	status := v1.PodStatus{
		Phase: v1.PodPending,
		InitContainerStatuses: []v1.ContainerStatus{
			{Name: "init1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
			{Name: "init2", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
		},
		ContainerStatuses: []v1.ContainerStatus{
			{Name: "main", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
		},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)

	// 2. Init1 starts Running (should be deferred)
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)

	if numUpdates := manager.consumeUpdates(ctx); numUpdates != 0 {
		t.Errorf("Expected 0 updates due to deferral, got %d", numUpdates)
	}

	// 3. Init1 terminates successfully (should remain deferred to batch with next init)
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Terminated: &v1.ContainerStateTerminated{ExitCode: 0},
	}
	manager.SetPodStatus(logger, pod, status)

	if numUpdates := manager.consumeUpdates(ctx); numUpdates != 0 {
		t.Errorf("Expected 0 updates due to continued deferral, got %d", numUpdates)
	}

	// 4. Init2 starts Running (should remain deferred)
	status.InitContainerStatuses[1].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)

	if numUpdates := manager.consumeUpdates(ctx); numUpdates != 0 {
		t.Errorf("Expected 0 updates due to continued deferral, got %d", numUpdates)
	}

	// 5. Init2 terminates successfully (Init phase over, must flush immediately)
	status.InitContainerStatuses[1].State = v1.ContainerState{
		Terminated: &v1.ContainerStateTerminated{ExitCode: 0},
	}
	manager.SetPodStatus(logger, pod, status)

	verifyUpdates(t, manager, 1)

	// 6. Main container starts Running
	status.ContainerStatuses[0].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)

	// Should flush the update now
	verifyUpdates(t, manager, 1)

	// Check actions to see what was sent:
	// 1 GET and 1 PATCH for the initial Waiting state.
	// 1 GET and 1 PATCH for the final Init state (which coalesced init1 running/terminated and init2 running).
	// 1 GET and 1 PATCH for the main container running state.
	actions := fakeClient.Actions()
	if len(actions) != 6 {
		t.Errorf("Expected 6 actions, got %d", len(actions))
	}
}

func TestIsEligibleForDeferral(t *testing.T) {
	m := &manager{}

	tests := []struct {
		name              string
		oldStatus         *v1.PodStatus
		newStatus         *v1.PodStatus
		hasActiveDeferral bool
		expected          bool
	}{
		{
			name: "Transition out of Pending",
			newStatus: &v1.PodStatus{
				Phase: v1.PodRunning,
			},
			hasActiveDeferral: true,
			expected:          false,
		},
		{
			name: "Init container failed",
			newStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 1}}},
				},
			},
			hasActiveDeferral: true,
			expected:          false,
		},
		{
			name: "Main container started",
			newStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				ContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Running: &v1.ContainerStateRunning{}}},
				},
			},
			hasActiveDeferral: true,
			expected:          false,
		},
		{
			name: "Has active deferral, no failure conditions",
			newStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Running: &v1.ContainerStateRunning{}}},
				},
			},
			hasActiveDeferral: true,
			expected:          true,
		},
		{
			name: "All init containers completed",
			newStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0}}},
				},
			},
			hasActiveDeferral: true,
			expected:          false,
		},
		{
			name: "No active deferral, init container waiting to running",
			oldStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				},
			},
			newStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Running: &v1.ContainerStateRunning{}}},
				},
			},
			hasActiveDeferral: false,
			expected:          true,
		},
		{
			name: "No active deferral, no state change",
			oldStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				},
			},
			newStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				},
			},
			hasActiveDeferral: false,
			expected:          false,
		},
		{
			name: "No active deferral, init container running to terminated",
			oldStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Running: &v1.ContainerStateRunning{}}},
				},
			},
			newStatus: &v1.PodStatus{
				Phase: v1.PodPending,
				InitContainerStatuses: []v1.ContainerStatus{
					{State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0}}},
				},
			},
			hasActiveDeferral: false,
			expected:          false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := m.isEligibleForDeferral(tt.oldStatus, tt.newStatus, tt.hasActiveDeferral)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestInitContainerStatusCoalescing_Failure(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	fakeClient := &clientgofake.Clientset{}
	manager := newTestManager(fakeClient)

	pod := getTestPod()
	pod.Spec.InitContainers = []v1.Container{{Name: "init1"}}

	status := v1.PodStatus{
		Phase: v1.PodPending,
		InitContainerStatuses: []v1.ContainerStatus{
			{Name: "init1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
		},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)

	// Start Running
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)
	if numUpdates := manager.consumeUpdates(ctx); numUpdates != 0 {
		t.Fatalf("Expected deferral, got %d updates", numUpdates)
	}

	// Fail
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Terminated: &v1.ContainerStateTerminated{ExitCode: 1},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)
}

func TestInitContainerStatusCoalescing_MainContainerStarts(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	fakeClient := &clientgofake.Clientset{}
	manager := newTestManager(fakeClient)

	pod := getTestPod()
	pod.Spec.InitContainers = []v1.Container{{Name: "init1"}}
	pod.Spec.Containers = []v1.Container{{Name: "main1"}}

	status := v1.PodStatus{
		Phase: v1.PodPending,
		InitContainerStatuses: []v1.ContainerStatus{
			{Name: "init1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
		},
		ContainerStatuses: []v1.ContainerStatus{
			{Name: "main1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
		},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)

	// Start Running
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)
	if numUpdates := manager.consumeUpdates(ctx); numUpdates != 0 {
		t.Fatalf("Expected deferral, got %d updates", numUpdates)
	}

	// Main container starts (this happens if init1 finishes instantly and kubelet proceeds)
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Terminated: &v1.ContainerStateTerminated{ExitCode: 0},
	}
	status.ContainerStatuses[0].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)
}

func TestInitContainerStatusCoalescing_PhaseChange(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	fakeClient := &clientgofake.Clientset{}
	manager := newTestManager(fakeClient)

	pod := getTestPod()
	pod.Spec.InitContainers = []v1.Container{{Name: "init1"}}

	status := v1.PodStatus{
		Phase: v1.PodPending,
		InitContainerStatuses: []v1.ContainerStatus{
			{Name: "init1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
		},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)

	// Start Running
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)
	if numUpdates := manager.consumeUpdates(ctx); numUpdates != 0 {
		t.Fatalf("Expected deferral, got %d updates", numUpdates)
	}

	// Phase changes to Failed (e.g. eviction)
	status.Phase = v1.PodFailed
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)
}

func TestInitContainerStatusCoalescing_TimerExpiration(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	fakeClient := &clientgofake.Clientset{}
	manager := newTestManager(fakeClient)

	// Inject a FakeClock so we can instantly fast-forward time
	fakeClock := testingclock.NewFakeClock(time.Now())
	manager.clock = fakeClock

	pod := getTestPod()
	pod.Spec.InitContainers = []v1.Container{{Name: "init1"}}

	status := v1.PodStatus{
		Phase: v1.PodPending,
		InitContainerStatuses: []v1.ContainerStatus{
			{Name: "init1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
		},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)

	// Start Running
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)
	if numUpdates := manager.consumeUpdates(ctx); numUpdates != 0 {
		t.Fatalf("Expected deferral, got %d updates", numUpdates)
	}

	// Wait for the timer's goroutine to register its waiter
	require.Eventually(t, func() bool {
		return fakeClock.HasWaiters()
	}, 5*time.Second, 10*time.Millisecond, "Expected timer to be registered")

	// Instantly advance the virtual clock past the deferral window
	fakeClock.Step(initContainerStatusDeferralWindow + 100*time.Millisecond)

	// Wait for the lightweight timer to fire and queue the update
	require.Eventually(t, func() bool {
		return manager.consumeUpdates(ctx) == 1
	}, 5*time.Second, 10*time.Millisecond, "Expected 1 update after timer expiration")
}

func TestInitContainerStatusCoalescing_AllInitComplete(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	fakeClient := &clientgofake.Clientset{}
	manager := newTestManager(fakeClient)

	pod := getTestPod()
	pod.Spec.InitContainers = []v1.Container{{Name: "init1"}}

	status := v1.PodStatus{
		Phase: v1.PodPending,
		InitContainerStatuses: []v1.ContainerStatus{
			{Name: "init1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
		},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)

	// Start Running
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()},
	}
	manager.SetPodStatus(logger, pod, status)
	if numUpdates := manager.consumeUpdates(ctx); numUpdates != 0 {
		t.Fatalf("Expected deferral, got %d updates", numUpdates)
	}

	// Terminate successfully (This is the LAST init container, so it should flush immediately!)
	status.InitContainerStatuses[0].State = v1.ContainerState{
		Terminated: &v1.ContainerStateTerminated{ExitCode: 0},
	}
	manager.SetPodStatus(logger, pod, status)
	verifyUpdates(t, manager, 1)
}
