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

package kubelet

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// contextAwareVolumeManager simulates a volume manager that respects context cancellation
type contextAwareVolumeManager struct {
	volumemanager.VolumeManager
	mu                sync.Mutex
	attachMountCalled chan struct{}
	unmountCalled     chan struct{}
}

func (c *contextAwareVolumeManager) WaitForAttachAndMount(ctx context.Context, pod *v1.Pod) error {
	c.mu.Lock()
	ch := c.attachMountCalled
	c.mu.Unlock()

	if ch != nil {
		close(ch)
	}
	// Block until context is cancelled
	<-ctx.Done()
	return ctx.Err()
}

func (c *contextAwareVolumeManager) WaitForUnmount(ctx context.Context, pod *v1.Pod) error {
	c.mu.Lock()
	ch := c.unmountCalled
	c.mu.Unlock()

	if ch != nil {
		close(ch)
	}
	// Block until context is cancelled
	<-ctx.Done()
	return ctx.Err()
}

// TestSyncPodCancellationDuringVolumeMount verifies that when the context is cancelled
// while SyncPod is waiting for volumes, the cancellation propagates correctly.
func TestSyncPodCancellationDuringVolumeMount(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "test-pod",
			Namespace: "test",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: "test-image",
				},
			},
		},
	}

	pods := []*v1.Pod{pod}
	kl.podManager.SetPods(pods)

	podStatus := &kubecontainer.PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}

	// Create a cancellable context
	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()

	// Replace volume manager with one that blocks on context
	attachMountCalled := make(chan struct{})
	kl.volumeManager = &contextAwareVolumeManager{
		VolumeManager:     kl.volumeManager,
		attachMountCalled: attachMountCalled,
	}

	// Start SyncPod in goroutine
	var err error
	var isTerminal bool
	done := make(chan struct{})
	go func() {
		defer close(done)
		isTerminal, _, err = kl.SyncPod(ctx, kubetypes.SyncPodCreate, pod, nil, podStatus)
	}()

	// Wait for volume operation to start
	select {
	case <-attachMountCalled:
		t.Log("SyncPod is blocked in WaitForAttachAndMount")
	case <-time.After(5 * time.Second):
		t.Fatal("WaitForAttachAndMount was not called")
	}

	// Cancel context
	t.Log("Cancelling context")
	cancel()

	// Verify SyncPod exits with cancellation error
	select {
	case <-done:
		t.Log("SyncPod exited")
	case <-time.After(5 * time.Second):
		t.Fatal("SyncPod did not exit after cancellation")
	}

	require.True(t, wait.Interrupted(err), "expected context cancellation error, got: %v", err)
	assert.False(t, isTerminal, "pod should not be terminal after cancellation")
}

// TestSyncPodCancellationDuringContainerSync verifies that when the context is cancelled
// while the container runtime is syncing, the cancellation propagates correctly through
// kubelet's SyncPod.
func TestSyncPodCancellationDuringContainerSync(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "test-pod",
			Namespace: "test",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: "test-image",
				},
			},
		},
	}

	pods := []*v1.Pod{pod}
	kl.podManager.SetPods(pods)

	podStatus := &kubecontainer.PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}

	// Set FakeRuntime to block on SyncPod until context is cancelled
	fakeRuntime.Lock()
	fakeRuntime.BlockSyncPod = true
	fakeRuntime.Unlock()

	// Create a cancellable context
	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()

	// Start SyncPod in goroutine
	var err error
	var isTerminal bool
	done := make(chan struct{})
	go func() {
		defer close(done)
		isTerminal, _, err = kl.SyncPod(ctx, kubetypes.SyncPodCreate, pod, nil, podStatus)
	}()

	// Give SyncPod time to reach the blocking runtime.SyncPod call
	time.Sleep(100 * time.Millisecond)
	t.Log("Cancelling context while SyncPod is blocked in runtime.SyncPod")

	// Cancel context
	cancel()

	// Verify SyncPod exits with cancellation error
	select {
	case <-done:
		t.Log("SyncPod exited")
	case <-time.After(5 * time.Second):
		t.Fatal("SyncPod did not exit after cancellation")
	}

	require.True(t, wait.Interrupted(err), "expected context cancellation error, got: %v", err)
	assert.False(t, isTerminal, "pod should not be terminal after cancellation")
}

// TestSyncTerminatingPodCancellation verifies that when the context is cancelled
// while KillPod is running, the cancellation propagates correctly through
// SyncTerminatingPod.
func TestSyncTerminatingPodCancellation(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "terminating-pod",
			Namespace: "test",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: "test-image",
				},
			},
		},
	}

	pods := []*v1.Pod{pod}
	kl.podManager.SetPods(pods)

	podStatus := &kubecontainer.PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
		ContainerStatuses: []*kubecontainer.Status{
			{
				ID:    kubecontainer.ContainerID{Type: "test", ID: "container1"},
				Name:  "test-container",
				State: kubecontainer.ContainerStateRunning,
			},
		},
	}

	// Set FakeRuntime to block on KillPod until context is cancelled
	fakeRuntime.Lock()
	fakeRuntime.BlockKillPod = true
	fakeRuntime.Unlock()

	// Create a cancellable context
	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()

	// Start SyncTerminatingPod in goroutine
	var err error
	done := make(chan struct{})
	go func() {
		defer close(done)
		gracePeriod := int64(30)
		err = kl.SyncTerminatingPod(ctx, pod, podStatus, &gracePeriod, nil)
	}()

	// Give SyncTerminatingPod time to reach the blocking runtime.KillPod call
	time.Sleep(100 * time.Millisecond)
	t.Log("Cancelling context while SyncTerminatingPod is blocked in runtime.KillPod")

	// Cancel context
	cancel()

	// Verify SyncTerminatingPod exits with cancellation error
	select {
	case <-done:
		t.Log("SyncTerminatingPod exited")
	case <-time.After(5 * time.Second):
		t.Fatal("SyncTerminatingPod did not exit after cancellation")
	}

	require.True(t, wait.Interrupted(err), "expected context cancellation error, got: %v", err)
}

// TestSyncTerminatedPodCancellation verifies that when the context is cancelled
// while waiting for volumes to unmount, the cancellation propagates correctly.
func TestSyncTerminatedPodCancellation(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "terminated-pod",
			Namespace: "test",
		},
	}

	pods := []*v1.Pod{pod}
	kl.podManager.SetPods(pods)

	podStatus := &kubecontainer.PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}

	// Create a cancellable context
	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()

	// Replace volume manager with one that blocks on context
	unmountCalled := make(chan struct{})
	kl.volumeManager = &contextAwareVolumeManager{
		VolumeManager: kl.volumeManager,
		unmountCalled: unmountCalled,
	}

	// Start SyncTerminatedPod in goroutine
	var err error
	done := make(chan struct{})
	go func() {
		defer close(done)
		err = kl.SyncTerminatedPod(ctx, pod, podStatus)
	}()

	// Wait for unmount to start
	select {
	case <-unmountCalled:
		t.Log("SyncTerminatedPod is blocked in WaitForUnmount")
	case <-time.After(5 * time.Second):
		t.Fatal("WaitForUnmount was not called")
	}

	// Cancel context
	t.Log("Cancelling context")
	cancel()

	// Verify SyncTerminatedPod exits with cancellation error
	select {
	case <-done:
		t.Log("SyncTerminatedPod exited")
	case <-time.After(5 * time.Second):
		t.Fatal("SyncTerminatedPod did not exit after cancellation")
	}

	require.True(t, wait.Interrupted(err), "expected context cancellation error, got: %v", err)
}
